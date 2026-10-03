"""Tests for the calibrators (Phase 7A, Step 9)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from airline_disruption.calibration import methods as cm


def simulated(n: int = 200_000, a: float = -0.4, b: float = 0.6, seed: int = 0):
    """Scores from a model that is over-confident by a known amount.

    The true event probability is expit(a + b * logit(score)). Slope b < 1 means the raw score is too spread out.
    """
    rng = np.random.default_rng(seed)
    score = np.clip(expit(rng.normal(-3.6, 1.0, n)), 1e-4, 0.9)
    y = (rng.random(n) < expit(a + b * logit(score))).astype("int8")
    return score, y


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def test_scores_must_be_probabilities() -> None:
    for bad in ([0.1, np.nan], [0.1, np.inf], [-0.1, 0.5], [0.5, 1.2], []):
        with pytest.raises(ValueError):
            cm.check_scores(bad)
    with pytest.raises(ValueError, match="one-dimensional"):
        cm.check_scores(np.zeros((2, 2)))


def test_labels_must_be_binary_with_two_classes_and_matching_length() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        cm.check_labels([0, 1, 2])
    with pytest.raises(ValueError, match="one class"):
        cm.check_labels([0, 0, 0])
    with pytest.raises(ValueError, match="same length"):
        cm.check_labels([0, 1], expected_length=3)


def test_an_unfitted_calibrator_refuses_to_transform_and_unknown_methods_are_rejected() -> None:
    for method in cm.METHODS:
        with pytest.raises(ValueError, match="not been fitted"):
            cm.make_calibrator(method).transform([0.1, 0.2])
    with pytest.raises(ValueError, match="Unknown calibration method"):
        cm.make_calibrator("beta")


# ---------------------------------------------------------------------------
# Logistic helper
# ---------------------------------------------------------------------------
def test_fit_logistic_recovers_known_intercept_and_slope() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(-3.0, 1.2, 400_000)
    y = (rng.random(len(z)) < expit(0.5 + 1.7 * z)).astype("int8")
    a, b = cm.fit_logistic(z, y, slope=True)
    assert a == pytest.approx(0.5, abs=0.06) and b == pytest.approx(1.7, abs=0.03)


def test_fit_logistic_with_a_fixed_slope_recovers_the_level_shift() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(-3.5, 1.0, 400_000)
    y = (rng.random(len(z)) < expit(0.35 + z)).astype("int8")
    a, slope = cm.fit_logistic(z, y, slope=False)
    assert slope == 1.0 and a == pytest.approx(0.35, abs=0.05)


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------
def test_platt_recovers_the_true_recalibration_line() -> None:
    score, y = simulated()
    platt = cm.PlattCalibrator().fit(score, y)
    assert platt.slope == pytest.approx(0.6, abs=0.03)
    assert platt.intercept == pytest.approx(-0.4, abs=0.12)
    assert platt.describe()["fit_rows"] == len(y)


def test_platt_never_changes_the_ranking() -> None:
    score, y = simulated(50_000)
    calibrated = cm.PlattCalibrator().fit(score, y).transform(score)
    order_before = np.argsort(score, kind="stable")
    assert (np.diff(calibrated[order_before]) >= 0).all()  # monotone: same order of flights


def test_platt_rejects_a_scorer_that_ranks_backwards() -> None:
    rng = np.random.default_rng(3)
    score = np.clip(expit(rng.normal(-3.0, 1.0, 100_000)), 1e-4, 0.9)
    y = (rng.random(len(score)) < expit(-1.0 - 0.8 * logit(score))).astype("int8")  # higher score, LOWER risk
    with pytest.raises(ValueError, match="not positive"):
        cm.PlattCalibrator().fit(score, y)


def test_platt_fixes_the_average_level_when_the_shape_is_right() -> None:
    score, y = simulated(300_000, a=0.5, b=1.0)  # the model is right in shape but 65% too low on average
    calibrated = cm.PlattCalibrator().fit(score, y).transform(score)
    assert score.mean() < 0.8 * y.mean()
    assert calibrated.mean() == pytest.approx(y.mean(), rel=0.03)


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------
def test_isotonic_is_non_decreasing_and_stays_inside_the_probability_floor() -> None:
    score, y = simulated(60_000)
    calibrator = cm.IsotonicCalibrator().fit(score, y)
    grid = np.linspace(0.0, 1.0, 5_000)
    out = calibrator.transform(grid)
    assert (np.diff(out) >= -1e-15).all()
    assert out.min() >= cm.PROBABILITY_FLOOR and out.max() <= 1.0 - cm.PROBABILITY_FLOOR
    assert calibrator.describe()["steps"] > 5


def test_isotonic_clips_scores_outside_the_fitted_range_to_the_nearest_fitted_value() -> None:
    score = np.linspace(0.02, 0.10, 4_000)
    rng = np.random.default_rng(4)
    y = (rng.random(len(score)) < score).astype("int8")
    calibrator = cm.IsotonicCalibrator().fit(score, y)
    low, high = calibrator.transform([0.0001, 0.02])[0], calibrator.transform([0.9, 0.10])[0]
    assert low == pytest.approx(calibrator.transform([0.02])[0])
    assert high == pytest.approx(calibrator.transform([0.10])[0])


def test_isotonic_merges_scores_into_ties_which_platt_does_not() -> None:
    score, y = simulated(80_000)
    isotonic = cm.IsotonicCalibrator().fit(score, y).transform(score)
    platt = cm.PlattCalibrator().fit(score, y).transform(score)
    assert len(np.unique(isotonic)) < len(np.unique(score))
    assert len(np.unique(platt)) >= 0.99 * len(np.unique(score))


def test_isotonic_recovers_a_nonlinear_truth_that_platt_cannot() -> None:
    """The true probability is a step: 2% below a score of 0.05 and 12% above it. Platt's smooth curve misses it."""
    rng = np.random.default_rng(5)
    score = rng.uniform(0.005, 0.10, 300_000)
    truth = np.where(score < 0.05, 0.02, 0.12)
    y = (rng.random(len(score)) < truth).astype("int8")
    iso = cm.IsotonicCalibrator().fit(score, y).transform(score)
    pla = cm.PlattCalibrator().fit(score, y).transform(score)
    assert np.mean((iso - truth) ** 2) < 0.5 * np.mean((pla - truth) ** 2)


def test_isotonic_can_output_high_probabilities() -> None:
    """The upper bound must be 1, not something smaller: a high-risk group with an 80% event rate stays near 80%."""
    rng = np.random.default_rng(6)
    score = rng.uniform(0.01, 0.9, 100_000)
    truth = np.where(score < 0.5, 0.05, 0.8)
    y = (rng.random(len(score)) < truth).astype("int8")
    out = cm.IsotonicCalibrator().fit(score, y).transform(score)
    assert out[score > 0.6].mean() == pytest.approx(0.8, abs=0.03)


# ---------------------------------------------------------------------------
# Identity and the factory
# ---------------------------------------------------------------------------
def test_uncalibrated_returns_the_scores_unchanged() -> None:
    score, y = simulated(2_000)
    out = cm.make_calibrator("uncalibrated").fit(score, y).transform(score)
    assert np.array_equal(out, score)
    assert out is not score  # a copy: editing the result cannot change the model's scores


def test_fit_calibrators_fits_every_method_on_the_same_rows() -> None:
    score, y = simulated(10_000)
    fitted = cm.fit_calibrators(cm.METHODS, score, y)
    assert set(fitted) == set(cm.METHODS)
    assert {c.fit_rows for c in fitted.values()} == {len(y)}
    assert {c.fit_events for c in fitted.values()} == {int(y.sum())}


def test_calibrators_do_not_depend_on_what_they_are_later_applied_to() -> None:
    score, y = simulated(20_000)
    calibrator = cm.IsotonicCalibrator().fit(score, y)
    first = calibrator.transform(score[:100]).copy()
    calibrator.transform(np.linspace(0, 1, 1000))
    assert np.array_equal(first, calibrator.transform(score[:100]))

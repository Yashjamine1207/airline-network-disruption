"""Tests for the calibration metrics and the Brier bootstrap (Phase 7A, Step 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from airline_disruption.calibration import metrics as mt


def test_brier_matches_a_hand_calculation() -> None:
    y = [1, 0, 1, 0]
    p = [0.9, 0.2, 0.6, 0.0]
    assert mt.brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.0) / 4)


def test_brier_skill_is_zero_for_the_constant_and_one_for_a_perfect_forecast() -> None:
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert mt.brier_skill(y, np.full(10, 0.1), reference_rate=0.1) == pytest.approx(0.0)
    assert mt.brier_skill(y, y.astype(float), reference_rate=0.1) == pytest.approx(1.0)
    assert mt.brier_skill(y, np.full(10, 0.5), reference_rate=0.1) < 0.0  # worse than the constant
    with pytest.raises(ValueError, match="reference_rate"):
        mt.brier_skill(y, np.full(10, 0.1), reference_rate=0.0)


def test_slope_and_level_are_one_and_zero_for_a_perfectly_calibrated_model() -> None:
    rng = np.random.default_rng(0)
    p = np.clip(expit(rng.normal(-3.4, 1.0, 500_000)), 1e-4, 0.9)
    y = (rng.random(len(p)) < p).astype("int8")
    slope, level = mt.calibration_slope_and_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.03) and level == pytest.approx(0.0, abs=0.05)


def test_slope_below_one_for_an_overconfident_model_and_level_shift_has_the_right_sign() -> None:
    rng = np.random.default_rng(1)
    p = np.clip(expit(rng.normal(-3.4, 1.0, 500_000)), 1e-4, 0.9)
    over = (rng.random(len(p)) < expit(0.6 * logit(p))).astype("int8")  # true spread is smaller than the model's
    assert mt.calibration_slope_and_intercept(over, p)[0] == pytest.approx(0.6, abs=0.03)
    too_low = (rng.random(len(p)) < expit(0.4 + logit(p))).astype("int8")  # the model under-predicts
    assert mt.calibration_slope_and_intercept(too_low, p)[1] == pytest.approx(0.4, abs=0.05)


def test_the_level_shift_is_the_intercept_at_slope_one_not_the_intercept_of_the_two_number_fit() -> None:
    """For an over-confident model the two differ. The level shift is found independently by root finding."""
    from scipy.optimize import brentq

    rng = np.random.default_rng(4)
    p = np.clip(expit(rng.normal(-3.4, 1.0, 300_000)), 1e-4, 0.9)
    y = (rng.random(len(p)) < expit(-0.5 + 0.6 * logit(p))).astype("int8")
    z = logit(p)
    expected = brentq(lambda a: expit(z + a).mean() - y.mean(), -5, 5)  # the shift that matches the average rate
    slope, level = mt.calibration_slope_and_intercept(y, p)
    assert level == pytest.approx(expected, abs=1e-6)
    assert slope == pytest.approx(0.6, abs=0.04)
    two_number_intercept = mt.fit_logistic(z, y, slope=True)[0] if hasattr(mt, "fit_logistic") else None
    assert abs(level - two_number_intercept) > 0.2  # not the same quantity


def test_equal_count_bins_have_equal_sizes_and_follow_the_score_order() -> None:
    p = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
    bins = mt.equal_count_bins(p, 5)
    assert np.bincount(bins).tolist() == [2, 2, 2, 2, 2]
    assert bins[np.argmin(p)] == 0 and bins[np.argmax(p)] == 4
    with pytest.raises(ValueError):
        mt.equal_count_bins(p, 11)


def test_equal_count_bins_are_deterministic_with_ties() -> None:
    p = np.array([0.1] * 6 + [0.2] * 6)
    assert np.array_equal(mt.equal_count_bins(p, 3), mt.equal_count_bins(p, 3))
    assert np.bincount(mt.equal_count_bins(p, 3)).tolist() == [4, 4, 4]


def test_reliability_table_and_ece_match_a_hand_calculation() -> None:
    # Two bins of four rows. Bin 1: mean p = 0.1, rate = 0.25. Bin 2: mean p = 0.5, rate = 0.5.
    p = np.array([0.05, 0.05, 0.15, 0.15, 0.4, 0.4, 0.6, 0.6])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    table = mt.reliability_table(y, p, 2)
    assert table["rows"].tolist() == [4, 4]
    assert table["mean_predicted"].tolist() == pytest.approx([0.1, 0.5])
    assert table["observed_rate"].tolist() == pytest.approx([0.25, 0.5])
    assert mt.expected_calibration_error(y, p, 2) == pytest.approx(0.5 * 0.15 + 0.5 * 0.0)


def test_evaluate_probabilities_reports_every_field_and_they_are_consistent() -> None:
    rng = np.random.default_rng(3)
    p = np.clip(expit(rng.normal(-3.4, 1.0, 50_000)), 1e-4, 0.9)
    y = (rng.random(len(p)) < p).astype("int8")
    r = mt.evaluate_probabilities(y, p, reference_rate=float(y.mean()), n_bins=10)
    assert r["rows"] == 50_000 and r["events"] == int(y.sum())
    assert r["predicted_to_observed"] == pytest.approx(r["mean_predicted"] / r["prevalence"])
    assert r["brier"] == pytest.approx(mt.brier(y, p))
    assert 0.5 < r["roc_auc"] < 1.0 and r["distinct_scores"] == len(np.unique(p))
    assert set(r) >= {"brier_skill", "calibration_slope", "calibration_in_the_large", "ece", "pr_auc"}


# ---------------------------------------------------------------------------
# Paired day-level bootstrap
# ---------------------------------------------------------------------------
def make_days(n_days: int = 120, per_day: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    p = np.clip(expit(rng.normal(-3.4, 0.8, len(day))), 1e-3, 0.9)
    y = (rng.random(len(day)) < p).astype("int8")
    return y, p, day


def test_bootstrap_of_identical_probabilities_is_exactly_zero() -> None:
    y, p, day = make_days()
    r = mt.paired_day_bootstrap_brier(y, p, p, day, n_boot=200)
    assert r["difference"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["n_days"] == 120


def test_bootstrap_says_the_better_forecast_is_better_with_a_negative_interval() -> None:
    y, p, day = make_days()
    worse = np.clip(p * 2.0, 1e-3, 0.99)  # doubled probabilities are clearly mis-calibrated
    r = mt.paired_day_bootstrap_brier(y, worse, p, day, n_boot=300)  # challenger = the calibrated p
    assert r["difference"] < 0 and r["ci_high"] < 0 and r["share_challenger_better"] > 0.99
    flipped = mt.paired_day_bootstrap_brier(y, p, worse, day, n_boot=300)
    assert flipped["ci_low"] > 0 and flipped["difference"] == pytest.approx(-r["difference"])


def test_bootstrap_point_estimate_equals_the_plain_brier_difference() -> None:
    y, p, day = make_days()
    q = np.clip(p * 1.3, 1e-3, 0.99)
    r = mt.paired_day_bootstrap_brier(y, p, q, day, n_boot=50)
    assert r["brier_reference"] == pytest.approx(mt.brier(y, p))
    assert r["brier_challenger"] == pytest.approx(mt.brier(y, q))
    assert r["difference"] == pytest.approx(mt.brier(y, q) - mt.brier(y, p))


def test_bootstrap_is_reproducible_and_needs_one_day_id_per_row() -> None:
    y, p, day = make_days(30, 100)
    q = np.clip(p * 1.2, 1e-3, 0.99)
    a = mt.paired_day_bootstrap_brier(y, p, q, day, n_boot=100, seed=7)
    b = mt.paired_day_bootstrap_brier(y, p, q, day, n_boot=100, seed=7)
    c = mt.paired_day_bootstrap_brier(y, p, q, day, n_boot=100, seed=8)
    assert a == b and a["ci_low"] != c["ci_low"]
    with pytest.raises(ValueError, match="one entry per row"):
        mt.paired_day_bootstrap_brier(y, p, q, day[:-1], n_boot=10)


def test_the_bootstrap_resamples_days_not_flights() -> None:
    """A day whose flights are all wrong makes the interval wide. A per-flight bootstrap would hide that."""
    rng = np.random.default_rng(9)
    n_days, per_day = 40, 500
    day = np.repeat(np.arange(n_days), per_day)
    y = (rng.random(len(day)) < 0.03).astype("int8")
    good = np.full(len(day), 0.03)
    # The challenger is right on most days and very wrong on a few whole days.
    bad_days = np.isin(day, [3, 17])
    challenger = np.where(bad_days, 0.5, 0.03)
    r = mt.paired_day_bootstrap_brier(y, good, challenger, day, n_boot=500)
    flight_level = mt.paired_day_bootstrap_brier(y, good, challenger, np.arange(len(day)), n_boot=500)
    assert (r["ci_high"] - r["ci_low"]) > 3 * (flight_level["ci_high"] - flight_level["ci_low"])


def test_bootstrap_accepts_day_labels_of_any_type() -> None:
    y, p, day = make_days(10, 200)
    labels = pd.Series(day).map(lambda d: f"2022-01-{d + 1:02d}").to_numpy()
    q = np.clip(p * 1.1, 1e-3, 0.99)
    a = mt.paired_day_bootstrap_brier(y, p, q, day, n_boot=50)
    b = mt.paired_day_bootstrap_brier(y, p, q, labels, n_boot=50)
    assert a == b

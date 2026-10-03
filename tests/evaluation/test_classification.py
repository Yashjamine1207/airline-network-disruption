"""Tests for airline_disruption.evaluation.classification (Phase 6, Step 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.evaluation.classification import (
    binary_metrics,
    calibration_table,
    paired_day_bootstrap,
    ranking_table,
)


# ---------------------------------------------------------------------------
# binary_metrics
# ---------------------------------------------------------------------------
def test_pr_auc_matches_hand_calculation() -> None:
    # Ranking: event, non-event, event, non-event. Precision at the two events: 1/1 and 2/3.
    # Average precision = (1 + 2/3) / 2.
    result = binary_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert result["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)


def test_roc_auc_brier_prevalence_and_lift() -> None:
    y = [1, 0, 1, 0]
    p = [0.9, 0.8, 0.7, 0.1]
    result = binary_metrics(y, p)
    assert result["roc_auc"] == pytest.approx(0.75)  # 3 of 4 event/non-event pairs ranked correctly
    assert result["brier_score"] == pytest.approx(np.mean((np.array(p) - np.array(y)) ** 2))
    assert result["prevalence"] == 0.5
    assert result["pr_auc_lift"] == pytest.approx(result["pr_auc"] / 0.5)
    assert (result["rows"], result["events"]) == (4, 2)
    assert result["mean_score"] == pytest.approx(np.mean(p))


def test_constant_scores_give_prevalence_as_pr_auc() -> None:
    y = np.array([1] * 3 + [0] * 97)
    assert binary_metrics(y, np.full(100, 0.03))["pr_auc"] == pytest.approx(0.03)


def test_non_probability_scores_skip_brier() -> None:
    result = binary_metrics([1, 0, 1, 0], [5.0, 2.0, 1.0, -3.0], probabilities=False)
    assert np.isnan(result["brier_score"]) and np.isnan(result["mean_score"])


@pytest.mark.parametrize(
    "y, s, message",
    [
        ([1, 0, 1], [0.1, 0.2], "same length"),
        ([1, 0, 1], [0.1, np.nan, 0.3], "NaN"),
        ([1, 0, 2], [0.1, 0.2, 0.3], "0 or 1"),
        ([0, 0, 0], [0.1, 0.2, 0.3], "one class"),
        ([], [], "No rows"),
    ],
)
def test_invalid_inputs_are_rejected(y, s, message) -> None:
    with pytest.raises(ValueError, match=message):
        binary_metrics(y, s)


# ---------------------------------------------------------------------------
# ranking_table
# ---------------------------------------------------------------------------
def _ten_rows() -> tuple[np.ndarray, np.ndarray]:
    scores = np.arange(10, 0, -1, dtype=float)  # 10, 9, ..., 1
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # events ranked first and third
    return y, scores


def test_ranking_table_hand_calculation() -> None:
    y, scores = _ten_rows()
    table = ranking_table(y, scores, fractions=(0.1, 0.2, 0.5)).set_index("fraction")
    assert table.loc[0.1, "reviewed"] == 1
    assert table.loc[0.1, "events_captured"] == 1
    assert table.loc[0.1, "precision"] == 1.0
    assert table.loc[0.1, "recall"] == 0.5
    assert table.loc[0.2, "precision"] == 0.5
    assert table.loc[0.5, "events_captured"] == 2
    assert table.loc[0.5, "precision"] == pytest.approx(0.4)
    assert table.loc[0.5, "recall"] == 1.0
    assert table.loc[0.5, "false_alerts"] == 3
    assert table.loc[0.5, "f1"] == pytest.approx(2 * 0.4 * 1.0 / 1.4)
    assert table.loc[0.5, "lift"] == pytest.approx(0.4 / 0.2)


def test_ranking_table_does_not_depend_on_row_order() -> None:
    y, scores = _ten_rows()
    permutation = np.random.default_rng(0).permutation(10)
    pd.testing.assert_frame_equal(ranking_table(y, scores), ranking_table(y[permutation], scores[permutation]))


def test_ties_at_the_cutoff_are_split_by_expectation() -> None:
    # Three rows share the top score and one of them is an event. Flagging the top 2 of 10
    # takes two of the three tied rows, so the expected number of events captured is 1 * 2/3.
    y = np.array([1, 0, 0] + [0] * 7)
    scores = np.array([5.0, 5.0, 5.0] + [1.0] * 7)
    table = ranking_table(y, scores, fractions=(0.2,))
    assert table.loc[0, "events_captured"] == pytest.approx(2 / 3)


def test_ties_give_the_same_answer_whatever_the_row_order() -> None:
    scores = np.full(10, 0.5)
    first = np.array([1, 1] + [0] * 8)
    second = np.array([0] * 8 + [1, 1])
    a = ranking_table(first, scores, fractions=(0.1,)).loc[0, "precision"]
    b = ranking_table(second, scores, fractions=(0.1,)).loc[0, "precision"]
    assert a == pytest.approx(b) == pytest.approx(0.2)  # no better than prevalence


def test_tiny_fraction_reviews_at_least_one_row() -> None:
    y, scores = _ten_rows()
    assert ranking_table(y, scores, fractions=(0.001,)).loc[0, "reviewed"] == 1


def test_recall_at_full_review_is_one() -> None:
    y, scores = _ten_rows()
    assert ranking_table(y, scores, fractions=(1.0,)).loc[0, "recall"] == 1.0


# ---------------------------------------------------------------------------
# calibration_table
# ---------------------------------------------------------------------------
def test_calibration_table_bins_have_equal_counts_and_correct_rates() -> None:
    p = np.linspace(0.05, 0.95, 10)
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    table = calibration_table(y, p, n_bins=2)
    assert table["rows"].tolist() == [5, 5]
    assert table["events"].tolist() == [1, 4]
    assert table["observed_rate"].tolist() == pytest.approx([0.2, 0.8])
    assert table["mean_predicted"].tolist() == pytest.approx([p[:5].mean(), p[5:].mean()])
    assert table["bin"].tolist() == [1, 2]


def test_calibration_table_handles_heavy_ties() -> None:
    y = np.array([1, 0, 0, 0] * 25)
    table = calibration_table(y, np.full(100, 0.25), n_bins=10)
    assert table["rows"].sum() == 100
    assert len(table) == 10


# ---------------------------------------------------------------------------
# paired_day_bootstrap
# ---------------------------------------------------------------------------
def _scored_days(seed: int = 0, n_days: int = 60, per_day: int = 400):
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    days = np.repeat(np.arange(n_days), per_day)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=1.5, size=n) > 2.2).astype(int)
    good = signal + rng.normal(scale=0.5, size=n)
    poor = rng.normal(size=n)
    return y, poor, good, days


def test_identical_models_have_zero_difference() -> None:
    y, poor, _, days = _scored_days()
    result = paired_day_bootstrap(y, poor, poor, days, n_boot=30)
    assert result["difference"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_clearly_better_model_has_positive_interval() -> None:
    y, poor, good, days = _scored_days()
    result = paired_day_bootstrap(y, poor, good, days, n_boot=60)
    assert result["difference"] > 0
    assert result["ci_low"] > 0
    assert result["share_challenger_better"] == 1.0
    assert result["n_days"] == 60


def test_swapping_reference_and_challenger_flips_the_sign() -> None:
    y, poor, good, days = _scored_days()
    forward = paired_day_bootstrap(y, poor, good, days, n_boot=40, seed=1)
    backward = paired_day_bootstrap(y, good, poor, days, n_boot=40, seed=1)
    assert backward["difference"] == pytest.approx(-forward["difference"])
    assert backward["ci_high"] == pytest.approx(-forward["ci_low"])


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    y, poor, good, days = _scored_days()
    a = paired_day_bootstrap(y, poor, good, days, n_boot=30, seed=5)
    b = paired_day_bootstrap(y, poor, good, days, n_boot=30, seed=5)
    c = paired_day_bootstrap(y, poor, good, days, n_boot=30, seed=6)
    assert a == b
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_day_resampling_gives_wider_intervals_than_flight_resampling() -> None:
    # Each model is informative only on some days (which days differs between the models).
    # The models' difference therefore varies from day to day, and resampling whole days
    # must show more uncertainty than resampling flights independently. This is the reason
    # the function resamples days.
    rng = np.random.default_rng(3)
    n_days, per_day = 40, 300
    days = np.repeat(np.arange(n_days), per_day)
    n = len(days)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=1.0, size=n) > 1.6).astype(int)
    a_is_good_today = (rng.random(n_days) < 0.5)[days]
    noise_a, noise_b = rng.normal(size=n), rng.normal(size=n)
    model_a = np.where(a_is_good_today, signal, noise_a)
    model_b = np.where(a_is_good_today, noise_b, signal)
    by_day = paired_day_bootstrap(y, model_a, model_b, days, n_boot=80, seed=2)
    by_flight = paired_day_bootstrap(y, model_a, model_b, np.arange(n), n_boot=80, seed=2)
    assert (by_day["ci_high"] - by_day["ci_low"]) > 1.5 * (by_flight["ci_high"] - by_flight["ci_low"])


def test_bootstrap_rejects_wrong_length_days() -> None:
    y, poor, good, days = _scored_days(n_days=5)
    with pytest.raises(ValueError, match="one entry per row"):
        paired_day_bootstrap(y, poor, good, days[:-1], n_boot=5)

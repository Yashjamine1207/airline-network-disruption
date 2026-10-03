"""Operating points learned on one period and applied to another (Phase 7B, Step 12)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from airline_disruption.final.operating_points import confusion_at, f1_optimal_cutoff, learn_cutoffs, operating_point_table


def test_confusion_counts_on_a_small_example():
    y = np.array([1, 1, 0, 0, 0, 1])
    score = np.array([0.9, 0.4, 0.8, 0.2, 0.1, 0.6])
    row = confusion_at(y, score, 0.5)  # flags 0.9 (event), 0.8 (not), 0.6 (event)
    assert (row["true_positives"], row["false_positives"], row["false_negatives"], row["true_negatives"]) == (2, 1, 1, 2)
    assert row["flagged"] == 3 and row["alert_rate"] == pytest.approx(0.5)
    assert row["precision"] == pytest.approx(2 / 3) and row["recall"] == pytest.approx(2 / 3)
    assert row["specificity"] == pytest.approx(2 / 3)
    assert row["f1"] == pytest.approx(f1_score(y, score >= 0.5))


def test_a_cutoff_above_every_score_flags_nothing_and_has_no_precision():
    row = confusion_at(np.array([1, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.9)
    assert row["flagged"] == 0 and np.isnan(row["precision"]) and row["recall"] == 0 and row["f1"] == 0.0


def test_f1_optimal_cutoff_matches_brute_force_over_every_distinct_score():
    rng = np.random.default_rng(4)
    y = (rng.random(800) < 0.1).astype(int)
    score = np.round(rng.random(800) + 0.5 * y, 2)
    cutoff, best = f1_optimal_cutoff(y, score)
    brute = {c: f1_score(y, score >= c) for c in np.unique(score)}
    assert best == pytest.approx(max(brute.values()), abs=1e-12)
    assert brute[cutoff] == pytest.approx(best, abs=1e-12)


def test_learned_capacity_cutoffs_flag_at_least_the_requested_share_and_keep_their_names():
    rng = np.random.default_rng(5)
    y = (rng.random(5000) < 0.05).astype(int)
    score = rng.random(5000) + 0.3 * y
    points = learn_cutoffs(y, score, (0.005, 0.01, 0.05, 0.10))
    assert list(points) == ["capacity_0.5pct", "capacity_1pct", "capacity_5pct", "capacity_10pct", "f1_optimal"]
    for fraction, name in zip((0.005, 0.01, 0.05, 0.10), list(points)[:4]):
        share = (score >= points[name]["cutoff_score"]).mean()
        assert fraction <= share < fraction + 0.01
    assert points["f1_optimal"]["kind"] == "f1_optimal"


def test_cutoffs_learned_on_one_period_are_applied_unchanged_to_another():
    """The cutoff comes from the validation scores only. A shifted second period changes the alert rate, not the cutoff."""
    rng = np.random.default_rng(6)
    y_val = (rng.random(4000) < 0.05).astype(int)
    s_val = rng.random(4000) + 0.3 * y_val
    points = learn_cutoffs(y_val, s_val, (0.01, 0.10))
    y_test = (rng.random(4000) < 0.05).astype(int)
    s_test = rng.random(4000) + 0.3 * y_test + 0.2  # every score higher
    table = operating_point_table(y_test, s_test, points, "test")
    assert (table["cutoff_score"].to_numpy() == [p["cutoff_score"] for p in points.values()]).all()
    assert table.loc[table["operating_point"] == "capacity_10pct", "alert_rate"].iloc[0] > 0.10 + 0.05


def test_probability_map_only_adds_a_display_column():
    points = {"p": {"kind": "capacity", "fraction": 0.1, "cutoff_score": 0.5}}
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.6, 0.4])
    plain = operating_point_table(y, s, points, "x")
    shown = operating_point_table(y, s, points, "x", probability_map=lambda a: a * 0.5)
    assert shown["cutoff_probability"].iloc[0] == pytest.approx(0.25)
    pd.testing.assert_frame_equal(plain, shown.drop(columns="cutoff_probability"))


def test_a_flight_scoring_exactly_the_cutoff_is_flagged():
    row = confusion_at(np.array([1, 0, 1]), np.array([0.5, 0.4, 0.3]), 0.5)
    assert row["flagged"] == 1 and row["true_positives"] == 1


def test_specificity_is_true_negatives_over_all_non_events():
    row = confusion_at(np.array([1, 0, 0, 0]), np.array([0.9, 0.8, 0.1, 0.1]), 0.5)
    assert row["recall"] == 1.0 and row["specificity"] == pytest.approx(2 / 3)

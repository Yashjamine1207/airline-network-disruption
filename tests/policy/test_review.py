"""Tests for the ranked-review scenarios and the illustrative penalty table (Phase 7B)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from airline_disruption.evaluation.classification import ranking_table
from airline_disruption.policy import review


def brute_top_k(score, k):
    order = sorted(range(len(score)), key=lambda i: (-score[i], i))
    flags = np.zeros(len(score), dtype=bool)
    flags[order[:k]] = True
    return flags


def test_review_count_rounds_up_tolerates_float_error_and_stays_in_range() -> None:
    assert review_count(100, 0.01) == 1
    assert review_count(100, 0.07) == 7  # 0.07 * 100 is 7.000000000000001 in floats; must not become 8
    assert review_count(100, 0.55) == 55
    assert review_count(101, 0.01) == 2
    assert review_count(1, 0.005) == 1 and review_count(3, 1.0) == 3
    assert review_count(200, 0.005) == 1 and review_count(201, 0.005) == 2
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            review.review_count(10, bad)
    with pytest.raises(ValueError):
        review.review_count(0, 0.1)


review_count = review.review_count


def test_whole_year_flags_match_a_brute_force_top_k_and_break_ties_by_row_order() -> None:
    rng = np.random.default_rng(0)
    score = rng.random(500)
    for fraction in review.DEFAULT_CAPACITIES:
        assert np.array_equal(review.review_flags(score, fraction), brute_top_k(score, review_count(500, fraction)))
    tied = np.array([0.5, 0.5, 0.5, 0.9, 0.5, 0.1])
    flags = review.review_flags(tied, 0.5)  # 3 of 6: the 0.9 and the first two tied rows
    assert flags.tolist() == [True, True, False, True, False, False]


def test_grouped_flags_apply_the_capacity_inside_every_group() -> None:
    rng = np.random.default_rng(1)
    score = rng.random(1000)
    groups = rng.integers(0, 7, size=1000)
    flags = review.review_flags(score, 0.05, groups)
    for g in np.unique(groups):
        inside = np.flatnonzero(groups == g)
        expected = brute_top_k(score[inside], review_count(len(inside), 0.05))
        assert np.array_equal(flags[inside], expected)
    assert flags.sum() >= math.ceil(0.05 * 1000 - 1e-9)  # never fewer than K% in total


def test_flags_do_not_depend_on_row_order_when_scores_are_distinct() -> None:
    rng = np.random.default_rng(2)
    score = rng.random(300)
    groups = rng.integers(0, 5, size=300)
    perm = rng.permutation(300)
    a = review.review_flags(score, 0.1, groups)
    b = review.review_flags(score[perm], 0.1, groups[perm])
    assert np.array_equal(a[perm], b)


def test_flags_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        review.review_flags([0.1, np.nan], 0.5)
    with pytest.raises(ValueError):
        review.review_flags([0.1, 0.2], 0.5, groups=[1])
    with pytest.raises(ValueError):
        review.review_flags([], 0.5)


def test_review_row_counts_and_rates() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    flags = np.array([True, True, True, False, False, False, False, False, False, False])
    row = review.review_row(y, flags)
    assert (row["reviewed"], row["events_captured"], row["false_alerts"], row["events_missed"]) == (3, 2, 1, 1)
    assert row["precision"] == pytest.approx(2 / 3) and row["recall"] == pytest.approx(2 / 3)
    assert row["prevalence"] == pytest.approx(0.3) and row["lift"] == pytest.approx((2 / 3) / 0.3)
    assert row["break_even_ratio"] == pytest.approx(0.5)  # 1 false alert per 2 captured events
    empty = review.review_row(y, np.zeros(10, dtype=bool))
    assert empty["events_captured"] == 0 and np.isnan(empty["precision"]) and empty["break_even_ratio"] == math.inf
    with pytest.raises(ValueError):
        review.review_row(y, flags[:5])


def test_perfect_ranking_has_precision_one_and_random_ranking_has_lift_near_one() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(20000) < 0.05).astype(int)
    perfect = review.top_k_review(y, y + rng.random(20000) * 0.1, (0.01, 0.05))
    assert (perfect["precision"] == 1.0).all()
    random_scores = review.top_k_review(y, rng.random(20000), (0.05, 0.10))
    assert random_scores["lift"].between(0.7, 1.3).all()


def test_whole_year_matches_the_existing_ranking_table_when_there_are_no_ties() -> None:
    rng = np.random.default_rng(4)
    n = 2000  # 0.005 * 2000 = 10, 0.01 * 2000 = 20: whole numbers, so the two rounding rules agree
    y = (rng.random(n) < 0.06).astype(int)
    s = rng.random(n) + 0.3 * y
    mine = review.top_k_review(y, s).set_index("fraction")
    theirs = ranking_table(y, s, review.DEFAULT_CAPACITIES).set_index("fraction")
    assert np.allclose(mine["events_captured"], theirs["events_captured"])
    assert np.allclose(mine["reviewed"], theirs["reviewed"])


def test_schemes_appear_only_when_their_groups_are_given_and_by_day_reviews_at_least_k_percent() -> None:
    rng = np.random.default_rng(5)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    s = rng.random(n)
    days = rng.integers(0, 60, size=n)
    only = review.top_k_review(y, s)
    assert set(only["scheme"]) == {"whole_year"}
    both = review.top_k_review(y, s, month_codes=days // 30, day_codes=days)
    assert set(both["scheme"]) == set(review.SCHEMES) and len(both) == 12
    assert (both["reviewed"] >= np.ceil(both["fraction"] * n - 1e-9)).all()
    assert (both["flights"] == n).all() and (both["events"] == y.sum()).all()
    for scheme, codes in (("whole_year", np.zeros(n, dtype=int)), ("by_month", days // 30), ("by_day", days)):
        sizes = np.bincount(codes)
        for fraction in review.DEFAULT_CAPACITIES:
            expected = sum(review.review_count(int(size), fraction) for size in sizes if size)
            got = both[(both["scheme"] == scheme) & (both["fraction"] == fraction)]["reviewed"].iloc[0]
            assert got == expected, (scheme, fraction)


def test_seasonal_level_helps_whole_year_but_not_by_day() -> None:
    """A score that only knows which month is busy ranks well over the year and no better than chance within a day."""
    rng = np.random.default_rng(6)
    n = 60000
    day = rng.integers(0, 120, size=n)
    month = day // 30
    rate = np.array([0.01, 0.02, 0.04, 0.08])[month]
    y = (rng.random(n) < rate).astype(int)
    score = month + rng.random(n) * 0.01  # knows the month, nothing inside it
    table = review.top_k_review(y, score, (0.05,), month_codes=month, day_codes=day).set_index("scheme")
    assert table.loc["whole_year", "lift"] > 1.6
    assert table.loc["by_day", "lift"] < 1.25 and table.loc["by_month", "lift"] < 1.25


def test_bootstrap_intervals_bracket_the_estimate_and_are_reproducible() -> None:
    rng = np.random.default_rng(7)
    n = 40000
    days = rng.integers(0, 200, size=n)
    y = (rng.random(n) < 0.05).astype(int)
    flags = review.review_flags(rng.random(n) + 0.3 * y, 0.02)  # reviewed (2%) differs from the events (5%), so precision and recall differ
    point = review.review_row(y, flags)
    assert abs(point["precision"] - point["recall"]) > 0.1
    one = review.bootstrap_day_intervals(y, flags, days, n_boot=400, seed=1)
    again = review.bootstrap_day_intervals(y, flags, days, n_boot=400, seed=1)
    assert one == again
    assert one["precision_low"] < point["precision"] < one["precision_high"]
    assert one["recall_low"] < point["recall"] < one["recall_high"]
    assert one["n_days"] == 200 and one["n_boot"] == 400


def test_day_bootstrap_is_wider_than_a_flight_level_one_when_alerts_and_events_cluster_on_days() -> None:
    """Flights on one day share conditions. Resampling days must show that extra uncertainty."""
    rng = np.random.default_rng(8)
    n_days, per_day = 150, 300
    days = np.repeat(np.arange(n_days), per_day)
    day_rate = rng.uniform(0.005, 0.15, size=n_days)
    y = (rng.random(len(days)) < day_rate[days]).astype(int)
    alert_days = rng.random(n_days) < 0.1  # alerts come in bursts on a few days
    flags = alert_days[days] & (rng.random(len(days)) < 0.5)
    clustered = review.bootstrap_day_intervals(y, flags, days, n_boot=500, seed=1)
    flat = review.bootstrap_day_intervals(y, flags, rng.permutation(days), n_boot=500, seed=1)  # day structure destroyed
    for name in ("precision", "recall"):
        assert (clustered[f"{name}_high"] - clustered[f"{name}_low"]) > 1.5 * (flat[f"{name}_high"] - flat[f"{name}_low"])


def test_utility_formulas_and_the_break_even_boundary() -> None:
    rows = pd.DataFrame([{"scheme": "whole_year", "capacity": "1%", "fraction": 0.01, "flights": 1000, "events": 50, "prevalence": 0.05,
                          "reviewed": 10, "events_captured": 2, "false_alerts": 8, "events_missed": 48, "precision": 0.2, "recall": 0.04,
                          "lift": 4.0, "break_even_ratio": 4.0}])
    table = review.utility_table(rows, ratios=(1, 4, 10)).set_index("miss_to_false_alert_ratio")
    assert table.loc[10, "net_reduction_units"] == pytest.approx(10 * 2 - 8)
    assert table.loc[1, "net_reduction_units"] == pytest.approx(2 - 8)
    assert table.loc[4, "net_reduction_units"] == pytest.approx(0.0) and not table.loc[4, "pays_off_at_this_ratio"]  # break even is not a profit
    assert bool(table.loc[10, "pays_off_at_this_ratio"]) and not bool(table.loc[1, "pays_off_at_this_ratio"])
    random_captured = 10 * 0.05
    assert table.loc[10, "random_review_reduction_units"] == pytest.approx(10 * random_captured - (10 - random_captured))
    assert table.loc[10, "advantage_over_random_units"] == pytest.approx(table.loc[10, "net_reduction_units"] - table.loc[10, "random_review_reduction_units"])
    scaled = review.utility_table(rows, ratios=(10,), false_alert_weight=3.0).iloc[0]
    assert scaled["net_reduction_units"] == pytest.approx(3.0 * (10 * 2 - 8))  # weights scale everything together
    for bad in ({"ratios": (0,)}, {"false_alert_weight": 0.0}):
        with pytest.raises(ValueError):
            review.utility_table(rows, **bad)


def test_capacity_name() -> None:
    assert [review.capacity_name(f) for f in review.DEFAULT_CAPACITIES] == ["0.5%", "1%", "5%", "10%"]

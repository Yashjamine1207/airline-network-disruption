"""Tests for rule-based case selection (Phase 7A, Step 10)."""

from __future__ import annotations

import numpy as np
import pytest

from airline_disruption.explain import cases


def simulated(n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    score = rng.beta(1.2, 40, n)
    y = (rng.random(n) < np.clip(score * 1.5, 0, 1)).astype(int)
    return y, score


def test_ranks_start_at_one_for_the_highest_score_and_break_ties_by_position() -> None:
    rank = cases.score_ranks([0.2, 0.9, 0.2, 0.5])
    assert rank.tolist() == [3, 1, 4, 2]


def test_every_category_obeys_its_definition() -> None:
    y, s = simulated()
    table = cases.select_cases(y, s)
    n = len(y)
    alert_k, review_k = round(0.01 * n), round(0.10 * n)
    for row in table.itertuples():
        assert row.label == y[row.position] and row.score == pytest.approx(s[row.position])
        if row.category == "true_positive":
            assert row.rank <= alert_k and row.label == 1
        elif row.category == "false_positive":
            assert row.rank <= alert_k and row.label == 0
        elif row.category == "false_negative":
            assert row.rank > review_k and row.label == 1
        elif row.category == "low_risk_correct":
            assert row.rank > n / 2 and row.label == 0
        elif row.category == "high_confidence_false_positive":
            assert row.label == 0
        else:
            assert row.category == "high_confidence_false_negative" and row.label == 1


def test_a_row_is_chosen_at_most_once_and_every_category_is_present_with_enough_rows() -> None:
    y, s = simulated()
    table = cases.select_cases(y, s)
    assert table["position"].is_unique
    assert set(table["category"]) == set(cases.CATEGORIES)
    assert (table.groupby("category").size() == 3).all()


def test_the_extremes_are_the_most_confident_mistakes() -> None:
    y, s = simulated()
    table = cases.select_cases(y, s, extremes=3)
    fp = table[table["category"] == "high_confidence_false_positive"]
    fn = table[table["category"] == "high_confidence_false_negative"]
    assert fp["score"].min() >= np.sort(s[y == 0])[-3] - 1e-15  # the three highest-scoring non-events
    assert fn["score"].max() <= np.sort(s[y == 1])[2] + 1e-15  # the three lowest-scoring events


def test_typical_cases_sit_at_the_requested_quantiles_of_their_group() -> None:
    y, s = simulated(50_000)
    table = cases.select_cases(y, s, extremes=0, quantiles=(0.0, 0.5, 1.0))
    tp = table[table["category"] == "true_positive"].sort_values("rank")
    rank = cases.score_ranks(s)
    pool = np.sort(rank[(rank <= round(0.01 * len(y))) & (y == 1)])
    assert tp["rank"].tolist() == [pool[0], pool[int(round(0.5 * (len(pool) - 1)))], pool[-1]]


def test_selection_is_deterministic_and_uses_only_score_and_label() -> None:
    y, s = simulated()
    a, b = cases.select_cases(y, s), cases.select_cases(y.copy(), s.copy())
    assert a.equals(b)


def test_small_groups_return_fewer_rows_and_never_fail() -> None:
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.06, 0.07])
    table = cases.select_cases(y, s, alert_fraction=0.1, missed_fraction=0.2)
    assert table["position"].is_unique and len(table) >= 2
    assert set(table["category"]) <= set(cases.CATEGORIES)


def test_bad_arguments_are_rejected() -> None:
    y, s = simulated(1000)
    with pytest.raises(ValueError, match="alert_fraction"):
        cases.select_cases(y, s, alert_fraction=0.2, missed_fraction=0.1)
    with pytest.raises(ValueError, match="same length"):
        cases.select_cases(y[:-1], s)
    with pytest.raises(ValueError, match="NaN"):
        cases.select_cases(y, np.where(np.arange(len(s)) == 3, np.nan, s))
    with pytest.raises(ValueError, match="0 or 1"):
        cases.select_cases(np.where(np.arange(len(y)) == 3, 2, y), s)
    with pytest.raises(ValueError, match="quantiles"):
        cases.select_cases(y, s, quantiles=(1.5,))

"""Tests for airline_disruption.models.rate_baselines (Phase 6, Step 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.models.rate_baselines import (
    apply_rate_table,
    build_rate_baselines,
    fit_rate_table,
)


def test_smoothed_rate_hand_calculation() -> None:
    table = fit_rate_table(pd.Series(["A", "A", "B"]), np.array([1, 0, 0]), strength=2.0)
    overall = 1 / 3
    assert table.overall_rate == pytest.approx(overall)
    assert table.rates["A"] == pytest.approx((1 + 2 * overall) / (2 + 2))
    assert table.rates["B"] == pytest.approx((0 + 2 * overall) / (1 + 2))


def test_large_groups_move_away_from_the_overall_rate() -> None:
    keys = pd.Series(["A"] * 1000 + ["B"] * 1000)
    y = np.array([1] * 100 + [0] * 900 + [0] * 1000)
    table = fit_rate_table(keys, y, strength=50.0)
    assert table.rates["A"] > 0.09 and table.rates["B"] < 0.01


def test_unseen_key_gets_overall_rate_or_the_fallback() -> None:
    table = fit_rate_table(pd.Series(["A", "B"]), np.array([1, 0]), strength=1.0)
    keys = pd.Series(["A", "C"])
    assert apply_rate_table(table, keys)[1] == pytest.approx(0.5)
    assert apply_rate_table(table, keys, fallback=np.array([0.9, 0.123]))[1] == pytest.approx(0.123)
    # A seen key ignores the fallback.
    assert apply_rate_table(table, keys, fallback=np.array([0.9, 0.123]))[0] == pytest.approx(table.rates["A"])


def test_categorical_keys_work() -> None:
    keys = pd.Series(pd.Categorical(["A", "A", "B"], categories=["A", "B", "Z"]))
    table = fit_rate_table(keys, np.array([1, 0, 0]))
    assert set(table.rates.index) == {"A", "B"}


def _frames():
    # ATL has two routes in the fit rows, so the route rate and the origin rate differ.
    fit = pd.DataFrame(
        {
            "carrier_identifier": ["C1", "C1", "C1", "C2"],
            "origin_airport": ["ATL", "ATL", "ATL", "JFK"],
            "destination_airport": ["JFK", "JFK", "BOS", "ATL"],
            "route": ["ATL-JFK", "ATL-JFK", "ATL-BOS", "JFK-ATL"],
        }
    )
    y = np.array([1, 1, 0, 0])
    apply = pd.DataFrame(
        {
            "carrier_identifier": ["C1", "C2"],
            "origin_airport": ["ATL", "ATL"],
            "destination_airport": ["JFK", "LAX"],
            "route": ["ATL-JFK", "ATL-LAX"],  # second route never seen in the fit rows
        }
    )
    return fit, y, apply


def test_build_baselines_returns_all_five() -> None:
    fit, y, apply = _frames()
    scores = build_rate_baselines(fit, y, apply)
    assert list(scores) == [
        "baseline_prevalence",
        "baseline_carrier_rate",
        "baseline_origin_rate",
        "baseline_destination_rate",
        "baseline_route_rate",
    ]
    assert all(len(v) == 2 for v in scores.values())
    assert scores["baseline_prevalence"].tolist() == [0.5, 0.5]


def test_unseen_route_falls_back_to_the_origin_rate() -> None:
    fit, y, apply = _frames()
    scores = build_rate_baselines(fit, y, apply)
    assert scores["baseline_route_rate"][1] == pytest.approx(scores["baseline_origin_rate"][1])
    assert scores["baseline_route_rate"][0] != pytest.approx(scores["baseline_origin_rate"][0])  # seen route uses its own rate


def test_baselines_are_learned_from_fit_rows_only() -> None:
    fit, y, apply = _frames()
    first = build_rate_baselines(fit, y, apply)
    # Adding columns or changing row order in the apply frame cannot change a row's own score.
    extra = apply.assign(severe_delay_120=[1, 1])
    again = build_rate_baselines(fit, y, extra.iloc[::-1].reset_index(drop=True))
    for name in first:
        assert again[name][::-1].tolist() == pytest.approx(first[name].tolist())

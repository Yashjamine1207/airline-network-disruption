"""Tests for the expanding-window roles (Phase 7B): chronology, boundaries and the locked final test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.models import lightgbm_benchmark as lb
from airline_disruption.policy.windows import expanding_window_roles


def stamps(start="2019-01-01", end="2022-12-31 23:00", freq="6h"):
    return pd.Series(pd.date_range(start, end, freq=freq, tz="UTC"))


def test_boundaries_are_exact_for_2022() -> None:
    s = pd.Series(pd.to_datetime(["2019-01-01 00:00", "2021-06-30 23:59", "2021-07-01 00:00", "2021-12-31 23:59",
                                  "2022-01-01 00:00", "2022-12-31 23:59"], utc=True))
    roles = expanding_window_roles(s, 2022)
    assert roles.tolist() == ["fit", "fit", "early_stop", "early_stop", "eval", "eval"]


def test_roles_are_chronological_and_disjoint_for_every_allowed_year() -> None:
    s = stamps()
    for year in (2020, 2021, 2022):
        roles = expanding_window_roles(s, year)
        fit, stop, ev = (s[roles == r] for r in ("fit", "early_stop", "eval"))
        assert fit.max() < stop.min() and stop.max() < ev.min()
        assert ev.min() >= pd.Timestamp(year=year, month=1, day=1, tz="UTC") and ev.max() < pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        assert len(fit) + len(stop) + len(ev) + (roles == "unused").sum() == len(s)
        assert (s[roles == "unused"] >= pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")).all()


def test_later_years_are_unused_not_leaked_into_fit() -> None:
    s = stamps()
    roles = expanding_window_roles(s, 2020)
    assert (s[roles == "unused"] >= pd.Timestamp("2021-01-01", tz="UTC")).all()
    assert s[roles == "fit"].max() < pd.Timestamp("2019-07-01", tz="UTC")


def test_the_final_test_year_and_final_test_rows_are_refused() -> None:
    with pytest.raises(ValueError):
        expanding_window_roles(stamps(end="2023-08-31"), 2022)  # the rows themselves include 2023
    with pytest.raises(ValueError):
        expanding_window_roles(stamps(), 2023)
    with pytest.raises(ValueError):
        expanding_window_roles(stamps(), 2024)


def test_a_year_without_earlier_data_and_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        expanding_window_roles(stamps(), 2019)  # no fit or early-stop rows before 2019
    with pytest.raises(ValueError):
        expanding_window_roles(pd.Series(pd.date_range("2019-01-01", "2022-12-31", freq="6h")), 2022)  # naive timestamps
    with pytest.raises(ValueError):
        expanding_window_roles(stamps(), 2022, stop_months=0)


def test_2022_matches_the_locked_phase6_roles() -> None:
    s = stamps()
    meta = pd.DataFrame({"prediction_timestamp_utc": s})
    meta["phase6_split"] = np.where(s < pd.Timestamp("2022-01-01", tz="UTC"), "development", "validation")
    locked = lb.assign_roles(meta, lb.PROTOCOL_PHASE6).to_numpy()
    mine = expanding_window_roles(s, 2022)
    assert (np.where(mine == "eval", "validation", mine) == locked).all()

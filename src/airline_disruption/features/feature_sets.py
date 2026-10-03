"""Approved Phase 6 feature sets.

Phase 6, Step 3. Feature version: ``phase6_features_v1``.

This module is the single place that says which columns a Phase 6 model may
use. Every model (LightGBM benchmark, LSTM, GRU, Transformer) must take its
columns from ``columns_for(...)`` so that the ablations compare the same
features on the same rows.

What is allowed
---------------
* BASE: schedule, carrier, airport and route columns. These are identical to
  the Phase 4 base set (a test compares them with ``BASE_FEATURE_COLUMNS`` in
  ``airline_disruption.models.tabular_preprocessing``).
* HISTORY: scheduled-flight counts for the previous completed UTC month.
* NETWORK: schedule-only airport-route degrees for the previous completed UTC
  month.

What is NOT allowed
-------------------
Historical outcome rates (severe-delay rate, cancellation rate, mean delay).
``docs/decisions/historical_outcome_availability.md`` blocks them because the
source has no field saying when an earlier flight's outcome became known.
``validate_feature_columns`` rejects any column whose name looks like an
outcome. The check is name based, so it catches accidents, not clever
renames. It does not replace the leakage audit.

There is no aircraft or tail identifier in the source. No feature here, and no
column in any Phase 6 table, identifies an aircraft.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

FEATURE_VERSION = "phase6_features_v1"

# Row identifier shared by every Phase 3 / Phase 6 table.
ID_COLUMN = "source_row_number"

# Scheduled departure in UTC. Not a model feature; used to order and verify rows.
TIMESTAMP_COLUMN = "scheduled_departure_utc"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
BASE_NUMERIC_COLUMNS = (
    "scheduled_departure_year",
    "scheduled_departure_month",
    "scheduled_departure_day_of_week",
    "scheduled_departure_hour_local",
    "scheduled_departure_minute_local",
    "scheduled_arrival_hour_local",
    "scheduled_arrival_minute_local",
    "scheduled_elapsed_time_minutes",
    "distance_miles",
)

BASE_CATEGORICAL_COLUMNS = (
    "origin_airport",
    "destination_airport",
    "route",
    "carrier_identifier",
)

BASE_COLUMNS = BASE_NUMERIC_COLUMNS + BASE_CATEGORICAL_COLUMNS

# Previous completed UTC month, scheduled flights only.
HISTORY_COLUMNS = (
    "prior_calendar_month_carrier_scheduled_flight_count",
    "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count",
)

# Previous completed UTC month, directed airport graph, edge weight = scheduled flights.
NETWORK_COLUMNS = (
    "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree",
    "prior_month_destination_in_degree",
    "prior_month_destination_weighted_in_degree",
)

# ---------------------------------------------------------------------------
# Named feature sets (the Phase 6 ablation ladder)
# ---------------------------------------------------------------------------
BASE_NO_YEAR_COLUMNS = tuple(c for c in BASE_COLUMNS if c != "scheduled_departure_year")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "base": BASE_COLUMNS,
    # Same as base without the calendar year. The year is a number the model
    # can only learn from 2019-2021, so it lets a tree isolate 2020 and then
    # treat 2022-2023 as "like 2021". On the 2022 validation rows dropping it
    # raised PR-AUC from 0.0415 to 0.0456 (Step 4), so the ladder below is
    # built on this set.
    "base_no_year": BASE_NO_YEAR_COLUMNS,
    # Ladder on the year-free base (used for the Phase 6 ablations).
    "base_no_year_history": BASE_NO_YEAR_COLUMNS + HISTORY_COLUMNS,
    "base_no_year_network": BASE_NO_YEAR_COLUMNS + NETWORK_COLUMNS,
    "base_no_year_history_network": BASE_NO_YEAR_COLUMNS + HISTORY_COLUMNS + NETWORK_COLUMNS,
    # Same ladder on the Phase 4 base (kept for comparison with Phase 4 only).
    "base_history": BASE_COLUMNS + HISTORY_COLUMNS,
    "base_network": BASE_COLUMNS + NETWORK_COLUMNS,
    "base_history_network": BASE_COLUMNS + HISTORY_COLUMNS + NETWORK_COLUMNS,
}

# Every column any set needs, in a stable order, without repeats.
ALL_FEATURE_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys(c for cols in FEATURE_SETS.values() for c in cols))

# ---------------------------------------------------------------------------
# Outcome guard
# ---------------------------------------------------------------------------
# Raw or derived columns that hold outcomes, matched exactly (case sensitive).
FORBIDDEN_EXACT = frozenset(
    {
        "ARR_TIME",
        "DEP_TIME",
        "ELAPSED_TIME",
        "AIR_TIME",
        "ARR_DELAY",
        "DEP_DELAY",
        "arrival_delay_minutes",
    }
)

# Words that mark an outcome when they appear in a column name. Names are split
# on every non-alphanumeric character, so "scheduled_elapsed_time_minutes" is
# fine ("elapsed" is not on this list) and "prior_month_severe_delay_rate" is
# rejected on three words.
FORBIDDEN_WORDS = frozenset(
    {
        "delay",
        "delays",
        "cancel",
        "cancelled",
        "cancellation",
        "diverted",
        "diversion",
        "severe",
        "actual",
        "outcome",
        "completed",
        "rate",
        "tail",
        "wheels",
        "taxi",
    }
)


def _words(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def find_forbidden_columns(columns: Iterable[str]) -> list[str]:
    """Return the columns whose names look like outcomes or aircraft identifiers."""
    flagged = []
    for name in columns:
        if name in FORBIDDEN_EXACT or _words(name) & FORBIDDEN_WORDS:
            flagged.append(name)
    return flagged


def validate_feature_columns(columns: Iterable[str]) -> None:
    """Raise ValueError if any column is duplicated or looks like an outcome."""
    columns = list(columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"Duplicate feature columns: {duplicates}")
    flagged = find_forbidden_columns(columns)
    if flagged:
        raise ValueError(f"Outcome-like or identifier columns are not allowed as features: {flagged}")


def columns_for(feature_set: str) -> tuple[str, ...]:
    """Return the validated column tuple for a named feature set."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {feature_set!r}. Available: {sorted(FEATURE_SETS)}")
    columns = FEATURE_SETS[feature_set]
    validate_feature_columns(columns)
    return columns


def categorical_columns_in(columns: Iterable[str]) -> list[str]:
    """The categorical columns among ``columns``, in the base order."""
    wanted = set(columns)
    return [c for c in BASE_CATEGORICAL_COLUMNS if c in wanted]

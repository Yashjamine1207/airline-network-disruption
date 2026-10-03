"""Tests for airline_disruption.features.feature_sets (Phase 6, Step 3)."""

from __future__ import annotations

import pytest

from airline_disruption.features.feature_sets import (
    ALL_FEATURE_COLUMNS,
    BASE_CATEGORICAL_COLUMNS,
    BASE_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    FEATURE_SETS,
    HISTORY_COLUMNS,
    NETWORK_COLUMNS,
    categorical_columns_in,
    columns_for,
    find_forbidden_columns,
    validate_feature_columns,
)


def test_base_set_matches_the_phase4_base_set() -> None:
    # The Phase 6 benchmark must reproduce Phase 4, so the columns must not drift apart.
    phase4 = pytest.importorskip("airline_disruption.models.tabular_preprocessing")
    assert BASE_NUMERIC_COLUMNS == phase4.BASE_NUMERIC_COLUMNS
    assert BASE_CATEGORICAL_COLUMNS == phase4.BASE_CATEGORICAL_COLUMNS
    assert BASE_COLUMNS == phase4.BASE_FEATURE_COLUMNS


@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_every_registered_set_is_clean(name: str) -> None:
    columns = columns_for(name)
    assert len(columns) == len(set(columns))
    assert find_forbidden_columns(columns) == []


def test_sets_are_nested_as_the_ablation_ladder_expects() -> None:
    base = set(FEATURE_SETS["base"])
    assert base < set(FEATURE_SETS["base_history"])
    assert base < set(FEATURE_SETS["base_network"])
    assert set(FEATURE_SETS["base_history"]) < set(FEATURE_SETS["base_history_network"])
    assert set(FEATURE_SETS["base_network"]) < set(FEATURE_SETS["base_history_network"])


def test_history_and_network_groups_do_not_overlap_the_base() -> None:
    assert not set(HISTORY_COLUMNS) & set(BASE_COLUMNS)
    assert not set(NETWORK_COLUMNS) & set(BASE_COLUMNS)
    assert not set(HISTORY_COLUMNS) & set(NETWORK_COLUMNS)


def test_no_year_set_differs_from_base_by_exactly_the_year() -> None:
    assert set(FEATURE_SETS["base"]) - set(FEATURE_SETS["base_no_year"]) == {"scheduled_departure_year"}
    assert set(FEATURE_SETS["base_no_year"]) <= set(FEATURE_SETS["base"])


def test_no_year_ladder_mirrors_the_base_ladder_without_the_year() -> None:
    for suffix in ("history", "network", "history_network"):
        with_year = set(FEATURE_SETS[f"base_{suffix}"])
        without_year = set(FEATURE_SETS[f"base_no_year_{suffix}"])
        assert with_year - without_year == {"scheduled_departure_year"}
        assert set(FEATURE_SETS["base_no_year"]) < without_year
    assert set(FEATURE_SETS["base_no_year_history"]) < set(FEATURE_SETS["base_no_year_history_network"])
    assert set(FEATURE_SETS["base_no_year_network"]) < set(FEATURE_SETS["base_no_year_history_network"])


def test_all_feature_columns_is_the_union_without_repeats() -> None:
    assert len(ALL_FEATURE_COLUMNS) == len(set(ALL_FEATURE_COLUMNS))
    assert set(ALL_FEATURE_COLUMNS) == set().union(*FEATURE_SETS.values())


@pytest.mark.parametrize(
    "bad",
    [
        "arrival_delay_minutes",
        "ARR_DELAY",
        "DEP_DELAY",
        "DELAY_DUE_WEATHER",
        "DELAY_DUE_LATE_AIRCRAFT",
        "CANCELLED",
        "cancelled_target",
        "CANCELLATION_CODE",
        "DIVERTED",
        "severe_delay_120",
        "completed_flight",
        "ARR_TIME",
        "DEP_TIME",
        "ELAPSED_TIME",
        "AIR_TIME",
        "TAXI_OUT",
        "WHEELS_ON",
        "prior_month_severe_delay_rate",
        "origin_historical_cancellation_rate",
        "carrier_historical_severe_delay_rate",
        "tail_number",
        "origin_historical_rate",
        "route_disruption_rate_prior_month",
        "actual_departure_hour",
    ],
)
def test_outcome_like_names_are_rejected(bad: str) -> None:
    assert find_forbidden_columns([bad]) == [bad]
    with pytest.raises(ValueError, match="not allowed"):
        validate_feature_columns(["distance_miles", bad])


@pytest.mark.parametrize(
    "good",
    [
        "scheduled_elapsed_time_minutes",  # scheduled, not actual elapsed time
        "scheduled_arrival_hour_local",
        "prior_month_origin_weighted_out_degree",
        "prior_calendar_month_route_scheduled_flight_count",
        "carrier_identifier",
    ],
)
def test_legitimate_schedule_names_are_accepted(good: str) -> None:
    assert find_forbidden_columns([good]) == []


def test_duplicate_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_feature_columns(["distance_miles", "distance_miles"])


def test_unknown_feature_set_lists_the_options() -> None:
    with pytest.raises(KeyError, match="base_history_network"):
        columns_for("weather")


def test_categorical_helper_keeps_base_order() -> None:
    columns = ["route", "distance_miles", "origin_airport"]
    assert categorical_columns_in(columns) == ["origin_airport", "route"]

"""
Tests for vectorised scheduled-arrival reconciliation.

These tests cover:
- same-day cross-timezone schedules;
- overnight destination-local arrivals;
- 2400 arrival-clock rollover;
- missing departure UTC;
- missing scheduled elapsed time;
- invalid arrival HHMM values;
- spring DST nonexistent arrival local times;
- autumn DST ambiguous arrival local times;
- duration differences exceeding the tolerance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.timestamps.vectorised_arrivals import (
    reconcile_scheduled_arrivals_vectorised,
    validate_vectorised_arrival_columns,
)


def build_vectorised_arrival_test_chunk() -> pd.DataFrame:
    """Build deterministic input rows for vectorised arrival tests."""
    return pd.DataFrame(
        {
            "flight_date_local": [
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-03-12",
                "2023-11-05",
                "2023-06-15",
            ],
            "scheduled_arrival_time_local_hhmm": [
                1130,
                30,
                2400,
                1130,
                1130,
                2360,
                230,
                130,
                1130,
            ],
            "scheduled_elapsed_minutes": [
                210.0,
                240.0,
                120.0,
                210.0,
                pd.NA,
                210.0,
                180.0,
                150.0,
                600.0,
            ],
            "destination_iana_timezone": [
                "America/Chicago",
                "America/Chicago",
                "America/Chicago",
                "America/Chicago",
                "America/Chicago",
                "America/Chicago",
                "America/New_York",
                "America/New_York",
                "America/Chicago",
            ],
        }
    )


def build_scheduled_departure_utc() -> pd.Series:
    """Build UTC departures aligned to the vectorised arrival test rows."""
    return pd.Series(
        [
            pd.Timestamp("2023-06-15 13:00:00+00:00"),
            pd.Timestamp("2023-06-16 01:30:00+00:00"),
            pd.Timestamp("2023-06-15 22:00:00+00:00"),
            pd.NaT,
            pd.Timestamp("2023-06-15 13:00:00+00:00"),
            pd.Timestamp("2023-06-15 13:00:00+00:00"),
            pd.Timestamp("2023-03-12 05:00:00+00:00"),
            pd.Timestamp("2023-11-05 04:00:00+00:00"),
            pd.Timestamp("2023-06-15 13:00:00+00:00"),
        ],
        dtype="datetime64[ns, UTC]",
    )


def test_validate_vectorised_arrival_columns_rejects_missing_column() -> None:
    """Required vectorised-arrival fields must be present."""
    invalid_chunk = pd.DataFrame(
        {
            "flight_date_local": ["2023-06-15"],
            "scheduled_arrival_time_local_hhmm": [1130],
            "scheduled_elapsed_minutes": [210.0],
        }
    )

    with pytest.raises(ValueError, match="destination_iana_timezone"):
        validate_vectorised_arrival_columns(invalid_chunk)


def test_vectorised_arrivals_reconcile_valid_schedule_cases() -> None:
    """Valid same-day, overnight, and 2400 schedules must reconcile in UTC."""
    chunk = build_vectorised_arrival_test_chunk()
    departure_utc = build_scheduled_departure_utc()

    result = reconcile_scheduled_arrivals_vectorised(
        standardised_chunk=chunk,
        scheduled_departure_utc=departure_utc,
    )

    cross_timezone_row = result.loc[0]
    assert (
        cross_timezone_row[
            "scheduled_arrival_reconciliation_status"
        ]
        == "reconciled"
    )
    assert cross_timezone_row[
        "scheduled_arrival_date_offset_days"
    ] == 0
    assert cross_timezone_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-15 16:30:00+00:00"
    )
    assert cross_timezone_row[
        "scheduled_duration_utc_minutes"
    ] == 210
    assert cross_timezone_row[
        "scheduled_duration_difference_minutes"
    ] == 0

    overnight_row = result.loc[1]
    assert (
        overnight_row[
            "scheduled_arrival_reconciliation_status"
        ]
        == "reconciled"
    )
    assert overnight_row[
        "scheduled_arrival_date_offset_days"
    ] == 1
    assert overnight_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-16 05:30:00+00:00"
    )
    assert overnight_row[
        "scheduled_duration_utc_minutes"
    ] == 240
    assert overnight_row[
        "scheduled_duration_difference_minutes"
    ] == 0

    rollover_row = result.loc[2]
    assert (
        rollover_row[
            "scheduled_arrival_reconciliation_status"
        ]
        == "duration_difference_exceeds_tolerance"
    )
    assert rollover_row[
        "scheduled_arrival_date_offset_days"
    ] == 1
    assert rollover_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-16 05:00:00+00:00"
    )
    assert rollover_row[
        "scheduled_duration_utc_minutes"
    ] == 420
    assert rollover_row[
        "scheduled_duration_difference_minutes"
    ] == 300


def test_vectorised_arrivals_flag_missing_and_invalid_inputs() -> None:
    """Missing departures, elapsed times, and invalid clocks must stay visible."""
    chunk = build_vectorised_arrival_test_chunk()
    departure_utc = build_scheduled_departure_utc()

    result = reconcile_scheduled_arrivals_vectorised(
        standardised_chunk=chunk,
        scheduled_departure_utc=departure_utc,
    )

    assert (
        result.loc[
            3,
            "scheduled_arrival_reconciliation_status",
        ]
        == "missing_or_invalid_departure_utc"
    )

    assert (
        result.loc[
            4,
            "scheduled_arrival_reconciliation_status",
        ]
        == "missing_scheduled_elapsed_time"
    )

    assert (
        result.loc[
            5,
            "scheduled_arrival_reconciliation_status",
        ]
        == "invalid_arrival_time"
    )


def test_vectorised_arrivals_classify_dst_edge_cases() -> None:
    """Ambiguous and nonexistent arrival-local times must not be guessed."""
    chunk = build_vectorised_arrival_test_chunk()
    departure_utc = build_scheduled_departure_utc()

    result = reconcile_scheduled_arrivals_vectorised(
        standardised_chunk=chunk,
        scheduled_departure_utc=departure_utc,
    )

    assert (
        result.loc[
            6,
            "scheduled_arrival_reconciliation_status",
        ]
        == "nonexistent_arrival_local_time"
    )
    assert pd.isna(result.loc[6, "scheduled_arrival_utc"])

    assert (
        result.loc[
            7,
            "scheduled_arrival_reconciliation_status",
        ]
        == "ambiguous_arrival_local_time"
    )
    assert pd.isna(result.loc[7, "scheduled_arrival_utc"])


def test_vectorised_arrivals_flag_duration_difference_above_tolerance() -> None:
    """Implausible duration mismatches must be retained as audit flags."""
    chunk = build_vectorised_arrival_test_chunk()
    departure_utc = build_scheduled_departure_utc()

    result = reconcile_scheduled_arrivals_vectorised(
        standardised_chunk=chunk,
        scheduled_departure_utc=departure_utc,
        maximum_duration_difference_minutes=30.0,
    )

    assert (
        result.loc[
            8,
            "scheduled_arrival_reconciliation_status",
        ]
        == "duration_difference_exceeds_tolerance"
    )
    assert result.loc[
        8,
        "scheduled_duration_difference_minutes",
    ] == 390
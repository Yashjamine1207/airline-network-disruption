"""
Tests for chunk-level scheduled timestamp normalisation.

These tests verify that the normalisation function:
- requires the necessary standardised schedule columns;
- creates scheduled departure, arrival, and prediction timestamps;
- uses UTC duration reconciliation for cross-timezone schedules;
- handles missing schedule departure times without inventing predictions;
- preserves auditable timestamp status fields.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.timestamps.normalise_schedule import (
    normalise_scheduled_timestamps,
    validate_standardised_schedule_columns,
)


def build_standardised_schedule_chunk() -> pd.DataFrame:
    """Build a small deterministic standardised-flight table for testing."""
    return pd.DataFrame(
        {
            "source_row_number": [0, 1],
            "flight_date_local": [
                "2023-06-15",
                "2023-06-15",
            ],
            "origin_airport": [
                "JFK",
                "LAX",
            ],
            "destination_airport": [
                "ORD",
                "JFK",
            ],
            "origin_iana_timezone": [
                "America/New_York",
                "America/Los_Angeles",
            ],
            "destination_iana_timezone": [
                "America/Chicago",
                "America/New_York",
            ],
            "scheduled_departure_time_local_hhmm": [
                900,
                pd.NA,
            ],
            "scheduled_arrival_time_local_hhmm": [
                1130,
                700,
            ],
            "scheduled_elapsed_minutes": [
                210.0,
                300.0,
            ],
        }
    )


def test_validate_standardised_schedule_columns_accepts_complete_chunk() -> None:
    """A chunk containing all required schedule fields should pass."""
    chunk = build_standardised_schedule_chunk()

    validate_standardised_schedule_columns(chunk)


def test_validate_standardised_schedule_columns_rejects_missing_column() -> None:
    """Missing required schedule data must stop timestamp reconstruction."""
    chunk = build_standardised_schedule_chunk().drop(
        columns=["origin_iana_timezone"]
    )

    with pytest.raises(ValueError, match="origin_iana_timezone"):
        validate_standardised_schedule_columns(chunk)


def test_normalise_scheduled_timestamps_creates_utc_timestamps() -> None:
    """A valid cross-timezone schedule should produce audited UTC timestamps."""
    chunk = build_standardised_schedule_chunk()

    result = normalise_scheduled_timestamps(chunk)

    valid_row = result.loc[0]

    assert valid_row["scheduled_departure_reconciliation_status"] == "reconciled"
    assert valid_row["scheduled_arrival_reconciliation_status"] == "reconciled"

    assert valid_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 13:00:00+00:00"
    )
    assert valid_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-15 16:30:00+00:00"
    )
    assert valid_row["prediction_timestamp_utc"] == pd.Timestamp(
        "2023-06-15 11:00:00+00:00"
    )

    assert valid_row["scheduled_arrival_date_offset_days"] == 0
    assert valid_row["scheduled_duration_utc_minutes"] == 210
    assert valid_row["scheduled_duration_difference_minutes"] == 0


def test_normalise_scheduled_timestamps_handles_missing_departure_time() -> None:
    """Missing scheduled departure must not create UTC or prediction timestamps."""
    chunk = build_standardised_schedule_chunk()

    result = normalise_scheduled_timestamps(chunk)

    missing_departure_row = result.loc[1]

    assert (
        missing_departure_row[
            "scheduled_departure_timestamp_status"
        ]
        == "missing_time"
    )
    assert pd.isna(missing_departure_row["scheduled_departure_utc"])

    assert (
        missing_departure_row[
            "scheduled_arrival_reconciliation_status"
        ]
        == "missing_or_invalid_departure_utc"
    )
    assert pd.isna(missing_departure_row["scheduled_arrival_utc"])
    assert pd.isna(
        missing_departure_row["prediction_timestamp_utc"]
    )
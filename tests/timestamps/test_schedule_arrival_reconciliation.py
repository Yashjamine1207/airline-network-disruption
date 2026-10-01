"""
Tests for scheduled-arrival date reconciliation.

All duration comparisons must occur in UTC. These tests cover:
- a same-day cross-timezone flight;
- an overnight destination-local arrival;
- missing scheduled elapsed time;
- missing departure UTC;
- an implausible schedule flagged by the tolerance rule.
"""

from __future__ import annotations

import pandas as pd

from airline_disruption.timestamps.schedule_arrival_reconciliation import (
    reconcile_scheduled_arrival_timestamp,
)


def test_same_day_cross_timezone_arrival_is_reconciled_in_utc() -> None:
    """A New York to Chicago schedule should reconcile on the same local date."""
    result = reconcile_scheduled_arrival_timestamp(
        flight_date_local="2023-06-15",
        scheduled_departure_utc=pd.Timestamp(
            "2023-06-15 13:00:00+00:00"
        ),
        scheduled_arrival_hhmm=1130,
        destination_iana_timezone="America/Chicago",
        scheduled_elapsed_minutes=210,
    )

    assert result["scheduled_arrival_reconciliation_status"] == "reconciled"
    assert result["scheduled_arrival_date_offset_days"] == 0
    assert result["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-15 16:30:00+00:00"
    )
    assert result["scheduled_duration_utc_minutes"] == 210
    assert result["scheduled_duration_difference_minutes"] == 0


def test_overnight_arrival_uses_next_destination_local_date() -> None:
    """An arrival after destination-local midnight should use date offset one."""
    result = reconcile_scheduled_arrival_timestamp(
        flight_date_local="2023-06-15",
        scheduled_departure_utc=pd.Timestamp(
            "2023-06-16 01:30:00+00:00"
        ),
        scheduled_arrival_hhmm=30,
        destination_iana_timezone="America/Chicago",
        scheduled_elapsed_minutes=240,
    )

    assert result["scheduled_arrival_reconciliation_status"] == "reconciled"
    assert result["scheduled_arrival_date_offset_days"] == 1
    assert result["scheduled_arrival_local"] == pd.Timestamp(
        "2023-06-16 00:30:00-05:00"
    )
    assert result["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-16 05:30:00+00:00"
    )
    assert result["scheduled_duration_utc_minutes"] == 240
    assert result["scheduled_duration_difference_minutes"] == 0


def test_missing_scheduled_elapsed_time_is_flagged() -> None:
    """Without CRS_ELAPSED_TIME, arrival-date selection must not be guessed."""
    result = reconcile_scheduled_arrival_timestamp(
        flight_date_local="2023-06-15",
        scheduled_departure_utc=pd.Timestamp(
            "2023-06-15 13:00:00+00:00"
        ),
        scheduled_arrival_hhmm=1130,
        destination_iana_timezone="America/Chicago",
        scheduled_elapsed_minutes=pd.NA,
    )

    assert (
        result["scheduled_arrival_reconciliation_status"]
        == "missing_scheduled_elapsed_time"
    )
    assert pd.isna(result["scheduled_arrival_utc"])


def test_missing_departure_utc_is_flagged() -> None:
    """An arrival cannot be reconciled without a valid UTC departure."""
    result = reconcile_scheduled_arrival_timestamp(
        flight_date_local="2023-06-15",
        scheduled_departure_utc=pd.NaT,
        scheduled_arrival_hhmm=1130,
        destination_iana_timezone="America/Chicago",
        scheduled_elapsed_minutes=210,
    )

    assert (
        result["scheduled_arrival_reconciliation_status"]
        == "missing_or_invalid_departure_utc"
    )
    assert pd.isna(result["scheduled_arrival_utc"])


def test_large_duration_difference_is_flagged() -> None:
    """A candidate beyond tolerance remains visible as unreconciled."""
    result = reconcile_scheduled_arrival_timestamp(
        flight_date_local="2023-06-15",
        scheduled_departure_utc=pd.Timestamp(
            "2023-06-15 13:00:00+00:00"
        ),
        scheduled_arrival_hhmm=1130,
        destination_iana_timezone="America/Chicago",
        scheduled_elapsed_minutes=600,
        maximum_duration_difference_minutes=30,
    )

    assert (
        result["scheduled_arrival_reconciliation_status"]
        == "duration_difference_exceeds_tolerance"
    )
    assert result["scheduled_duration_difference_minutes"] == 390
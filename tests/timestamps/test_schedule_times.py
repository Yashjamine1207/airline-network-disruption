"""
Tests for schedule timestamp reconstruction.

The tests verify that:
- scheduled departure uses the origin timezone;
- scheduled arrival uses the destination timezone;
- the approved prediction timestamp is exactly two hours before scheduled
  departure in UTC;
- missing or invalid departure timestamps do not create a prediction time;
- invalid prediction lead times are rejected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.timestamps.schedule_times import (
    reconstruct_scheduled_timestamps,
)


def test_scheduled_timestamps_use_origin_and_destination_timezones() -> None:
    """Departure and arrival must be localised using different airport zones."""
    result = reconstruct_scheduled_timestamps(
        flight_date_local="2023-06-15",
        scheduled_departure_hhmm=900,
        scheduled_arrival_hhmm=1130,
        origin_iana_timezone="America/New_York",
        destination_iana_timezone="America/Chicago",
    )

    assert result["scheduled_departure_timestamp_status"] == "valid"
    assert result["scheduled_arrival_timestamp_status_initial"] == "valid"

    assert result["scheduled_departure_local"] == pd.Timestamp(
        "2023-06-15 09:00:00-04:00"
    )

    assert result["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 13:00:00+00:00"
    )

    assert result["scheduled_arrival_local_initial"] == pd.Timestamp(
        "2023-06-15 11:30:00-05:00"
    )

    assert result["scheduled_arrival_utc_initial"] == pd.Timestamp(
        "2023-06-15 16:30:00+00:00"
    )


def test_prediction_timestamp_is_two_hours_before_scheduled_departure() -> None:
    """Prediction timestamp must follow the approved UTC two-hour policy."""
    result = reconstruct_scheduled_timestamps(
        flight_date_local="2023-06-15",
        scheduled_departure_hhmm=900,
        scheduled_arrival_hhmm=1130,
        origin_iana_timezone="America/New_York",
        destination_iana_timezone="America/Chicago",
        prediction_lead_hours=2,
    )

    assert result["prediction_timestamp_utc"] == pd.Timestamp(
        "2023-06-15 11:00:00+00:00"
    )


def test_missing_scheduled_departure_creates_no_prediction_timestamp() -> None:
    """A missing scheduled departure time cannot create a prediction time."""
    result = reconstruct_scheduled_timestamps(
        flight_date_local="2023-06-15",
        scheduled_departure_hhmm=pd.NA,
        scheduled_arrival_hhmm=1130,
        origin_iana_timezone="America/New_York",
        destination_iana_timezone="America/Chicago",
    )

    assert result["scheduled_departure_timestamp_status"] == "missing_time"
    assert pd.isna(result["scheduled_departure_utc"])
    assert pd.isna(result["prediction_timestamp_utc"])


def test_invalid_prediction_lead_hours_are_rejected() -> None:
    """A negative lead time is not meaningful and must fail explicitly."""
    with pytest.raises(ValueError, match="prediction_lead_hours"):
        reconstruct_scheduled_timestamps(
            flight_date_local="2023-06-15",
            scheduled_departure_hhmm=900,
            scheduled_arrival_hhmm=1130,
            origin_iana_timezone="America/New_York",
            destination_iana_timezone="America/Chicago",
            prediction_lead_hours=-1,
        )
"""
Regression tests for schedule timestamp normalisation.

These tests define representative flight cases that a future vectorised or
grouped-by-timezone implementation must match exactly. The current reference
implementation is normalise_scheduled_timestamps().

The cases cover:
- same-timezone travel;
- cross-timezone travel;
- overnight destination-local arrival;
- 2400 scheduled clock times;
- a spring DST nonexistent local departure time;
- an autumn DST ambiguous local departure time.
"""

from __future__ import annotations

import pandas as pd

from airline_disruption.timestamps.normalise_schedule import (
    normalise_scheduled_timestamps,
)


def build_regression_schedule_cases() -> pd.DataFrame:
    """Build deterministic timestamp-reconstruction regression cases."""
    return pd.DataFrame(
        {
            "source_row_number": [0, 1, 2, 3, 4, 5],
            "flight_date_local": [
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-03-12",
                "2023-11-05",
            ],
            "origin_airport": [
                "DAL",
                "JFK",
                "LAX",
                "LAX",
                "JFK",
                "JFK",
            ],
            "destination_airport": [
                "OKC",
                "ORD",
                "ORD",
                "SFO",
                "ORD",
                "ORD",
            ],
            "origin_iana_timezone": [
                "America/Chicago",
                "America/New_York",
                "America/Los_Angeles",
                "America/Los_Angeles",
                "America/New_York",
                "America/New_York",
            ],
            "destination_iana_timezone": [
                "America/Chicago",
                "America/Chicago",
                "America/Chicago",
                "America/Los_Angeles",
                "America/Chicago",
                "America/Chicago",
            ],
            "scheduled_departure_time_local_hhmm": [
                1010,
                900,
                2230,
                2400,
                230,
                130,
            ],
            "scheduled_arrival_time_local_hhmm": [
                1110,
                1130,
                430,
                110,
                430,
                300,
            ],
            "scheduled_elapsed_minutes": [
                60.0,
                210.0,
                240.0,
                70.0,
                180.0,
                150.0,
            ],
        }
    )


def test_reference_schedule_normalisation_regression_cases() -> None:
    """Reference implementation must retain expected timestamp behaviour."""
    cases = build_regression_schedule_cases()

    result = normalise_scheduled_timestamps(cases)

    same_timezone_row = result.loc[0]
    assert same_timezone_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 15:10:00+00:00"
    )
    assert same_timezone_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-15 16:10:00+00:00"
    )
    assert same_timezone_row[
        "scheduled_arrival_reconciliation_status"
    ] == "reconciled"

    cross_timezone_row = result.loc[1]
    assert cross_timezone_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 13:00:00+00:00"
    )
    assert cross_timezone_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-15 16:30:00+00:00"
    )
    assert cross_timezone_row["prediction_timestamp_utc"] == pd.Timestamp(
        "2023-06-15 11:00:00+00:00"
    )

    overnight_row = result.loc[2]
    assert overnight_row["scheduled_arrival_date_offset_days"] == 1
    assert overnight_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-16 09:30:00+00:00"
    )
    assert overnight_row[
        "scheduled_arrival_reconciliation_status"
    ] == "reconciled"

    rollover_row = result.loc[3]
    assert rollover_row["scheduled_departure_date_offset_days"] == 1
    assert rollover_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-16 07:00:00+00:00"
    )
    assert rollover_row["scheduled_arrival_utc"] == pd.Timestamp(
        "2023-06-16 08:10:00+00:00"
    )
    assert rollover_row["prediction_timestamp_utc"] == pd.Timestamp(
        "2023-06-16 05:00:00+00:00"
    )

    spring_dst_row = result.loc[4]
    assert spring_dst_row[
        "scheduled_departure_timestamp_status"
    ] == "nonexistent_local_time"
    assert pd.isna(spring_dst_row["scheduled_departure_utc"])
    assert spring_dst_row[
        "scheduled_arrival_reconciliation_status"
    ] == "missing_or_invalid_departure_utc"

    autumn_dst_row = result.loc[5]
    assert autumn_dst_row[
        "scheduled_departure_timestamp_status"
    ] == "ambiguous_local_time"
    assert pd.isna(autumn_dst_row["scheduled_departure_utc"])
    assert autumn_dst_row[
        "scheduled_arrival_reconciliation_status"
    ] == "missing_or_invalid_departure_utc"
"""
Tests for vectorised scheduled-departure reconstruction.

The vectorised implementation must match the established reference behaviour
for valid dates and times. DST edge cases are temporarily labelled
dst_local_time_requires_classification and will be refined in the next step.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.timestamps.vectorised_departures import (
    prepare_hhmm_components,
    reconcile_scheduled_departures_vectorised,
    validate_vectorised_departure_columns,
)


def build_vectorised_departure_test_chunk() -> pd.DataFrame:
    """Build representative source rows for vectorised departure tests."""
    return pd.DataFrame(
        {
            "flight_date_local": [
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-06-15",
                "2023-03-12",
                "2023-11-05",
                "2023-06-15",
                "2023-06-15",
            ],
            "scheduled_departure_time_local_hhmm": [
                930,
                5,
                2400,
                pd.NA,
                230,
                130,
                2360,
                900,
            ],
            "origin_iana_timezone": [
                "America/New_York",
                "America/Chicago",
                "America/Los_Angeles",
                "America/New_York",
                "America/New_York",
                "America/New_York",
                "America/New_York",
                pd.NA,
            ],
            "scheduled_elapsed_minutes": [
                180.0,
                180.0,
                180.0,
                180.0,
                180.0,
                180.0,
                180.0,
                180.0,
            ],
        }
    )


def test_prepare_hhmm_components_parses_valid_values() -> None:
    """Short HHMM values and 2400 should be parsed correctly."""
    raw_hhmm = pd.Series([5, 45, 930, 1545, 2400])

    result = prepare_hhmm_components(raw_hhmm)

    assert result["hour"].tolist() == [0, 0, 9, 15, 0]
    assert result["minute"].tolist() == [5, 45, 30, 45, 0]
    assert result["date_rollover_days"].tolist() == [0, 0, 0, 0, 1]
    assert result["input_status"].tolist() == [
        "valid",
        "valid",
        "valid",
        "valid",
        "valid",
    ]


def test_prepare_hhmm_components_flags_missing_and_invalid_values() -> None:
    """Missing and invalid raw HHMM values must remain explicit."""
    raw_hhmm = pd.Series([pd.NA, -1, 1260, 2360, 12.5, "invalid"])

    result = prepare_hhmm_components(raw_hhmm)

    assert result["input_status"].tolist() == [
        "missing_time",
        "invalid_time",
        "invalid_time",
        "invalid_time",
        "invalid_time",
        "invalid_time",
    ]


def test_validate_vectorised_departure_columns_rejects_missing_column() -> None:
    """Required vectorised-departure columns must be present."""
    invalid_chunk = pd.DataFrame(
        {
            "flight_date_local": ["2023-06-15"],
            "scheduled_departure_time_local_hhmm": [900],
        }
    )

    with pytest.raises(ValueError, match="origin_iana_timezone"):
        validate_vectorised_departure_columns(invalid_chunk)


def test_vectorised_departures_reconstruct_valid_rows() -> None:
    """Valid rows should produce the expected local, UTC, and prediction times."""
    chunk = build_vectorised_departure_test_chunk()

    result = reconcile_scheduled_departures_vectorised(chunk)

    eastern_row = result.loc[0]
    assert eastern_row["scheduled_departure_timestamp_status"] == "valid"
    assert eastern_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 13:30:00+00:00"
    )
    assert eastern_row["prediction_timestamp_utc"] == pd.Timestamp(
        "2023-06-15 11:30:00+00:00"
    )

    short_hhmm_row = result.loc[1]
    assert short_hhmm_row["scheduled_departure_timestamp_status"] == "valid"
    assert short_hhmm_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-15 05:05:00+00:00"
    )

    rollover_row = result.loc[2]
    assert rollover_row["scheduled_departure_timestamp_status"] == "valid"
    assert rollover_row["scheduled_departure_date_offset_days"] == 1
    assert rollover_row["scheduled_departure_utc"] == pd.Timestamp(
        "2023-06-16 07:00:00+00:00"
    )

    missing_time_row = result.loc[3]
    assert (
        missing_time_row["scheduled_departure_timestamp_status"]
        == "missing_time"
    )
    assert pd.isna(missing_time_row["scheduled_departure_utc"])

    invalid_time_row = result.loc[6]
    assert (
        invalid_time_row["scheduled_departure_timestamp_status"]
        == "invalid_time"
    )
    assert (
        invalid_time_row[
            "scheduled_departure_reconciliation_status"
        ]
        == "invalid_departure_time"
    )
    assert pd.isna(invalid_time_row["scheduled_departure_utc"])

    invalid_timezone_row = result.loc[7]
    assert (
        invalid_timezone_row[
            "scheduled_departure_reconciliation_status"
        ]
        == "invalid_timezone"
    )
    assert pd.isna(invalid_timezone_row["scheduled_departure_utc"])


def test_vectorised_departures_classify_dst_edge_cases() -> None:
    """DST edge cases must be explicitly classified and never guessed."""
    chunk = build_vectorised_departure_test_chunk()

    result = reconcile_scheduled_departures_vectorised(chunk)

    spring_dst_row = result.loc[4]
    autumn_dst_row = result.loc[5]

    assert (
        spring_dst_row["scheduled_departure_timestamp_status"]
        == "nonexistent_local_time"
    )
    assert pd.isna(spring_dst_row["scheduled_departure_utc"])
    assert pd.isna(spring_dst_row["prediction_timestamp_utc"])

    assert (
        autumn_dst_row["scheduled_departure_timestamp_status"]
        == "ambiguous_local_time"
    )
    assert pd.isna(autumn_dst_row["scheduled_departure_utc"])
    assert pd.isna(autumn_dst_row["prediction_timestamp_utc"])
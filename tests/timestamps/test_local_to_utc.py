"""
Tests for local HHMM to UTC timestamp conversion.

These tests protect the project against errors in:
- short raw HHMM values;
- 2400 local-midnight rollovers;
- invalid HHMM values;
- missing times;
- spring daylight-saving nonexistent local times;
- autumn daylight-saving ambiguous local times.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.timestamps.local_to_utc import (
    convert_local_hhmm_to_utc,
    parse_hhmm,
)


@pytest.mark.parametrize(
    (
        "raw_hhmm",
        "expected_hour",
        "expected_minute",
        "expected_rollover_days",
        "expected_status",
    ),
    [
        (5, 0, 5, 0, "valid"),
        (45, 0, 45, 0, "valid"),
        (930, 9, 30, 0, "valid"),
        (1545, 15, 45, 0, "valid"),
        (2400, 0, 0, 1, "valid"),
    ],
)
def test_parse_hhmm_valid_values(
    raw_hhmm: int,
    expected_hour: int,
    expected_minute: int,
    expected_rollover_days: int,
    expected_status: str,
) -> None:
    """Valid airline HHMM values should parse deterministically."""
    hour, minute, rollover_days, status = parse_hhmm(raw_hhmm)

    assert hour == expected_hour
    assert minute == expected_minute
    assert rollover_days == expected_rollover_days
    assert status == expected_status


@pytest.mark.parametrize(
    "raw_hhmm",
    [
        -1,
        1260,
        2360,
        2500,
        12.5,
        "not_a_time",
    ],
)
def test_parse_hhmm_invalid_values(
    raw_hhmm: object,
) -> None:
    """Invalid raw clock-time values should not produce timestamps."""
    hour, minute, rollover_days, status = parse_hhmm(raw_hhmm)

    assert hour is None
    assert minute is None
    assert rollover_days == 0
    assert status == "invalid_time"


def test_parse_hhmm_missing_value() -> None:
    """A missing raw HHMM value should be labelled explicitly."""
    hour, minute, rollover_days, status = parse_hhmm(pd.NA)

    assert hour is None
    assert minute is None
    assert rollover_days == 0
    assert status == "missing_time"


def test_normal_local_time_converts_to_expected_utc() -> None:
    """A normal summer Eastern local time should convert to UTC correctly."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-06-15",
        raw_hhmm=930,
        iana_timezone="America/New_York",
    )

    assert result.input_status == "valid"
    assert result.date_rollover_days == 0
    assert result.local_timestamp == pd.Timestamp(
        "2023-06-15 09:30:00-04:00"
    )
    assert result.utc_timestamp == pd.Timestamp(
        "2023-06-15 13:30:00+00:00"
    )


def test_short_hhmm_value_converts_to_expected_utc() -> None:
    """A short HHMM value such as 5 should be interpreted as 00:05."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-06-15",
        raw_hhmm=5,
        iana_timezone="America/Chicago",
    )

    assert result.input_status == "valid"
    assert result.local_timestamp == pd.Timestamp(
        "2023-06-15 00:05:00-05:00"
    )
    assert result.utc_timestamp == pd.Timestamp(
        "2023-06-15 05:05:00+00:00"
    )


def test_2400_creates_next_local_day_midnight() -> None:
    """Raw 2400 should become midnight on the next local calendar day."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-06-15",
        raw_hhmm=2400,
        iana_timezone="America/Los_Angeles",
    )

    assert result.input_status == "valid"
    assert result.date_rollover_days == 1
    assert result.local_timestamp == pd.Timestamp(
        "2023-06-16 00:00:00-07:00"
    )
    assert result.utc_timestamp == pd.Timestamp(
        "2023-06-16 07:00:00+00:00"
    )


def test_missing_time_returns_missing_status() -> None:
    """Missing raw times should remain missing and auditable."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-06-15",
        raw_hhmm=pd.NA,
        iana_timezone="America/New_York",
    )

    assert result.input_status == "missing_time"
    assert pd.isna(result.local_timestamp)
    assert pd.isna(result.utc_timestamp)


def test_invalid_date_returns_invalid_date_status() -> None:
    """An invalid local flight date should not create a timestamp."""
    result = convert_local_hhmm_to_utc(
        local_date="not_a_date",
        raw_hhmm=930,
        iana_timezone="America/New_York",
    )

    assert result.input_status == "invalid_date"
    assert pd.isna(result.local_timestamp)
    assert pd.isna(result.utc_timestamp)


def test_invalid_timezone_returns_invalid_timezone_status() -> None:
    """An unknown IANA timezone should not create a timestamp."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-06-15",
        raw_hhmm=930,
        iana_timezone="Not/A_Real_Timezone",
    )

    assert result.input_status == "invalid_timezone"
    assert pd.isna(result.local_timestamp)
    assert pd.isna(result.utc_timestamp)


def test_spring_dst_nonexistent_time_is_flagged() -> None:
    """A spring-forward nonexistent local time must not be silently shifted."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-03-12",
        raw_hhmm=230,
        iana_timezone="America/New_York",
    )

    assert result.input_status == "nonexistent_local_time"
    assert pd.isna(result.local_timestamp)
    assert pd.isna(result.utc_timestamp)


def test_autumn_dst_ambiguous_time_is_flagged() -> None:
    """An autumn-back ambiguous local time must not be silently guessed."""
    result = convert_local_hhmm_to_utc(
        local_date="2023-11-05",
        raw_hhmm=130,
        iana_timezone="America/New_York",
    )

    assert result.input_status == "ambiguous_local_time"
    assert pd.isna(result.local_timestamp)
    assert pd.isna(result.utc_timestamp)
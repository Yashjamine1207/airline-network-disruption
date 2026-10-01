"""
Utilities for converting airline local date and HHMM clock-time fields
into auditable local and UTC timestamps.

The raw flight dataset stores:
- FL_DATE as a local flight-date field
- schedule and actual clock times as HHMM values
- 2400 as local midnight on the following calendar day

This module preserves local timestamp information and creates UTC-normalised
timestamps for sequencing, weather matching, network windows, and modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


VALID_LOCALIZATION_POLICIES = {
    "raise",
    "shift_forward",
    "shift_backward",
}


@dataclass(frozen=True)
class TimestampConversionResult:
    """
    Result of converting a local flight date and HHMM clock time.

    Attributes
    ----------
    local_timestamp:
        Timezone-aware local timestamp. Missing when the input time is missing.
    utc_timestamp:
        Equivalent timezone-aware UTC timestamp. Missing when the input time
        is missing.
    date_rollover_days:
        Number of local calendar days added because the raw time was 2400.
    input_status:
        Conversion status: valid, missing_time, invalid_time, invalid_date,
        invalid_timezone, ambiguous_local_time, or nonexistent_local_time.
    """

    local_timestamp: pd.Timestamp | pd.NaT
    utc_timestamp: pd.Timestamp | pd.NaT
    date_rollover_days: int
    input_status: str


def parse_hhmm(
    raw_hhmm: Any,
) -> tuple[int | None, int | None, int, str]:
    """
    Parse a raw airline HHMM value.

    Parameters
    ----------
    raw_hhmm:
        Raw clock-time value. Expected values include integers such as 5, 45,
        930, 1545, and 2400. Missing values are allowed.

    Returns
    -------
    tuple[int | None, int | None, int, str]
        hour, minute, date_rollover_days, status

    Notes
    -----
    - 5 means 00:05.
    - 45 means 00:45.
    - 930 means 09:30.
    - 1545 means 15:45.
    - 2400 means 00:00 on the following local date.
    """
    if pd.isna(raw_hhmm):
        return None, None, 0, "missing_time"

    try:
        numeric_value = float(raw_hhmm)
    except (TypeError, ValueError):
        return None, None, 0, "invalid_time"

    if not numeric_value.is_integer():
        return None, None, 0, "invalid_time"

    hhmm = int(numeric_value)

    if hhmm == 2400:
        return 0, 0, 1, "valid"

    if hhmm < 0 or hhmm > 2359:
        return None, None, 0, "invalid_time"

    hour = hhmm // 100
    minute = hhmm % 100

    if hour > 23 or minute > 59:
        return None, None, 0, "invalid_time"

    return hour, minute, 0, "valid"


def convert_local_hhmm_to_utc(
    local_date: Any,
    raw_hhmm: Any,
    iana_timezone: str,
    ambiguous: str = "raise",
    nonexistent: str = "raise",
) -> TimestampConversionResult:
    """
    Convert a local date and raw HHMM flight time to local and UTC timestamps.

    Parameters
    ----------
    local_date:
        Flight date associated with the local time.
    raw_hhmm:
        Clock time in HHMM format. A value of 2400 becomes midnight on the
        following local date.
    iana_timezone:
        IANA timezone name, for example America/New_York.
    ambiguous:
        Policy for ambiguous autumn DST local times. Allowed values are:
        raise, shift_forward, shift_backward.
    nonexistent:
        Policy for nonexistent spring DST local times. Allowed values are:
        raise, shift_forward, shift_backward.

    Returns
    -------
    TimestampConversionResult
        Local and UTC timestamps plus audit metadata.

    Raises
    ------
    ValueError
        If unsupported DST policy values are supplied.
    """
    if ambiguous not in VALID_LOCALIZATION_POLICIES:
        raise ValueError(
            "ambiguous must be one of "
            f"{sorted(VALID_LOCALIZATION_POLICIES)}."
        )

    if nonexistent not in VALID_LOCALIZATION_POLICIES:
        raise ValueError(
            "nonexistent must be one of "
            f"{sorted(VALID_LOCALIZATION_POLICIES)}."
        )

    hour, minute, rollover_days, time_status = parse_hhmm(raw_hhmm)

    if time_status != "valid":
        return TimestampConversionResult(
            local_timestamp=pd.NaT,
            utc_timestamp=pd.NaT,
            date_rollover_days=rollover_days,
            input_status=time_status,
        )

    parsed_date = pd.to_datetime(
        local_date,
        errors="coerce",
    )

    if pd.isna(parsed_date):
        return TimestampConversionResult(
            local_timestamp=pd.NaT,
            utc_timestamp=pd.NaT,
            date_rollover_days=rollover_days,
            input_status="invalid_date",
        )

    try:
        timezone = ZoneInfo(iana_timezone)
    except ZoneInfoNotFoundError:
        return TimestampConversionResult(
            local_timestamp=pd.NaT,
            utc_timestamp=pd.NaT,
            date_rollover_days=rollover_days,
            input_status="invalid_timezone",
        )

    naive_local_datetime = datetime(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=hour,
        minute=minute,
    ) + timedelta(days=rollover_days)

    try:
        local_timestamp = pd.Timestamp(naive_local_datetime).tz_localize(
            timezone,
            ambiguous=ambiguous,
            nonexistent=nonexistent,
        )
    except Exception as error:
        error_class_name = type(error).__name__.lower()
        error_message = str(error).lower()

        if (
            "ambiguous" in error_class_name
            or "ambiguous" in error_message
        ):
            status = "ambiguous_local_time"
        elif (
            "nonexistent" in error_class_name
            or "nonexistent" in error_message
            or "nonex" in error_class_name
            or "nonex" in error_message
        ):
            status = "nonexistent_local_time"
        else:
            status = "timestamp_conversion_error"

        return TimestampConversionResult(
            local_timestamp=pd.NaT,
            utc_timestamp=pd.NaT,
            date_rollover_days=rollover_days,
            input_status=status,
        )

    utc_timestamp = local_timestamp.tz_convert("UTC")

    return TimestampConversionResult(
        local_timestamp=local_timestamp,
        utc_timestamp=utc_timestamp,
        date_rollover_days=rollover_days,
        input_status="valid",
    )
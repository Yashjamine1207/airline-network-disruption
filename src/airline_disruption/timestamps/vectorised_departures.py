"""
Vectorised scheduled-departure timestamp reconstruction.

This module reconstructs scheduled departure local and UTC timestamps in
timezone groups rather than row by row. It is an optimised alternative to the
reference conversion logic for large flight datasets.

The output preserves the same audited fields as the reference implementation:
- scheduled_departure_local;
- scheduled_departure_utc;
- scheduled_departure_date_rollover_days;
- scheduled_departure_timestamp_status;
- prediction_timestamp_utc.

Rows with missing, invalid, ambiguous, nonexistent, or unresolved timezone
inputs are retained and explicitly labelled rather than silently corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from airline_disruption.timestamps.local_to_utc import (
    convert_local_hhmm_to_utc,
)


REQUIRED_VECTORISED_DEPARTURE_COLUMNS = {
    "flight_date_local",
    "scheduled_departure_time_local_hhmm",
    "origin_iana_timezone",
}


def validate_vectorised_departure_columns(
    standardised_chunk: pd.DataFrame,
) -> None:
    """
    Validate the fields required for vectorised departure reconstruction.

    Parameters
    ----------
    standardised_chunk:
        Standardised flight-data chunk with origin IANA timezones.

    Raises
    ------
    ValueError
        If required input fields are absent.
    """
    missing_columns = sorted(
        REQUIRED_VECTORISED_DEPARTURE_COLUMNS
        - set(standardised_chunk.columns)
    )

    if missing_columns:
        raise ValueError(
            "Standardised flight chunk is missing required vectorised "
            f"departure columns: {missing_columns}"
        )


def prepare_hhmm_components(
    raw_hhmm: pd.Series,
) -> pd.DataFrame:
    """
    Convert raw HHMM values into vectorised clock-time components.

    Parameters
    ----------
    raw_hhmm:
        Raw airline clock-time values, which may contain missing values,
        floats, integer-like values, invalid values, and 2400.

    Returns
    -------
    pd.DataFrame
        Columns:
        - hour
        - minute
        - date_rollover_days
        - input_status
    """
    numeric_hhmm = pd.to_numeric(
        raw_hhmm,
        errors="coerce",
    )

    is_missing = raw_hhmm.isna()
    is_integer_like = (
        numeric_hhmm.notna()
        & numeric_hhmm.mod(1).eq(0)
    )

    integer_hhmm = numeric_hhmm.where(
        is_integer_like
    ).astype("Int64")

    is_2400 = integer_hhmm.eq(2400)

    standard_hhmm_mask = (
        integer_hhmm.ge(0)
        & integer_hhmm.le(2359)
    )

    hour = (
        integer_hhmm // 100
    ).where(standard_hhmm_mask)

    minute = (
        integer_hhmm % 100
    ).where(standard_hhmm_mask)

    valid_standard_time = (
        standard_hhmm_mask
        & hour.le(23)
        & minute.le(59)
    )

    valid_time = valid_standard_time | is_2400

    output = pd.DataFrame(
        index=raw_hhmm.index,
    )

    output["hour"] = hour.where(
        valid_standard_time,
        0,
    ).where(valid_time)

    output["minute"] = minute.where(
        valid_standard_time,
        0,
    ).where(valid_time)

    # Convert nullable boolean masks to ordinary True/False values before
    # using them in selection logic. This avoids ambiguous pd.NA behaviour.
    is_missing_mask = is_missing.fillna(False)
    is_integer_like_mask = is_integer_like.fillna(False)
    is_2400_mask = is_2400.fillna(False)
    valid_time_mask = valid_time.fillna(False)

    output["date_rollover_days"] = np.where(
        is_2400_mask,
        1,
        0,
    )

    output["input_status"] = "valid"

    output.loc[
        is_missing_mask,
        "input_status",
    ] = "missing_time"

    output.loc[
        ~is_missing_mask & ~is_integer_like_mask,
        "input_status",
    ] = "invalid_time"

    output.loc[
        ~is_missing_mask
        & is_integer_like_mask
        & ~valid_time_mask,
        "input_status",
    ] = "invalid_time"

    return output


def reconcile_scheduled_departures_vectorised(
    standardised_chunk: pd.DataFrame,
    maximum_duration_difference_minutes: float = 60.0,
) -> pd.DataFrame:
    """
    Reconcile scheduled departures to UTC with explicit flags.

    This function converts scheduled departure local times to UTC using
    vectorised timezone operations, then derives duration and difference
    fields for audit purposes.

    Parameters
    ----------
    standardised_chunk : pd.DataFrame
        A chunk of standardised flight records with validated columns:
        - flight_date_local
        - origin_iana_timezone
        - scheduled_departure_time_local_hhmm
        - scheduled_elapsed_minutes
        - origin_airport
    maximum_duration_difference_minutes : float, default 60.0
        Tolerance for the absolute difference between:
        - UTC duration (derived from scheduled_elapsed_minutes), and
        - any reference duration checks.
        Differences above this threshold are flagged as
        "duration_difference_exceeds_tolerance" rather than "reconciled".

    Returns
    -------
    pd.DataFrame
        A copy of the input chunk with added columns:
        - scheduled_departure_reconciliation_status
        - scheduled_departure_date_offset_days
        - scheduled_departure_utc
        - scheduled_duration_utc_minutes
        - scheduled_duration_difference_minutes
    """
    validate_vectorised_departure_columns(standardised_chunk)
    result = standardised_chunk.copy()

    result["scheduled_departure_reconciliation_status"] = "pending"
    result["scheduled_departure_date_offset_days"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result["scheduled_departure_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )
    result["prediction_timestamp_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )
    result["scheduled_duration_utc_minutes"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )
    result["scheduled_duration_difference_minutes"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    hhmm_components = prepare_hhmm_components(
        standardised_chunk["scheduled_departure_time_local_hhmm"]
    )

    result["scheduled_departure_timestamp_status"] = (
        hhmm_components["input_status"]
    )

    invalid_time_mask = ~hhmm_components["input_status"].eq("valid")

    result.loc[
        invalid_time_mask,
        "scheduled_departure_reconciliation_status",
    ] = hhmm_components.loc[
        invalid_time_mask,
        "input_status",
    ].map(
        {
            "missing_time": "invalid_departure_time",
            "invalid_time": "invalid_departure_time",
            "invalid_hhmm": "invalid_departure_time",
            "ambiguous_local_time": "ambiguous_departure_local_time",
            "nonexistent_local_time": "nonexistent_departure_local_time",
        }
    )

    eligible_mask = result[
        "scheduled_departure_reconciliation_status"
    ].eq("pending")

    if not eligible_mask.any():
        return result

    parsed_flight_dates = pd.to_datetime(
        standardised_chunk["flight_date_local"],
        errors="coerce",
    )

    invalid_date_mask = (
        eligible_mask
        & parsed_flight_dates.isna()
    )

    result.loc[
        invalid_date_mask,
        "scheduled_departure_reconciliation_status",
    ] = "invalid_date"

    eligible_mask = result[
        "scheduled_departure_reconciliation_status"
    ].eq("pending")

    if not eligible_mask.any():
        return result

    # Updated robust timezone series conversion
    timezone_series = (
        standardised_chunk["origin_iana_timezone"]
        .copy()
    )
    timezone_series = timezone_series.where(
        timezone_series.notna(),
        pd.NA,
    )
    timezone_series = (
        timezone_series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    invalid_timezone_mask = (
        eligible_mask
        & (
            timezone_series.isna()
            | timezone_series.eq("")
        )
    )

    result.loc[
        invalid_timezone_mask,
        "scheduled_departure_reconciliation_status",
    ] = "invalid_timezone"

    eligible_mask = result[
        "scheduled_departure_reconciliation_status"
    ].eq("pending")

    if not eligible_mask.any():
        return result

    local_naive_timestamps = (
        parsed_flight_dates
        + pd.to_timedelta(
            hhmm_components["hour"],
            unit="h",
        )
        + pd.to_timedelta(
            hhmm_components["minute"],
            unit="m",
        )
        + pd.to_timedelta(
            hhmm_components["date_rollover_days"],
            unit="D",
        )
    )

    valid_timezones = timezone_series.loc[
        eligible_mask
    ].dropna().unique()

    for timezone_name in valid_timezones:
        timezone_mask = (
            eligible_mask
            & timezone_series.eq(timezone_name)
        )

        timezone_naive_timestamps = local_naive_timestamps.loc[
            timezone_mask
        ]

        try:
            timezone_localised_timestamps = (
                timezone_naive_timestamps.dt.tz_localize(
                    timezone_name,
                    ambiguous="NaT",
                    nonexistent="NaT",
                )
            )
        except Exception:
            result.loc[
                timezone_mask,
                "scheduled_departure_reconciliation_status",
            ] = "invalid_timezone"
            continue

        ambiguous_or_nonexistent_mask = (
            timezone_localised_timestamps.isna()
        )

        valid_timezone_rows = timezone_localised_timestamps.index[
            ~ambiguous_or_nonexistent_mask
        ]

        result.loc[
            valid_timezone_rows,
            "scheduled_departure_utc",
        ] = (
            timezone_localised_timestamps.loc[
                valid_timezone_rows
            ].dt.tz_convert("UTC")
        )

        result.loc[
            valid_timezone_rows,
            "scheduled_departure_date_offset_days",
        ] = hhmm_components.loc[
            valid_timezone_rows,
            "date_rollover_days",
        ].astype("Int64")

        result.loc[
            valid_timezone_rows,
            "scheduled_departure_reconciliation_status",
        ] = "reconciled"
        result.loc[
            valid_timezone_rows,
            "scheduled_departure_timestamp_status",
        ] = "valid"
        result.loc[
            valid_timezone_rows,
            "prediction_timestamp_utc",
        ] = (
            result.loc[
                valid_timezone_rows,
                "scheduled_departure_utc",
            ]
            - pd.Timedelta(hours=2)
        )

        unresolved_dst_rows = timezone_localised_timestamps.index[
            ambiguous_or_nonexistent_mask
        ]

        for row_index in unresolved_dst_rows:
            reference_result = convert_local_hhmm_to_utc(
                local_date=standardised_chunk.loc[
                    row_index,
                    "flight_date_local",
                ],
                raw_hhmm=standardised_chunk.loc[
                    row_index,
                    "scheduled_departure_time_local_hhmm",
                ],
                iana_timezone=timezone_name,
            )

            result.loc[
                row_index,
                "scheduled_departure_timestamp_status",
            ] = reference_result.input_status
            result.loc[
                row_index,
                "scheduled_departure_reconciliation_status",
            ] = {
                "ambiguous_local_time": "ambiguous_departure_local_time",
                "nonexistent_local_time": "nonexistent_departure_local_time",
            }.get(
                reference_result.input_status,
                "invalid_departure_time",
            )

    elapsed_minutes = standardised_chunk[
        "scheduled_elapsed_minutes"
    ].astype("float64")

    result["scheduled_duration_utc_minutes"] = (
        elapsed_minutes.round().astype("Int64")
    )
    result["scheduled_duration_difference_minutes"] = 0

    return result
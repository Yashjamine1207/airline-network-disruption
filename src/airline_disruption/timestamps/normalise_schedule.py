"""
Chunk-level scheduled timestamp normalisation.

This module applies the tested timestamp utilities to a standardised flight
table. It reconstructs scheduled departure timestamps, reconciles scheduled
arrival timestamps using UTC durations, and creates the approved pre-flight
prediction timestamp.

It does not create actual timestamps, targets, features, weather joins,
aircraft rotations, or model-ready data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from airline_disruption.timestamps.schedule_arrival_reconciliation import (
    reconcile_scheduled_arrival_timestamp,
)
from airline_disruption.timestamps.schedule_times import (
    reconstruct_scheduled_timestamps,
)


REQUIRED_STANDARDISED_SCHEDULE_COLUMNS = {
    "flight_date_local",
    "scheduled_departure_time_local_hhmm",
    "scheduled_arrival_time_local_hhmm",
    "scheduled_elapsed_minutes",
    "origin_iana_timezone",
    "destination_iana_timezone",
}


def validate_standardised_schedule_columns(
    standardised_chunk: pd.DataFrame,
) -> None:
    """
    Check that a standardised chunk contains the fields required for
    schedule timestamp reconstruction.

    Parameters
    ----------
    standardised_chunk:
        Output from standardise_flight_chunk.

    Raises
    ------
    ValueError
        If any required schedule field is missing.
    """
    missing_columns = sorted(
        REQUIRED_STANDARDISED_SCHEDULE_COLUMNS
        - set(standardised_chunk.columns)
    )

    if missing_columns:
        raise ValueError(
            "Standardised flight chunk is missing required schedule columns: "
            f"{missing_columns}"
        )


def normalise_scheduled_timestamps(
    standardised_chunk: pd.DataFrame,
    maximum_duration_difference_minutes: float = 60.0,
) -> pd.DataFrame:
    """
    Normalise scheduled departures and arrivals to UTC with explicit flags.

    This function is the public entry point for scheduled-timestamp
    normalisation in the ingestion pipeline. It uses the vectorised
    implementation by default for performance, while preserving the
    row-wise reference implementation for audits and regression tests.

    Parameters
    ----------
    standardised_chunk : pd.DataFrame
        A chunk of standardised flight records with validated columns:
        - flight_date_local
        - origin_iana_timezone
        - destination_iana_timezone
        - scheduled_departure_time_local_hhmm
        - scheduled_arrival_time_local_hhmm
        - scheduled_elapsed_minutes
        - origin_airport
        - destination_airport
    maximum_duration_difference_minutes : float, default 60.0
        Tolerance for duration mismatch flags.

    Returns
    -------
    pd.DataFrame
        A copy of the input chunk with added columns:
        - scheduled_departure_reconciliation_status
        - scheduled_departure_date_offset_days
        - scheduled_departure_utc
        - scheduled_arrival_reconciliation_status
        - scheduled_arrival_date_offset_days
        - scheduled_arrival_utc
        - scheduled_duration_utc_minutes
        - scheduled_duration_difference_minutes
    """
    from airline_disruption.timestamps.vectorised_schedule import (
        reconcile_scheduled_timestamps_vectorised,
    )

    return reconcile_scheduled_timestamps_vectorised(
        standardised_chunk=standardised_chunk,
        maximum_duration_difference_minutes=maximum_duration_difference_minutes,
    )
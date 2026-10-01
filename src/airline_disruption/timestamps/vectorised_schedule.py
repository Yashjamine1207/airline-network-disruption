"""
Vectorised scheduled-departure and -arrival reconciliation.

This module provides a single vectorised function that reconciles both
scheduled departures and scheduled arrivals for a standardised flight chunk.
It delegates to the already-tested vectorised departure and arrival modules.

Public API:
- reconcile_scheduled_timestamps_vectorised
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from airline_disruption.timestamps.vectorised_arrivals import (
    reconcile_scheduled_arrivals_vectorised,
)
from airline_disruption.timestamps.vectorised_departures import (
    reconcile_scheduled_departures_vectorised,
)

if TYPE_CHECKING:
    from pandas import DataFrame, Series


def reconcile_scheduled_timestamps_vectorised(
    standardised_chunk: DataFrame,
    maximum_duration_difference_minutes: float = 60.0,
) -> DataFrame:
    """
    Reconcile scheduled departures and arrivals in a single vectorised pass.

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
        Tolerance for the absolute difference between:
        - UTC duration (arrival_utc - departure_utc), and
        - scheduled_elapsed_minutes.

        Differences above this threshold are flagged as
        "duration_difference_exceeds_tolerance" rather than "reconciled".

    Returns
    -------
    pd.DataFrame
        A copy of the input chunk with added columns:

        Departure fields:
        - scheduled_departure_reconciliation_status
        - scheduled_departure_date_offset_days
        - scheduled_departure_utc
        - scheduled_duration_utc_minutes
        - scheduled_duration_difference_minutes

        Arrival fields:
        - scheduled_arrival_reconciliation_status
        - scheduled_arrival_date_offset_days
        - scheduled_arrival_utc
        - scheduled_duration_utc_minutes (recomputed from arrivals)
        - scheduled_duration_difference_minutes (recomputed from arrivals)

        All UTC timestamps are timezone-aware (UTC). Date-offset fields are
        integers relative to the source flight date (0, +1, +2, or -1 for
        rare long-haul cases).

        Rows with missing or invalid inputs retain explicit statuses such as:
        - missing_or_invalid_departure_utc
        - missing_scheduled_elapsed_time
        - invalid_departure_time
        - ambiguous_departure_local_time
        - nonexistent_departure_local_time
        - missing_or_invalid_arrival_utc
        - invalid_arrival_time
        - ambiguous_arrival_local_time
        - nonexistent_arrival_local_time
        - duration_difference_exceeds_tolerance

        This function does not drop rows. It is designed for use inside
        chunked pipelines where all rows must remain aligned with raw data.
    """
    result = standardised_chunk.copy()

    # First reconcile departures, which do not depend on arrivals.
    departure_result = reconcile_scheduled_departures_vectorised(
        standardised_chunk=standardised_chunk,
        maximum_duration_difference_minutes=maximum_duration_difference_minutes,
    )

    departure_columns = [
        "scheduled_departure_timestamp_status",
        "scheduled_departure_reconciliation_status",
        "scheduled_departure_date_offset_days",
        "scheduled_departure_utc",
        "prediction_timestamp_utc",
        "scheduled_duration_utc_minutes",
        "scheduled_duration_difference_minutes",
    ]

    for column_name in departure_columns:
        result[column_name] = departure_result[column_name]

    # Then reconcile arrivals using the newly created departure UTC.
    arrival_result = reconcile_scheduled_arrivals_vectorised(
        standardised_chunk=standardised_chunk,
        scheduled_departure_utc=result["scheduled_departure_utc"],
        maximum_duration_difference_minutes=maximum_duration_difference_minutes,
    )

    arrival_columns = [
        "scheduled_arrival_reconciliation_status",
        "scheduled_arrival_date_offset_days",
        "scheduled_arrival_utc",
        "scheduled_duration_utc_minutes",
        "scheduled_duration_difference_minutes",
    ]

    for column_name in arrival_columns:
        result[column_name] = arrival_result[column_name]

    return result
"""
Scheduled-arrival date reconciliation for airline flight records.

The raw source supplies:
- FL_DATE: flight date
- CRS_DEP_TIME: scheduled origin-local departure time in HHMM
- CRS_ARR_TIME: scheduled destination-local arrival time in HHMM
- CRS_ELAPSED_TIME: scheduled flight duration in minutes

Because scheduled arrival is stored as a local HHMM clock time, its calendar
date may be FL_DATE or a later destination-local date. This module evaluates
candidate destination-local arrival dates and selects the candidate whose UTC
duration is closest to the source scheduled elapsed time.

No local timestamps from different airports are directly subtracted.
All duration comparisons are performed in UTC.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from airline_disruption.timestamps.local_to_utc import (
    convert_local_hhmm_to_utc,
)


def reconcile_scheduled_arrival_timestamp(
    flight_date_local: Any,
    scheduled_departure_utc: pd.Timestamp | pd.NaT,
    scheduled_arrival_hhmm: Any,
    destination_iana_timezone: str,
    scheduled_elapsed_minutes: Any,
    maximum_arrival_date_offset_days: int = 2,
    maximum_duration_difference_minutes: float = 30.0,
) -> dict[str, Any]:
    """
    Select the most plausible scheduled arrival timestamp.

    Parameters
    ----------
    flight_date_local:
        Raw source local flight date.
    scheduled_departure_utc:
        Already reconstructed scheduled departure timestamp in UTC.
    scheduled_arrival_hhmm:
        Raw scheduled arrival time in HHMM form.
    destination_iana_timezone:
        IANA timezone of the destination airport.
    scheduled_elapsed_minutes:
        Source CRS_ELAPSED_TIME value in minutes.
    maximum_arrival_date_offset_days:
        Largest destination-local calendar-day offset tested after FL_DATE.
    maximum_duration_difference_minutes:
        Maximum acceptable absolute difference between UTC duration and the
        source scheduled elapsed time. Rows above this threshold are flagged
        as unreconciled rather than silently accepted.

    Returns
    -------
    dict[str, Any]
        Reconciled arrival timestamps, selected offset, duration comparison,
        and status fields.
    """
    if maximum_arrival_date_offset_days < 0:
        raise ValueError(
            "maximum_arrival_date_offset_days must be zero or greater."
        )

    if maximum_duration_difference_minutes < 0:
        raise ValueError(
            "maximum_duration_difference_minutes must be zero or greater."
        )

    if pd.isna(scheduled_departure_utc):
        return {
            "scheduled_arrival_local": pd.NaT,
            "scheduled_arrival_utc": pd.NaT,
            "scheduled_arrival_date_offset_days": pd.NA,
            "scheduled_duration_utc_minutes": pd.NA,
            "scheduled_duration_difference_minutes": pd.NA,
            "scheduled_arrival_reconciliation_status": (
                "missing_or_invalid_departure_utc"
            ),
        }

    parsed_flight_date = pd.to_datetime(
        flight_date_local,
        errors="coerce",
    )

    if pd.isna(parsed_flight_date):
        return {
            "scheduled_arrival_local": pd.NaT,
            "scheduled_arrival_utc": pd.NaT,
            "scheduled_arrival_date_offset_days": pd.NA,
            "scheduled_duration_utc_minutes": pd.NA,
            "scheduled_duration_difference_minutes": pd.NA,
            "scheduled_arrival_reconciliation_status": "invalid_flight_date",
        }

    numeric_scheduled_elapsed_minutes = pd.to_numeric(
        scheduled_elapsed_minutes,
        errors="coerce",
    )

    if pd.isna(numeric_scheduled_elapsed_minutes):
        return {
            "scheduled_arrival_local": pd.NaT,
            "scheduled_arrival_utc": pd.NaT,
            "scheduled_arrival_date_offset_days": pd.NA,
            "scheduled_duration_utc_minutes": pd.NA,
            "scheduled_duration_difference_minutes": pd.NA,
            "scheduled_arrival_reconciliation_status": (
                "missing_scheduled_elapsed_time"
            ),
        }

    candidate_arrivals = []

    for date_offset_days in range(
        maximum_arrival_date_offset_days + 1
    ):
        candidate_local_date = (
            parsed_flight_date
            + pd.Timedelta(days=date_offset_days)
        )

        candidate_result = convert_local_hhmm_to_utc(
            local_date=candidate_local_date,
            raw_hhmm=scheduled_arrival_hhmm,
            iana_timezone=destination_iana_timezone,
        )

        if candidate_result.input_status != "valid":
            continue

        candidate_duration_minutes = (
            candidate_result.utc_timestamp
            - scheduled_departure_utc
        ).total_seconds() / 60

        # A scheduled arrival before departure in UTC is physically impossible.
        if candidate_duration_minutes < 0:
            continue

        candidate_difference_minutes = abs(
            candidate_duration_minutes
            - float(numeric_scheduled_elapsed_minutes)
        )

        candidate_arrivals.append(
            {
                "local_timestamp": candidate_result.local_timestamp,
                "utc_timestamp": candidate_result.utc_timestamp,
                "date_offset_days": date_offset_days,
                "duration_utc_minutes": candidate_duration_minutes,
                "duration_difference_minutes": (
                    candidate_difference_minutes
                ),
            }
        )

    if not candidate_arrivals:
        return {
            "scheduled_arrival_local": pd.NaT,
            "scheduled_arrival_utc": pd.NaT,
            "scheduled_arrival_date_offset_days": pd.NA,
            "scheduled_duration_utc_minutes": pd.NA,
            "scheduled_duration_difference_minutes": pd.NA,
            "scheduled_arrival_reconciliation_status": (
                "no_valid_arrival_candidate"
            ),
        }

    best_candidate = min(
        candidate_arrivals,
        key=lambda candidate: candidate["duration_difference_minutes"],
    )

    reconciliation_status = "reconciled"

    if (
        best_candidate["duration_difference_minutes"]
        > maximum_duration_difference_minutes
    ):
        reconciliation_status = "duration_difference_exceeds_tolerance"

    return {
        "scheduled_arrival_local": best_candidate["local_timestamp"],
        "scheduled_arrival_utc": best_candidate["utc_timestamp"],
        "scheduled_arrival_date_offset_days": (
            best_candidate["date_offset_days"]
        ),
        "scheduled_duration_utc_minutes": (
            best_candidate["duration_utc_minutes"]
        ),
        "scheduled_duration_difference_minutes": (
            best_candidate["duration_difference_minutes"]
        ),
        "scheduled_arrival_reconciliation_status": reconciliation_status,
    }
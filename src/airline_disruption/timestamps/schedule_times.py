"""
Schedule timestamp reconstruction for airline flight records.

This module creates timezone-aware local and UTC timestamps from:
- a local flight date;
- scheduled departure and arrival HHMM clock times;
- origin and destination IANA timezones.

The module does not infer timezone mappings. Callers must provide valid
origin and destination IANA timezone values from the controlled project
airport-timezone mapping table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from airline_disruption.timestamps.local_to_utc import (
    convert_local_hhmm_to_utc,
)


def reconstruct_scheduled_timestamps(
    flight_date_local: Any,
    scheduled_departure_hhmm: Any,
    scheduled_arrival_hhmm: Any,
    origin_iana_timezone: str,
    destination_iana_timezone: str,
    prediction_lead_hours: int = 2,
) -> dict[str, Any]:
    """
    Reconstruct scheduled local and UTC timestamps for one flight.

    Parameters
    ----------
    flight_date_local:
        Source flight date associated with the scheduled local times.
    scheduled_departure_hhmm:
        Scheduled departure clock time in source HHMM format.
    scheduled_arrival_hhmm:
        Scheduled arrival clock time in source HHMM format.
    origin_iana_timezone:
        IANA timezone for the origin airport.
    destination_iana_timezone:
        IANA timezone for the destination airport.
    prediction_lead_hours:
        Number of hours before scheduled departure at which the pre-flight
        prediction is defined. The approved project value is 2.

    Returns
    -------
    dict[str, Any]
        Reconstructed timestamps and auditable conversion status fields.

    Notes
    -----
    The scheduled arrival source field is initially interpreted against
    FL_DATE in the destination timezone. A later reconciliation step must
    use CRS_ELAPSED_TIME and UTC ordering to identify cases where arrival
    belongs to the next local date. This prevents unsafe assumptions from
    local clock values alone.
    """
    if prediction_lead_hours < 0:
        raise ValueError(
            "prediction_lead_hours must be zero or greater."
        )

    scheduled_departure_result = convert_local_hhmm_to_utc(
        local_date=flight_date_local,
        raw_hhmm=scheduled_departure_hhmm,
        iana_timezone=origin_iana_timezone,
    )

    scheduled_arrival_result = convert_local_hhmm_to_utc(
        local_date=flight_date_local,
        raw_hhmm=scheduled_arrival_hhmm,
        iana_timezone=destination_iana_timezone,
    )

    prediction_timestamp_utc = pd.NaT

    if scheduled_departure_result.input_status == "valid":
        prediction_timestamp_utc = (
            scheduled_departure_result.utc_timestamp
            - pd.Timedelta(hours=prediction_lead_hours)
        )

    return {
        "scheduled_departure_local": (
            scheduled_departure_result.local_timestamp
        ),
        "scheduled_departure_utc": (
            scheduled_departure_result.utc_timestamp
        ),
        "scheduled_departure_date_rollover_days": (
            scheduled_departure_result.date_rollover_days
        ),
        "scheduled_departure_timestamp_status": (
            scheduled_departure_result.input_status
        ),
        "scheduled_arrival_local_initial": (
            scheduled_arrival_result.local_timestamp
        ),
        "scheduled_arrival_utc_initial": (
            scheduled_arrival_result.utc_timestamp
        ),
        "scheduled_arrival_date_rollover_days_initial": (
            scheduled_arrival_result.date_rollover_days
        ),
        "scheduled_arrival_timestamp_status_initial": (
            scheduled_arrival_result.input_status
        ),
        "prediction_timestamp_utc": prediction_timestamp_utc,
    }
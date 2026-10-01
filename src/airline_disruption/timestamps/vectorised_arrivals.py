"""
Vectorised scheduled-arrival timestamp reconciliation.

This module reconstructs scheduled arrival timestamps from destination-local
HHMM values and chooses the candidate destination-local calendar date whose
UTC duration is closest to CRS_ELAPSED_TIME.

The implementation evaluates FL_DATE plus 0, 1, and 2 destination-local days
in timezone groups. It does not subtract origin and destination local clocks.
All schedule-duration comparisons occur in UTC.

Rows with missing/invalid inputs, DST anomalies, or duration mismatches are
retained with explicit audit statuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airline_disruption.timestamps.local_to_utc import (
    convert_local_hhmm_to_utc,
)
from airline_disruption.timestamps.vectorised_departures import (
    prepare_hhmm_components,
)


REQUIRED_VECTORISED_ARRIVAL_COLUMNS = {
    "flight_date_local",
    "scheduled_arrival_time_local_hhmm",
    "scheduled_elapsed_minutes",
    "destination_iana_timezone",
}


def validate_vectorised_arrival_columns(
    standardised_chunk: pd.DataFrame,
) -> None:
    """
    Validate fields required for vectorised scheduled-arrival reconstruction.

    Parameters
    ----------
    standardised_chunk:
        Standardised flight-data chunk.

    Raises
    ------
    ValueError
        If required arrival-reconstruction columns are missing.
    """
    missing_columns = sorted(
        REQUIRED_VECTORISED_ARRIVAL_COLUMNS
        - set(standardised_chunk.columns)
    )

    if missing_columns:
        raise ValueError(
            "Standardised flight chunk is missing required vectorised "
            f"arrival columns: {missing_columns}"
        )


def reconcile_scheduled_arrivals_vectorised(
    standardised_chunk: pd.DataFrame,
    scheduled_departure_utc: pd.Series,
    maximum_arrival_date_offset_days: int = 2,
    maximum_duration_difference_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Reconstruct and reconcile scheduled arrival timestamps in timezone groups.

    Parameters
    ----------
    standardised_chunk:
        Standardised flight rows with source local date, scheduled arrival
        HHMM field, scheduled duration, and destination IANA timezone.
    scheduled_departure_utc:
        UTC scheduled-departure timestamps aligned to standardised_chunk.
    maximum_arrival_date_offset_days:
        Maximum destination-local day offset tested after FL_DATE.
    maximum_duration_difference_minutes:
        Maximum allowed absolute UTC-duration difference from source
        CRS_ELAPSED_TIME before a row is flagged as unreconciled.

    Returns
    -------
    pd.DataFrame
        Arrival local and UTC timestamps, chosen offset, duration comparison,
        and reconciliation status.
    """
    validate_vectorised_arrival_columns(standardised_chunk)

    if maximum_arrival_date_offset_days < 0:
        raise ValueError(
            "maximum_arrival_date_offset_days must be zero or greater."
        )

    if maximum_duration_difference_minutes < 0:
        raise ValueError(
            "maximum_duration_difference_minutes must be zero or greater."
        )

    if not scheduled_departure_utc.index.equals(
        standardised_chunk.index
    ):
        raise ValueError(
            "scheduled_departure_utc must have the same index as "
            "standardised_chunk."
        )

    result = pd.DataFrame(index=standardised_chunk.index)

    # Local timestamps can belong to different destination zones, therefore
    # this column must use object dtype. UTC timestamps share a single zone.
    result["scheduled_arrival_local"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="object",
    )

    result["scheduled_arrival_utc"] = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    result["scheduled_arrival_date_offset_days"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result["scheduled_duration_utc_minutes"] = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
    )

    result["scheduled_duration_difference_minutes"] = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
    )

    result["scheduled_arrival_reconciliation_status"] = "pending"

    arrival_components = prepare_hhmm_components(
        standardised_chunk[
            "scheduled_arrival_time_local_hhmm"
        ]
    )

    parsed_flight_dates = pd.to_datetime(
        standardised_chunk["flight_date_local"],
        errors="coerce",
    )

    destination_timezones = (
        standardised_chunk["destination_iana_timezone"]
        .astype("string")
        .str.strip()
    )

    numeric_scheduled_elapsed_minutes = pd.to_numeric(
        standardised_chunk["scheduled_elapsed_minutes"],
        errors="coerce",
    )

    missing_or_invalid_departure_mask = (
        scheduled_departure_utc.isna()
    )

    result.loc[
        missing_or_invalid_departure_mask,
        "scheduled_arrival_reconciliation_status",
    ] = "missing_or_invalid_departure_utc"

    invalid_arrival_time_mask = (
        ~missing_or_invalid_departure_mask
        & arrival_components["input_status"].ne("valid")
    )

    result.loc[
        invalid_arrival_time_mask,
        "scheduled_arrival_reconciliation_status",
    ] = arrival_components.loc[
        invalid_arrival_time_mask,
        "input_status",
    ].map(
        {
            "missing_time": "missing_arrival_time",
            "invalid_time": "invalid_arrival_time",
        }
    )

    invalid_flight_date_mask = (
        result["scheduled_arrival_reconciliation_status"].eq("pending")
        & parsed_flight_dates.isna()
    )

    result.loc[
        invalid_flight_date_mask,
        "scheduled_arrival_reconciliation_status",
    ] = "invalid_flight_date"

    missing_timezone_mask = (
        result["scheduled_arrival_reconciliation_status"].eq("pending")
        & (
            destination_timezones.isna()
            | destination_timezones.eq("")
        )
    )

    result.loc[
        missing_timezone_mask,
        "scheduled_arrival_reconciliation_status",
    ] = "invalid_destination_timezone"

    missing_elapsed_time_mask = (
        result["scheduled_arrival_reconciliation_status"].eq("pending")
        & numeric_scheduled_elapsed_minutes.isna()
    )

    result.loc[
        missing_elapsed_time_mask,
        "scheduled_arrival_reconciliation_status",
    ] = "missing_scheduled_elapsed_time"

    eligible_mask = result[
        "scheduled_arrival_reconciliation_status"
    ].eq("pending")

    if not eligible_mask.any():
        return result

    candidate_records: list[pd.DataFrame] = []

    # First classify DST anomalies on the base source flight date.
    #
    # A raw destination-local scheduled-arrival time that is nonexistent
    # during spring DST change or ambiguous during autumn DST change must not
    # be "rescued" by trying later calendar dates. The later dates represent
    # different local timestamps, not valid interpretations of the raw input.
    #
    # We use vectorised localisation first, then the already-tested scalar
    # reference converter only for rare NaT DST cases.
    base_date_dst_anomaly_indices: list[object] = []

    valid_timezones_for_base_check = destination_timezones.loc[
        eligible_mask
    ].dropna().unique()

    for timezone_name in valid_timezones_for_base_check:
        timezone_base_mask = (
            eligible_mask
            & destination_timezones.eq(timezone_name)
        )
        timezone_base_indices = standardised_chunk.index[
            timezone_base_mask
        ]

        base_naive_timestamps = (
            parsed_flight_dates.loc[timezone_base_indices]
            + pd.to_timedelta(
                arrival_components.loc[
                    timezone_base_indices,
                    "hour",
                ],
                unit="h",
            )
            + pd.to_timedelta(
                arrival_components.loc[
                    timezone_base_indices,
                    "minute",
                ],
                unit="m",
            )
            + pd.to_timedelta(
                arrival_components.loc[
                    timezone_base_indices,
                    "date_rollover_days",
                ],
                unit="D",
            )
        )

        try:
            base_localised_timestamps = (
                base_naive_timestamps.dt.tz_localize(
                    timezone_name,
                    ambiguous="NaT",
                    nonexistent="NaT",
                )
            )
        except Exception:
            continue

        base_dst_anomaly_indices = base_localised_timestamps.index[
            base_localised_timestamps.isna()
        ]

        for row_index in base_dst_anomaly_indices:
            reference_result = convert_local_hhmm_to_utc(
                local_date=standardised_chunk.loc[
                    row_index,
                    "flight_date_local",
                ],
                raw_hhmm=standardised_chunk.loc[
                    row_index,
                    "scheduled_arrival_time_local_hhmm",
                ],
                iana_timezone=timezone_name,
            )

            if reference_result.input_status == "ambiguous_local_time":
                result.loc[
                    row_index,
                    "scheduled_arrival_reconciliation_status",
                ] = "ambiguous_arrival_local_time"
                base_date_dst_anomaly_indices.append(row_index)

            elif reference_result.input_status == "nonexistent_local_time":
                result.loc[
                    row_index,
                    "scheduled_arrival_reconciliation_status",
                ] = "nonexistent_arrival_local_time"
                base_date_dst_anomaly_indices.append(row_index)

    # Exclude identified base-date DST anomalies from all later-date candidates.
    eligible_mask = result[
        "scheduled_arrival_reconciliation_status"
    ].eq("pending")

    if not eligible_mask.any():
        return result

    valid_timezones = destination_timezones.loc[
        eligible_mask
    ].dropna().unique()

    for timezone_name in valid_timezones:
        timezone_mask = (
            eligible_mask
            & destination_timezones.eq(timezone_name)
        )

        timezone_indices = standardised_chunk.index[
            timezone_mask
        ]

        for date_offset_days in range(
            maximum_arrival_date_offset_days + 1
        ):
            candidate_naive_timestamps = (
                parsed_flight_dates.loc[timezone_indices]
                + pd.to_timedelta(
                    arrival_components.loc[
                        timezone_indices,
                        "hour",
                    ],
                    unit="h",
                )
                + pd.to_timedelta(
                    arrival_components.loc[
                        timezone_indices,
                        "minute",
                    ],
                    unit="m",
                )
                + pd.to_timedelta(
                    arrival_components.loc[
                        timezone_indices,
                        "date_rollover_days",
                    ] + date_offset_days,
                    unit="D",
                )
            )

            try:
                candidate_local_timestamps = (
                    candidate_naive_timestamps.dt.tz_localize(
                        timezone_name,
                        ambiguous="NaT",
                        nonexistent="NaT",
                    )
                )
            except Exception:
                result.loc[
                    timezone_indices,
                    "scheduled_arrival_reconciliation_status",
                ] = "invalid_destination_timezone"
                continue

            valid_candidate_mask = (
                candidate_local_timestamps.notna()
            )

            if not valid_candidate_mask.any():
                continue

            valid_candidate_indices = (
                candidate_local_timestamps.index[
                    valid_candidate_mask
                ]
            )

            candidate_utc_timestamps = (
                candidate_local_timestamps.loc[
                    valid_candidate_indices
                ].dt.tz_convert("UTC")
            )

            candidate_duration_minutes = (
                (
                    candidate_utc_timestamps
                    - scheduled_departure_utc.loc[
                        valid_candidate_indices
                    ]
                )
                .dt.total_seconds()
                / 60
            )

            physically_possible_mask = (
                candidate_duration_minutes.ge(0)
            )

            if not physically_possible_mask.any():
                continue

            physically_possible_indices = (
                candidate_duration_minutes.index[
                    physically_possible_mask
                ]
            )

            duration_difference_minutes = (
                candidate_duration_minutes.loc[
                    physically_possible_indices
                ]
                - numeric_scheduled_elapsed_minutes.loc[
                    physically_possible_indices
                ]
            ).abs()

            candidate_records.append(
                pd.DataFrame(
                    {
                        "row_index": physically_possible_indices,
                        "candidate_local": (
                            candidate_local_timestamps.loc[
                                physically_possible_indices
                            ].to_numpy()
                        ),
                        "candidate_utc": (
                            candidate_utc_timestamps.loc[
                                physically_possible_indices
                            ].to_numpy()
                        ),
                        "candidate_date_offset_days": (
                            arrival_components.loc[
                                physically_possible_indices,
                                "date_rollover_days",
                            ].to_numpy()
                            + date_offset_days
                        ),
                        "candidate_duration_utc_minutes": (
                            candidate_duration_minutes.loc[
                                physically_possible_indices
                            ].to_numpy()
                        ),
                        "candidate_duration_difference_minutes": (
                            duration_difference_minutes.to_numpy()
                        ),
                    }
                )
            )

    if not candidate_records:
        unresolved_eligible_mask = result[
            "scheduled_arrival_reconciliation_status"
        ].eq("pending")

        result.loc[
            unresolved_eligible_mask,
            "scheduled_arrival_reconciliation_status",
        ] = "no_valid_arrival_candidate"

        return result

    candidates = pd.concat(
        candidate_records,
        ignore_index=True,
    )

    # For each original row, choose the candidate with the smallest
    # difference from CRS_ELAPSED_TIME. Sorting by offset makes ties
    # deterministic in favour of the earlier local arrival date.
    candidates = candidates.sort_values(
        by=[
            "row_index",
            "candidate_duration_difference_minutes",
            "candidate_date_offset_days",
        ],
        ascending=True,
    )

    best_candidates = candidates.drop_duplicates(
        subset="row_index",
        keep="first",
    ).set_index("row_index")

    best_indices = best_candidates.index

    result.loc[
        best_indices,
        "scheduled_arrival_local",
    ] = best_candidates["candidate_local"].to_numpy()

    result.loc[
        best_indices,
        "scheduled_arrival_utc",
    ] = best_candidates["candidate_utc"].to_numpy()

    result.loc[
        best_indices,
        "scheduled_arrival_date_offset_days",
    ] = best_candidates[
        "candidate_date_offset_days"
    ].astype("Int64")

    result.loc[
        best_indices,
        "scheduled_duration_utc_minutes",
    ] = best_candidates[
        "candidate_duration_utc_minutes"
    ].astype(float)

    result.loc[
        best_indices,
        "scheduled_duration_difference_minutes",
    ] = best_candidates[
        "candidate_duration_difference_minutes"
    ].astype(float)

    within_tolerance_mask = (
        best_candidates[
            "candidate_duration_difference_minutes"
        ].le(maximum_duration_difference_minutes)
    )

    result.loc[
        best_indices[within_tolerance_mask],
        "scheduled_arrival_reconciliation_status",
    ] = "reconciled"

    result.loc[
        best_indices[~within_tolerance_mask],
        "scheduled_arrival_reconciliation_status",
    ] = "duration_difference_exceeds_tolerance"

    # Rows still pending had no candidate despite at least one candidate
    # record existing for another row.
    still_pending_mask = result[
        "scheduled_arrival_reconciliation_status"
    ].eq("pending")

    result.loc[
        still_pending_mask,
        "scheduled_arrival_reconciliation_status",
    ] = "no_valid_arrival_candidate"

    return result
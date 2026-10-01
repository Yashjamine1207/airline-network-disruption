"""
Compare row-wise and vectorised scheduled-timestamp reconciliation.

This script validates that the combined vectorised schedule implementation
matches the established row-wise reference on real flight records. It does
not write processed data.

Compared fields:
- scheduled_departure_timestamp_status;
- scheduled_departure_reconciliation_status;
- scheduled_departure_date_offset_days;
- scheduled_departure_utc;
- prediction_timestamp_utc;
- scheduled_arrival_reconciliation_status;
- scheduled_arrival_date_offset_days;
- scheduled_arrival_utc;
- scheduled_duration_utc_minutes;
- scheduled_duration_difference_minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from airline_disruption.data.ingestion import (  # noqa: E402
    load_airport_timezone_mapping,
    standardise_flight_chunk,
)
from airline_disruption.timestamps.normalise_schedule import (  # noqa: E402
    normalise_scheduled_timestamps,
)
from airline_disruption.timestamps.vectorised_schedule import (  # noqa: E402
    reconcile_scheduled_timestamps_vectorised,
)


COMPARISON_ROW_COUNT = 10_000

RAW_FLIGHTS_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "flights_kaggle"
    / "flights_sample_3m.csv"
)

AIRPORT_TIMEZONE_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "airport_timezone_mapping.csv"
)


def values_match(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Compare two series while treating paired missing values as equal.
    """
    return (left == right) | (left.isna() & right.isna())


def main() -> None:
    """Compare reference and vectorised schedule reconciliation results."""
    airport_timezone_mapping = load_airport_timezone_mapping(
        AIRPORT_TIMEZONE_MAPPING_PATH
    )

    raw_sample = pd.read_csv(
        RAW_FLIGHTS_CSV_PATH,
        nrows=COMPARISON_ROW_COUNT,
        low_memory=False,
    )

    standardised_sample = standardise_flight_chunk(
        raw_chunk=raw_sample,
        airport_timezone_mapping=airport_timezone_mapping,
        source_row_start=0,
    )

    print("Running row-wise reference schedule normalisation...")

    rowwise_result = normalise_scheduled_timestamps(
        standardised_sample
    )

    print("Running vectorised scheduled-timestamp reconciliation...")

    vectorised_result = reconcile_scheduled_timestamps_vectorised(
        standardised_chunk=standardised_sample,
    )

    departure_status_match = values_match(
        rowwise_result[
            "scheduled_departure_timestamp_status"
        ],
        vectorised_result[
            "scheduled_departure_timestamp_status"
        ],
    )

    departure_recon_match = values_match(
        rowwise_result[
            "scheduled_departure_reconciliation_status"
        ],
        vectorised_result[
            "scheduled_departure_reconciliation_status"
        ],
    )

    departure_offset_match = values_match(
        rowwise_result[
            "scheduled_departure_date_offset_days"
        ],
        vectorised_result[
            "scheduled_departure_date_offset_days"
        ],
    )

    departure_utc_match = values_match(
        rowwise_result["scheduled_departure_utc"],
        vectorised_result["scheduled_departure_utc"],
    )

    prediction_utc_match = values_match(
        rowwise_result["prediction_timestamp_utc"],
        vectorised_result["prediction_timestamp_utc"],
    )

    arrival_status_match = values_match(
        rowwise_result[
            "scheduled_arrival_reconciliation_status"
        ],
        vectorised_result[
            "scheduled_arrival_reconciliation_status"
        ],
    )

    arrival_offset_match = values_match(
        rowwise_result[
            "scheduled_arrival_date_offset_days"
        ],
        vectorised_result[
            "scheduled_arrival_date_offset_days"
        ],
    )

    arrival_utc_match = values_match(
        rowwise_result["scheduled_arrival_utc"],
        vectorised_result["scheduled_arrival_utc"],
    )

    duration_match = values_match(
        rowwise_result["scheduled_duration_utc_minutes"],
        vectorised_result["scheduled_duration_utc_minutes"],
    )

    difference_match = values_match(
        rowwise_result[
            "scheduled_duration_difference_minutes"
        ],
        vectorised_result[
            "scheduled_duration_difference_minutes"
        ],
    )

    all_fields_match = (
        departure_status_match
        & departure_recon_match
        & departure_offset_match
        & departure_utc_match
        & prediction_utc_match
        & arrival_status_match
        & arrival_offset_match
        & arrival_utc_match
        & duration_match
        & difference_match
    )

    mismatch_columns = [
        "source_row_number",
        "flight_date_local",
        "origin_airport",
        "destination_airport",
        "origin_iana_timezone",
        "destination_iana_timezone",
        "scheduled_departure_time_local_hhmm",
        "scheduled_arrival_time_local_hhmm",
        "scheduled_elapsed_minutes",
    ]

    mismatches = standardised_sample.loc[
        ~all_fields_match,
        mismatch_columns,
    ].copy()

    mismatches["rowwise_departure_status"] = rowwise_result.loc[
        ~all_fields_match,
        "scheduled_departure_timestamp_status",
    ]

    mismatches["vectorised_departure_status"] = vectorised_result.loc[
        ~all_fields_match,
        "scheduled_departure_timestamp_status",
    ]

    mismatches["rowwise_arrival_status"] = rowwise_result.loc[
        ~all_fields_match,
        "scheduled_arrival_reconciliation_status",
    ]

    mismatches["vectorised_arrival_status"] = vectorised_result.loc[
        ~all_fields_match,
        "scheduled_arrival_reconciliation_status",
    ]

    print("\nVectorised schedule comparison")
    print("-" * 60)
    print(f"Rows compared: {len(standardised_sample):,}")
    print(f"Departure status matches: {departure_status_match.sum():,}")
    print(f"Departure reconciliation matches: {departure_recon_match.sum():,}")
    print(f"Departure offset matches: {departure_offset_match.sum():,}")
    print(f"Departure UTC matches: {departure_utc_match.sum():,}")
    print(f"Prediction UTC matches: {prediction_utc_match.sum():,}")
    print(f"Arrival status matches: {arrival_status_match.sum():,}")
    print(f"Arrival offset matches: {arrival_offset_match.sum():,}")
    print(f"Arrival UTC matches: {arrival_utc_match.sum():,}")
    print(f"Duration matches: {duration_match.sum():,}")
    print(f"Duration-difference matches: {difference_match.sum():,}")
    print(f"All compared fields match: {all_fields_match.sum():,}")
    print(f"Rows with one or more mismatches: {(~all_fields_match).sum():,}")

    if not mismatches.empty:
        print("\nFirst 20 mismatches:")
        print(mismatches.head(20).to_string(index=False))

        raise AssertionError(
            "Vectorised schedule does not yet match the row-wise reference."
        )

    print(
        "\nSuccess: vectorised scheduled-timestamp reconciliation matches "
        "the row-wise reference for every compared row."
    )


if __name__ == "__main__":
    main()
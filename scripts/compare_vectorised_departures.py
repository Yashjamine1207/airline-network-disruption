"""
Compare row-wise and vectorised scheduled-departure reconstruction.

This script checks whether the optimised vectorised departure implementation
matches the established row-wise reference implementation on a real raw-flight
sample. It does not write processed data.

Comparison fields:
- scheduled_departure_timestamp_status;
- scheduled_departure_date_rollover_days;
- scheduled_departure_utc;
- prediction_timestamp_utc.
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
from airline_disruption.timestamps.vectorised_departures import (  # noqa: E402
    reconstruct_scheduled_departures_vectorised,
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


def timestamp_values_match(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Compare timestamp series while treating paired missing values as equal.
    """
    return (left == right) | (left.isna() & right.isna())


def main() -> None:
    """Compare reference and vectorised departure results on real source rows."""
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

    print("Running vectorised scheduled-departure reconstruction...")
    vectorised_departure_result = (
        reconstruct_scheduled_departures_vectorised(
            standardised_sample
        )
    )

    status_match = (
        rowwise_result["scheduled_departure_timestamp_status"]
        == vectorised_departure_result[
            "scheduled_departure_timestamp_status"
        ]
    )

    rollover_match = (
        rowwise_result["scheduled_departure_date_rollover_days"]
        == vectorised_departure_result[
            "scheduled_departure_date_rollover_days"
        ]
    )

    departure_utc_match = timestamp_values_match(
        rowwise_result["scheduled_departure_utc"],
        vectorised_departure_result[
            "scheduled_departure_utc"
        ],
    )

    prediction_utc_match = timestamp_values_match(
        rowwise_result["prediction_timestamp_utc"],
        vectorised_departure_result[
            "prediction_timestamp_utc"
        ],
    )

    all_fields_match = (
        status_match
        & rollover_match
        & departure_utc_match
        & prediction_utc_match
    )

    mismatch_columns = [
        "source_row_number",
        "flight_date_local",
        "origin_airport",
        "origin_iana_timezone",
        "scheduled_departure_time_local_hhmm",
    ]

    mismatches = standardised_sample.loc[
        ~all_fields_match,
        mismatch_columns,
    ].copy()

    mismatches["rowwise_status"] = rowwise_result.loc[
        ~all_fields_match,
        "scheduled_departure_timestamp_status",
    ]

    mismatches["vectorised_status"] = vectorised_departure_result.loc[
        ~all_fields_match,
        "scheduled_departure_timestamp_status",
    ]

    mismatches["rowwise_departure_utc"] = rowwise_result.loc[
        ~all_fields_match,
        "scheduled_departure_utc",
    ]

    mismatches["vectorised_departure_utc"] = (
        vectorised_departure_result.loc[
            ~all_fields_match,
            "scheduled_departure_utc",
        ]
    )

    print("\nVectorised departure comparison")
    print("-" * 60)
    print(f"Rows compared: {len(standardised_sample):,}")
    print(f"Status matches: {status_match.sum():,}")
    print(f"Rollover matches: {rollover_match.sum():,}")
    print(f"Departure UTC matches: {departure_utc_match.sum():,}")
    print(f"Prediction UTC matches: {prediction_utc_match.sum():,}")
    print(f"All compared fields match: {all_fields_match.sum():,}")
    print(f"Rows with one or more mismatches: {(~all_fields_match).sum():,}")

    if not mismatches.empty:
        print("\nFirst 20 mismatches:")
        print(mismatches.head(20).to_string(index=False))

        raise AssertionError(
            "Vectorised departures do not yet match the row-wise reference."
        )

    print(
        "\nSuccess: vectorised scheduled-departure reconstruction matches "
        "the row-wise reference for every compared row."
    )


if __name__ == "__main__":
    main()
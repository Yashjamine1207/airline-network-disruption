"""
Benchmark vectorised scheduled-departure timestamp reconstruction.

This script measures the performance of the validated vectorised departure
implementation on a controlled real-data sample. It does not write outputs.

The benchmark is compared later with the earlier row-wise baseline:
- Row-wise full schedule normalisation: about 2,213 rows per second
- Vectorised departure reconstruction: measured by this script
"""

from __future__ import annotations

import sys
import time
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
from airline_disruption.timestamps.vectorised_departures import (  # noqa: E402
    reconstruct_scheduled_departures_vectorised,
)


BENCHMARK_ROW_COUNT = 100_000

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


def main() -> None:
    """Benchmark vectorised departure reconstruction on real source rows."""
    airport_timezone_mapping = load_airport_timezone_mapping(
        AIRPORT_TIMEZONE_MAPPING_PATH
    )

    print(f"Reading {BENCHMARK_ROW_COUNT:,} raw flight rows...")

    raw_sample = pd.read_csv(
        RAW_FLIGHTS_CSV_PATH,
        nrows=BENCHMARK_ROW_COUNT,
        low_memory=False,
    )

    print("Standardising source fields and joining airport timezones...")

    standardised_sample = standardise_flight_chunk(
        raw_chunk=raw_sample,
        airport_timezone_mapping=airport_timezone_mapping,
        source_row_start=0,
    )

    print("Running vectorised scheduled-departure reconstruction...")

    benchmark_start_time = time.perf_counter()

    vectorised_result = reconstruct_scheduled_departures_vectorised(
        standardised_sample
    )

    elapsed_seconds = time.perf_counter() - benchmark_start_time

    valid_departure_count = int(
        vectorised_result[
            "scheduled_departure_timestamp_status"
        ].eq("valid").sum()
    )

    prediction_timestamp_count = int(
        vectorised_result[
            "prediction_timestamp_utc"
        ].notna().sum()
    )

    rollover_count = int(
        vectorised_result[
            "scheduled_departure_date_rollover_days"
        ].eq(1).sum()
    )

    print("\nVectorised departure benchmark")
    print("-" * 60)
    print(f"Benchmark rows: {len(vectorised_result):,}")
    print(f"Vectorised execution time: {elapsed_seconds:.2f} seconds")
    print(
        "Approximate vectorised speed: "
        f"{len(vectorised_result) / elapsed_seconds:,.0f} rows per second"
    )
    print(f"Valid scheduled departures: {valid_departure_count:,}")
    print(
        "Available prediction timestamps: "
        f"{prediction_timestamp_count:,}"
    )
    print(f"Scheduled departures with 2400 rollover: {rollover_count:,}")


if __name__ == "__main__":
    main()
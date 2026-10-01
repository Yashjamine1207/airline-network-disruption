"""
Benchmark the schedule timestamp-normalisation pipeline on a controlled sample.

This script does not write processed flight data. It measures:
- raw CSV read time;
- standardisation and airport-timezone join time;
- schedule timestamp normalisation time;
- total elapsed time;
- timestamp-status and reconciliation-status outcomes.

Use the benchmark before deciding the safe chunk size for the complete
3-million-row Phase 1 normalisation run.
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
from airline_disruption.timestamps.normalise_schedule import (  # noqa: E402
    normalise_scheduled_timestamps,
)


BENCHMARK_ROW_COUNT = 10_000
PREDICTION_LEAD_HOURS = 2
MAXIMUM_ARRIVAL_DATE_OFFSET_DAYS = 2
MAXIMUM_DURATION_DIFFERENCE_MINUTES = 30.0

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
    """Run and report the controlled schedule-normalisation benchmark."""
    if not RAW_FLIGHTS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Raw flight CSV was not found: {RAW_FLIGHTS_CSV_PATH}"
        )

    if not AIRPORT_TIMEZONE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Airport timezone mapping was not found: "
            f"{AIRPORT_TIMEZONE_MAPPING_PATH}"
        )

    pipeline_start_time = time.perf_counter()

    airport_timezone_mapping = load_airport_timezone_mapping(
        AIRPORT_TIMEZONE_MAPPING_PATH
    )

    raw_read_start_time = time.perf_counter()

    raw_benchmark_sample = pd.read_csv(
        RAW_FLIGHTS_CSV_PATH,
        nrows=BENCHMARK_ROW_COUNT,
        low_memory=False,
    )

    raw_read_seconds = time.perf_counter() - raw_read_start_time

    standardisation_start_time = time.perf_counter()

    standardised_benchmark_sample = standardise_flight_chunk(
        raw_chunk=raw_benchmark_sample,
        airport_timezone_mapping=airport_timezone_mapping,
        source_row_start=0,
    )

    standardisation_seconds = (
        time.perf_counter() - standardisation_start_time
    )

    normalisation_start_time = time.perf_counter()

    normalised_benchmark_sample = normalise_scheduled_timestamps(
        standardised_chunk=standardised_benchmark_sample,
        prediction_lead_hours=PREDICTION_LEAD_HOURS,
        maximum_arrival_date_offset_days=(
            MAXIMUM_ARRIVAL_DATE_OFFSET_DAYS
        ),
        maximum_duration_difference_minutes=(
            MAXIMUM_DURATION_DIFFERENCE_MINUTES
        ),
    )

    normalisation_seconds = (
        time.perf_counter() - normalisation_start_time
    )

    total_pipeline_seconds = time.perf_counter() - pipeline_start_time

    valid_departure_count = int(
        normalised_benchmark_sample[
            "scheduled_departure_timestamp_status"
        ].eq("valid").sum()
    )

    reconciled_arrival_count = int(
        normalised_benchmark_sample[
            "scheduled_arrival_reconciliation_status"
        ].eq("reconciled").sum()
    )

    prediction_timestamp_count = int(
        normalised_benchmark_sample[
            "prediction_timestamp_utc"
        ].notna().sum()
    )

    print("Schedule normalisation benchmark")
    print("-" * 60)
    print(f"Benchmark rows: {len(normalised_benchmark_sample):,}")
    print(f"Raw CSV read time: {raw_read_seconds:.2f} seconds")
    print(
        "Standardisation and timezone join time: "
        f"{standardisation_seconds:.2f} seconds"
    )
    print(
        "Schedule timestamp normalisation time: "
        f"{normalisation_seconds:.2f} seconds"
    )
    print(
        f"Total pipeline time: {total_pipeline_seconds:.2f} seconds"
    )
    print(
        "Approximate normalisation speed: "
        f"{len(normalised_benchmark_sample) / normalisation_seconds:,.0f} "
        "rows per second"
    )
    print()
    print(f"Valid scheduled departures: {valid_departure_count:,}")
    print(f"Reconciled scheduled arrivals: {reconciled_arrival_count:,}")
    print(
        "Available prediction timestamps: "
        f"{prediction_timestamp_count:,}"
    )


if __name__ == "__main__":
    main()
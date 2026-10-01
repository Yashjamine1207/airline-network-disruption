"""
Benchmark vectorised vs row-wise schedule reconciliation.

This script measures the performance improvement from the vectorised
implementation compared to the row-wise reference on realistic data volumes.
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
from airline_disruption.timestamps.vectorised_schedule import (  # noqa: E402
    reconcile_scheduled_timestamps_vectorised,
)


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

BENCHMARK_ROW_COUNTS = [10_000, 50_000, 100_000]


def main() -> None:
    """Benchmark vectorised vs row-wise schedule reconciliation."""
    airport_timezone_mapping = load_airport_timezone_mapping(
        AIRPORT_TIMEZONE_MAPPING_PATH
    )

    raw_full = pd.read_csv(
        RAW_FLIGHTS_CSV_PATH,
        low_memory=False,
    )

    print("Vectorised schedule benchmark")
    print("=" * 70)
    print(f"Total available rows: {len(raw_full):,}")
    print()

    results: list[dict[str, object]] = []

    for row_count in BENCHMARK_ROW_COUNTS:
        if row_count > len(raw_full):
            print(f"Skipping {row_count:,}: exceeds available data.")
            continue

        print(f"Benchmarking {row_count:,} rows...")

        raw_sample = raw_full.iloc[:row_count].copy()

        standardised_sample = standardise_flight_chunk(
            raw_chunk=raw_sample,
            airport_timezone_mapping=airport_timezone_mapping,
            source_row_start=0,
        )

        # Row-wise timing
        start = time.perf_counter()
        _ = normalise_scheduled_timestamps(standardised_sample)
        rowwise_elapsed = time.perf_counter() - start

        # Vectorised timing
        start = time.perf_counter()
        _ = reconcile_scheduled_timestamps_vectorised(standardised_sample)
        vectorised_elapsed = time.perf_counter() - start

        speedup = rowwise_elapsed / vectorised_elapsed if vectorised_elapsed > 0 else float("inf")

        results.append(
            {
                "rows": row_count,
                "rowwise_seconds": round(rowwise_elapsed, 3),
                "vectorised_seconds": round(vectorised_elapsed, 3),
                "speedup": round(speedup, 2),
            }
        )

        print(f"  Row-wise:     {rowwise_elapsed:7.3f}s")
        print(f"  Vectorised:   {vectorised_elapsed:7.3f}s")
        print(f"  Speedup:      {speedup:7.2f}x")
        print()

    print("Summary")
    print("-" * 70)

    for result in results:
        print(
            f"{result['rows']:>10,} rows: "
            f"rowwise={result['rowwise_seconds']:>7.3f}s, "
            f"vectorised={result['vectorised_seconds']:>7.3f}s, "
            f"speedup={result['speedup']:>6.2f}x"
        )

    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print()
        print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
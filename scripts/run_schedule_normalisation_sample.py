"""
Run a small end-to-end schedule timestamp normalisation sample.

This script:
1. Reads the first 1,000 immutable raw flight records.
2. Applies raw-to-canonical standardisation.
3. Joins the controlled airport-to-IANA-timezone mapping.
4. Reconstructs scheduled departure timestamps.
5. Reconciles scheduled arrival timestamps in UTC.
6. Creates prediction_timestamp_utc at two hours before scheduled departure.
7. Writes a small local-only Parquet preview and an audit summary.

It does not modify raw data and does not process the full 3-million-row source.
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


SAMPLE_ROW_COUNT = 1_000
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

SAMPLE_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "sample"
)

SAMPLE_PARQUET_PATH = (
    SAMPLE_OUTPUT_DIRECTORY
    / "scheduled_timestamp_normalisation_sample.parquet"
)

SAMPLE_AUDIT_PATH = (
    SAMPLE_OUTPUT_DIRECTORY
    / "scheduled_timestamp_normalisation_sample_audit.csv"
)


def main() -> None:
    """Execute the end-to-end sample schedule-normalisation pipeline."""
    if not RAW_FLIGHTS_CSV_PATH.exists():
        raise FileNotFoundError(
            "Raw flight CSV was not found: "
            f"{RAW_FLIGHTS_CSV_PATH}"
        )

    if not AIRPORT_TIMEZONE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "Airport timezone mapping was not found: "
            f"{AIRPORT_TIMEZONE_MAPPING_PATH}"
        )

    SAMPLE_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading controlled airport-timezone mapping...")
    airport_timezone_mapping = load_airport_timezone_mapping(
        AIRPORT_TIMEZONE_MAPPING_PATH
    )

    print(f"Reading first {SAMPLE_ROW_COUNT:,} raw flight rows...")
    raw_sample = pd.read_csv(
        RAW_FLIGHTS_CSV_PATH,
        nrows=SAMPLE_ROW_COUNT,
        low_memory=False,
    )

    print("Standardising source fields and joining airport timezones...")
    standardised_sample = standardise_flight_chunk(
        raw_chunk=raw_sample,
        airport_timezone_mapping=airport_timezone_mapping,
        source_row_start=0,
    )

    print("Reconstructing schedule timestamps and prediction timestamps...")
    normalised_sample = normalise_scheduled_timestamps(
        standardised_chunk=standardised_sample,
        prediction_lead_hours=PREDICTION_LEAD_HOURS,
        maximum_arrival_date_offset_days=(
            MAXIMUM_ARRIVAL_DATE_OFFSET_DAYS
        ),
        maximum_duration_difference_minutes=(
            MAXIMUM_DURATION_DIFFERENCE_MINUTES
        ),
    )

    print("Writing local sample Parquet output...")
    normalised_sample.to_parquet(
        SAMPLE_PARQUET_PATH,
        index=False,
        engine="pyarrow",
    )

    audit_summary = pd.DataFrame(
        [
            {
                "sample_row_count": len(normalised_sample),
                "origin_timezone_found_count": int(
                    normalised_sample["origin_timezone_found"].sum()
                ),
                "destination_timezone_found_count": int(
                    normalised_sample["destination_timezone_found"].sum()
                ),
                "valid_scheduled_departure_count": int(
                    normalised_sample[
                        "scheduled_departure_timestamp_status"
                    ].eq("valid").sum()
                ),
                "reconciled_scheduled_arrival_count": int(
                    normalised_sample[
                        "scheduled_arrival_reconciliation_status"
                    ].eq("reconciled").sum()
                ),
                "arrival_duration_tolerance_exceeded_count": int(
                    normalised_sample[
                        "scheduled_arrival_reconciliation_status"
                    ].eq(
                        "duration_difference_exceeds_tolerance"
                    ).sum()
                ),
                "missing_or_invalid_departure_count": int(
                    normalised_sample[
                        "scheduled_arrival_reconciliation_status"
                    ].eq(
                        "missing_or_invalid_departure_utc"
                    ).sum()
                ),
                "prediction_timestamp_available_count": int(
                    normalised_sample[
                        "prediction_timestamp_utc"
                    ].notna().sum()
                ),
            }
        ]
    )

    audit_summary.to_csv(
        SAMPLE_AUDIT_PATH,
        index=False,
    )

    print("\nSample normalisation completed successfully.")
    print(f"Parquet sample: {SAMPLE_PARQUET_PATH}")
    print(f"Audit summary: {SAMPLE_AUDIT_PATH}")
    print("\nAudit summary:")
    print(audit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
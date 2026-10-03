"""
Phase 3 — Step 9
Build leakage-safe base schedule, calendar, and entity features.

Included feature groups:
- Schedule features.
- Calendar features.
- Entity features.
- Route features.

Excluded feature groups:
- Historical features.
- Rotation features.
- Weather features.
- Network features.

The raw CSV and target table are never modified.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FLIGHT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "flights_kaggle"
    / "flights_sample_3m.csv"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
    / "flight_targets.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "base"
)

OUTPUT_FILE = OUTPUT_DIRECTORY / "base_features.csv"

CHUNK_SIZE = 250_000

REQUIRED_SOURCE_COLUMNS = {
    "FL_DATE",
    "AIRLINE_CODE",
    "DOT_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
}

REQUIRED_TARGET_COLUMNS = {
    "source_row_number",
    "cancelled_target",
    "arrival_delay_minutes",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
    "completed_flight",
}


def validate_files() -> None:
    """Confirm that required input files exist."""

    if not RAW_FLIGHT_FILE.exists():
        raise FileNotFoundError(
            f"Raw flight file was not found:\n{RAW_FLIGHT_FILE}"
        )

    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"Target file was not found:\n{TARGET_FILE}"
        )


def parse_hhmm(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Parse integer-like HHMM values.

    Returns:
        hour:
            Parsed hour in the range 0 to 23.
        minute:
            Parsed minute in the range 0 to 59.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    valid_numeric = numeric.notna() & np.isfinite(numeric)
    integer_like = numeric.eq(np.floor(numeric))

    numeric = numeric.where(valid_numeric & integer_like)

    hour = np.floor(numeric / 100)
    minute = numeric % 100

    valid_time = (
        hour.between(0, 23)
        & minute.between(0, 59)
    )

    hour = hour.where(valid_time).astype("Int8")
    minute = minute.where(valid_time).astype("Int8")

    return hour, minute


def build_features(
    source_chunk: pd.DataFrame,
    target_chunk: pd.DataFrame,
) -> pd.DataFrame:
    """Build base features for one aligned source and target chunk."""

    source = source_chunk.copy()
    targets = target_chunk.copy()

    source_row_number = pd.to_numeric(
        targets["source_row_number"],
        errors="coerce",
    ).astype("int64")

    flight_date = pd.to_datetime(
        source["FL_DATE"],
        errors="coerce",
    )

    departure_hour, departure_minute = parse_hhmm(
        source["CRS_DEP_TIME"]
    )

    arrival_hour, arrival_minute = parse_hhmm(
        source["CRS_ARR_TIME"]
    )

    airline_code = source["AIRLINE_CODE"].astype("string")
    dot_code = source["DOT_CODE"].astype("string")

    carrier_identifier = (
        "DOT_" + dot_code.fillna("MISSING")
    )

    route = (
        source["ORIGIN"].astype("string").fillna("MISSING")
        + "-"
        + source["DEST"].astype("string").fillna("MISSING")
    )

    output = pd.DataFrame(
        {
            "source_row_number": source_row_number,
            "FL_DATE": source["FL_DATE"],
            "scheduled_departure_date": flight_date.dt.date.astype("string"),
            "scheduled_departure_year": flight_date.dt.year.astype("Int16"),
            "scheduled_departure_month": flight_date.dt.month.astype("Int8"),
            "scheduled_departure_quarter": flight_date.dt.quarter.astype("Int8"),
            "scheduled_departure_day_of_month": (
                flight_date.dt.day.astype("Int8")
            ),
            "scheduled_departure_day_of_week": (
                flight_date.dt.dayofweek.astype("Int8")
            ),
            "scheduled_departure_week_of_year": (
                flight_date.dt.isocalendar().week.astype("Int8")
            ),
            "scheduled_departure_is_weekend": (
                flight_date.dt.dayofweek.isin([5, 6]).astype("Int8")
            ),
            "scheduled_departure_hour_local": departure_hour,
            "scheduled_departure_minute_local": departure_minute,
            "scheduled_arrival_hour_local": arrival_hour,
            "scheduled_arrival_minute_local": arrival_minute,
            "scheduled_elapsed_time_minutes": pd.to_numeric(
                source["CRS_ELAPSED_TIME"],
                errors="coerce",
            ),
            "distance_miles": pd.to_numeric(
                source["DISTANCE"],
                errors="coerce",
            ),
            "origin_airport": source["ORIGIN"].astype("string"),
            "destination_airport": source["DEST"].astype("string"),
            "route": route,
            "airline_code": airline_code,
            "dot_code": dot_code,
            "carrier_identifier": carrier_identifier,
            "flight_number": pd.to_numeric(
                source["FL_NUMBER"],
                errors="coerce",
            ),
            "scheduled_departure_time_valid": departure_hour.notna()
            & departure_minute.notna(),
            "scheduled_arrival_time_valid": arrival_hour.notna()
            & arrival_minute.notna(),
            "scheduled_date_valid": flight_date.notna(),
            "feature_code_version": "phase3_step9_v1",
        }
    )

    target_columns_to_attach = [
        "cancelled_target",
        "arrival_delay_minutes",
        "severe_delay_60",
        "severe_delay_90",
        "severe_delay_120",
        "severe_delay_180",
        "completed_flight",
    ]

    for column in target_columns_to_attach:
        output[column] = targets[column].to_numpy()

    return output


def main() -> None:
    """Build and write the base feature table in chunks."""

    validate_files()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    source_columns = sorted(REQUIRED_SOURCE_COLUMNS)

    target_columns = sorted(REQUIRED_TARGET_COLUMNS)

    source_reader = pd.read_csv(
        RAW_FLIGHT_FILE,
        usecols=source_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    target_reader = pd.read_csv(
        TARGET_FILE,
        usecols=target_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total_rows = 0
    chunk_number = 0
    first_chunk = True

    for source_chunk, target_chunk in zip(
        source_reader,
        target_reader,
    ):
        if len(source_chunk) != len(target_chunk):
            raise ValueError(
                "Source and target chunks have different row counts."
            )

        expected_row_numbers = np.arange(
            total_rows,
            total_rows + len(source_chunk),
            dtype=np.int64,
        )

        actual_row_numbers = (
            target_chunk["source_row_number"]
            .astype("int64")
            .to_numpy()
        )

        if not np.array_equal(
            expected_row_numbers,
            actual_row_numbers,
        ):
            raise ValueError(
                "Source and target rows are no longer aligned."
            )

        feature_chunk = build_features(
            source_chunk=source_chunk,
            target_chunk=target_chunk,
        )

        feature_chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(feature_chunk)
        chunk_number += 1
        first_chunk = False

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_rows:,} rows"
        )

    print("\nBase feature construction completed.")
    print(f"Rows written: {total_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
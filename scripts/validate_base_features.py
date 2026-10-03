"""
Validate the Phase 3 base-feature table against the raw flight CSV
and the already-validated target table.

This script checks row alignment, selected feature calculations,
target alignment, and accidental inclusion of actual-flight outcomes.
It does not train a model or modify any input file.
"""

from itertools import zip_longest
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"
)
TARGET_FILE = (
    PROJECT_ROOT / "data/processed/targets/flight_targets.csv"
)
FEATURE_FILE = (
    PROJECT_ROOT / "data/features/base/base_features.csv"
)

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000

RAW_COLUMNS = [
    "FL_DATE",
    "CRS_DEP_TIME",
    "ORIGIN",
    "DEST",
]
TARGET_COLUMNS = [
    "source_row_number",
    "cancelled_target",
    "arrival_delay_minutes",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
    "completed_flight",
]
FEATURE_COLUMNS = [
    "source_row_number",
    "FL_DATE",
    "scheduled_departure_year",
    "scheduled_departure_month",
    "scheduled_departure_day_of_week",
    "scheduled_departure_is_weekend",
    "scheduled_departure_hour_local",
    "scheduled_departure_minute_local",
    "scheduled_departure_time_valid",
    "scheduled_date_valid",
    "origin_airport",
    "destination_airport",
    "route",
    *TARGET_COLUMNS[1:],
]

# These raw outcome columns must not appear in the base-feature table.
PROHIBITED_RAW_OUTCOMES = {
    "DEP_TIME",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CANCELLATION_CODE",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
}


def check_equal(
    actual: pd.Series,
    expected: pd.Series,
    description: str,
    chunk_number: int,
) -> None:
    """Compare values row by row, accepting matching missing values."""
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    if len(actual) != len(expected):
        raise ValueError(
            f"Chunk {chunk_number}: {description} has different row counts."
        )
    both_missing = actual.isna() & expected.isna()
    both_present = actual.notna() & expected.notna()
    equal_values = actual.eq(expected).fillna(False)
    mismatch = ~(both_missing | (both_present & equal_values))
    if mismatch.any():
        first_position = int(np.flatnonzero(mismatch.to_numpy())[0])
        raise ValueError(
            f"Chunk {chunk_number}: {description} differs at chunk row "
            f"{first_position}. Feature={actual.iloc[first_position]!r}; "
            f"expected={expected.iloc[first_position]!r}"
        )


def as_number(series: pd.Series) -> pd.Series:
    """Convert CSV values to numeric for reliable comparisons."""
    return pd.to_numeric(series, errors="coerce")


def as_text(series: pd.Series) -> pd.Series:
    """Convert CSV values to nullable text for reliable comparisons."""
    return series.astype("string")


def main() -> None:
    """Run the validation without changing the input files."""
    for path in (RAW_FILE, TARGET_FILE, FEATURE_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    feature_header = set(pd.read_csv(FEATURE_FILE, nrows=0).columns)

    missing_features = set(FEATURE_COLUMNS) - feature_header
    if missing_features:
        raise ValueError(
            f"Missing feature-table columns: {sorted(missing_features)}"
        )

    leaked_outcomes = PROHIBITED_RAW_OUTCOMES & feature_header
    if leaked_outcomes:
        raise ValueError(
            f"Raw outcome columns found in feature table: "
            f"{sorted(leaked_outcomes)}"
        )

    raw_reader = pd.read_csv(
        RAW_FILE,
        usecols=RAW_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    target_reader = pd.read_csv(
        TARGET_FILE,
        usecols=TARGET_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    feature_reader = pd.read_csv(
        FEATURE_FILE,
        usecols=FEATURE_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total_rows = 0
    invalid_dates = 0
    invalid_scheduled_departure_times = 0

    for chunk_number, chunks in enumerate(
        zip_longest(raw_reader, target_reader, feature_reader),
        start=1,
    ):
        if any(chunk is None for chunk in chunks):
            raise ValueError(
                "The raw, target, and feature files have different "
                "numbers of chunks."
            )

        raw, targets, features = chunks

        if not (len(raw) == len(targets) == len(features)):
            raise ValueError(
                f"Chunk {chunk_number}: input row counts differ."
            )

        expected_ids = pd.Series(
            np.arange(total_rows, total_rows + len(raw)),
            dtype="int64",
        )

        for table_name, table in (
            ("target", targets),
            ("feature", features),
        ):
            check_equal(
                as_number(table["source_row_number"]),
                expected_ids,
                f"{table_name} source_row_number",
                chunk_number,
            )

        check_equal(
            as_text(features["FL_DATE"]),
            as_text(raw["FL_DATE"]),
            "flight date",
            chunk_number,
        )

        for output_column, source_column in (
            ("origin_airport", "ORIGIN"),
            ("destination_airport", "DEST"),
        ):
            check_equal(
                as_text(features[output_column]),
                as_text(raw[source_column]),
                output_column,
                chunk_number,
            )

        expected_route = (
            as_text(raw["ORIGIN"]).fillna("MISSING")
            + "-"
            + as_text(raw["DEST"]).fillna("MISSING")
        )
        check_equal(
            as_text(features["route"]),
            expected_route,
            "route",
            chunk_number,
        )

        dates = pd.to_datetime(
            raw["FL_DATE"], errors="coerce", format="%Y-%m-%d"
        )
        valid_dates = dates.notna()

        check_equal(
            as_number(features["scheduled_date_valid"]),
            valid_dates.astype("int8"),
            "scheduled_date_valid",
            chunk_number,
        )
        check_equal(
            as_number(features["scheduled_departure_year"]),
            dates.dt.year,
            "scheduled_departure_year",
            chunk_number,
        )
        check_equal(
            as_number(features["scheduled_departure_month"]),
            dates.dt.month,
            "scheduled_departure_month",
            chunk_number,
        )
        check_equal(
            as_number(features["scheduled_departure_day_of_week"]),
            dates.dt.dayofweek,
            "scheduled_departure_day_of_week",
            chunk_number,
        )

        # A missing date must not silently become "not weekend."
        expected_weekend = dates.dt.dayofweek.isin([5, 6])
        weekend_mismatch = (
            valid_dates
            & as_number(features["scheduled_departure_is_weekend"])
            .ne(expected_weekend.astype("int8"))
        )
        if weekend_mismatch.any():
            raise ValueError(
                f"Chunk {chunk_number}: weekend indicator is incorrect."
            )

        time_value = as_number(raw["CRS_DEP_TIME"])
        integer_like = time_value.notna() & time_value.eq(
            np.floor(time_value)
        )
        hours = np.floor(time_value / 100)
        minutes = time_value % 100
        valid_time = (
            integer_like
            & hours.between(0, 23)
            & minutes.between(0, 59)
        )

        check_equal(
            as_number(features["scheduled_departure_time_valid"]),
            valid_time.astype("int8"),
            "scheduled_departure_time_valid",
            chunk_number,
        )
        check_equal(
            as_number(features["scheduled_departure_hour_local"]),
            hours.where(valid_time),
            "scheduled_departure_hour_local",
            chunk_number,
        )
        check_equal(
            as_number(features["scheduled_departure_minute_local"]),
            minutes.where(valid_time),
            "scheduled_departure_minute_local",
            chunk_number,
        )

        for column in TARGET_COLUMNS[1:]:
            if column == "completed_flight":
                actual = as_number(features[column])
                expected = as_number(targets[column])
            else:
                actual = as_number(features[column])
                expected = as_number(targets[column])

            check_equal(
                actual,
                expected,
                f"attached target {column}",
                chunk_number,
            )

        invalid_dates += int((~valid_dates).sum())
        invalid_scheduled_departure_times += int((~valid_time).sum())
        total_rows += len(raw)
        print(f"Validated chunk {chunk_number}: {total_rows:,} rows")

    if total_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; found {total_rows:,}."
        )

    print("\nBase-feature validation passed.")
    print(f"Rows validated: {total_rows:,}")
    print(f"Invalid flight dates: {invalid_dates:,}")
    print(
        "Invalid scheduled departure times: "
        f"{invalid_scheduled_departure_times:,}"
    )
    print(
        "\nImportant: This table contains target columns for alignment. "
        "Do not pass the whole table into a model as predictors."
    )
    print(
        "UTC prediction timestamps and two-hour feature availability "
        "have not yet been validated."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\nVALIDATION FAILED: {error}")
        sys.exit(1)
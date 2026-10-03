"""
Phase 3 — Step 2
Construct leakage-safe target labels from the audited flight CSV.

Targets created:
- severe_delay_60
- severe_delay_90
- severe_delay_120
- severe_delay_180
- cancelled_target
- arrival_delay_minutes

The raw CSV is never modified.

Target rules:
- Severe-delay labels require a completed, non-cancelled,
  non-diverted flight with a valid ARR_DELAY.
- Cancellation is copied from CANCELLED only when the source value
  is valid and equals either 0 or 1.
- Arrival-delay regression uses only completed flights with a valid ARR_DELAY.
- Source delay-cause fields are not used as predictors or targets.
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
)

OUTPUT_FILE = OUTPUT_DIRECTORY / "flight_targets.csv"

CHUNK_SIZE = 250_000

SEVERE_DELAY_THRESHOLDS = {
    "severe_delay_60": 60,
    "severe_delay_90": 90,
    "severe_delay_120": 120,
    "severe_delay_180": 180,
}

REQUIRED_COLUMNS = {
    "FL_DATE",
    "AIRLINE",
    "AIRLINE_CODE",
    "DOT_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
}


def validate_source_file() -> None:
    """Confirm that the raw source file exists and has required columns."""

    if not RAW_FLIGHT_FILE.exists():
        raise FileNotFoundError(
            f"Raw flight file was not found:\n{RAW_FLIGHT_FILE}"
        )

    source_header = pd.read_csv(RAW_FLIGHT_FILE, nrows=0)
    source_columns = set(source_header.columns)

    missing_columns = REQUIRED_COLUMNS - source_columns

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Required source columns are missing: {missing_text}"
        )


def clean_binary_source_column(
    series: pd.Series,
    column_name: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert a source binary field to a nullable integer series.

    Returns:
        cleaned_values:
            Values are 0, 1, or pandas.NA.
        valid_values:
            True only when the source value is exactly 0 or 1.
    """

    numeric_values = pd.to_numeric(series, errors="coerce")

    valid_values = numeric_values.isin([0, 1])

    cleaned_values = numeric_values.where(valid_values).astype("Int8")

    if column_name == "CANCELLED":
        cleaned_values.name = "cancelled_source"
    else:
        cleaned_values.name = "diverted_source"

    return cleaned_values, valid_values


def build_targets(chunk: pd.DataFrame, row_offset: int) -> pd.DataFrame:
    """Build target columns for one source chunk."""

    chunk = chunk.copy()

    chunk["source_row_number"] = np.arange(
        row_offset,
        row_offset + len(chunk),
        dtype=np.int64,
    )

    cancelled_source, cancelled_valid = clean_binary_source_column(
        chunk["CANCELLED"],
        "CANCELLED",
    )

    diverted_source, diverted_valid = clean_binary_source_column(
        chunk["DIVERTED"],
        "DIVERTED",
    )

    arrival_delay = pd.to_numeric(
        chunk["ARR_DELAY"],
        errors="coerce",
    )

    arrival_delay_valid = arrival_delay.notna() & np.isfinite(arrival_delay)

    completed_flight = (
        cancelled_valid
        & diverted_valid
        & cancelled_source.eq(0)
        & diverted_source.eq(0)
        & arrival_delay_valid
    )

    target_frame = pd.DataFrame(
        {
            "source_row_number": chunk["source_row_number"],
            "FL_DATE": chunk["FL_DATE"],
            "AIRLINE": chunk["AIRLINE"],
            "AIRLINE_CODE": chunk["AIRLINE_CODE"],
            "DOT_CODE": chunk["DOT_CODE"],
            "FL_NUMBER": chunk["FL_NUMBER"],
            "ORIGIN": chunk["ORIGIN"],
            "DEST": chunk["DEST"],
            "CRS_DEP_TIME": chunk["CRS_DEP_TIME"],
            "cancelled_source": cancelled_source,
            "diverted_source": diverted_source,
            "cancelled_source_valid": cancelled_valid,
            "diverted_source_valid": diverted_valid,
            "arrival_delay_valid": arrival_delay_valid,
            "completed_flight": completed_flight,
            "cancelled_target": cancelled_source,
            "arrival_delay_minutes": arrival_delay.where(completed_flight),
        }
    )

    for target_name, threshold in SEVERE_DELAY_THRESHOLDS.items():
        target_values = pd.Series(pd.NA, index=chunk.index, dtype="Int8")

        target_values.loc[completed_flight] = (
            arrival_delay.loc[completed_flight] >= threshold
        ).astype("int8")

        target_frame[target_name] = target_values.to_numpy()

    target_frame["target_exclusion_reason"] = np.select(
        [
            ~cancelled_valid,
            cancelled_source.eq(1),
            ~diverted_valid,
            diverted_source.eq(1),
            ~arrival_delay_valid,
        ],
        [
            "invalid_cancelled_value",
            "cancelled_flight",
            "invalid_diverted_value",
            "diverted_flight",
            "missing_or_invalid_arrival_delay",
        ],
        default="eligible_completed_flight",
    )

    target_frame["source_file"] = str(
        RAW_FLIGHT_FILE.relative_to(PROJECT_ROOT)
    )

    target_frame["target_code_version"] = "phase3_step2_v1"

    return target_frame


def main() -> None:
    """Read the source file in chunks and write the target table."""

    validate_source_file()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    total_chunks = 0
    first_chunk = True

    reader = pd.read_csv(
        RAW_FLIGHT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        target_chunk = build_targets(
            chunk=chunk,
            row_offset=total_rows,
        )

        target_chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(chunk)
        total_chunks += 1
        first_chunk = False

        print(
            f"Processed chunk {total_chunks}: "
            f"{total_rows:,} rows"
        )

    print("\nTarget construction completed.")
    print(f"Rows written: {total_rows:,}")
    print(f"Output file: {OUTPUT_FILE}")

    target_sample = pd.read_csv(
        OUTPUT_FILE,
        nrows=5,
    )

    print("\nOutput columns:")
    for column in target_sample.columns:
        print(f"- {column}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
"""
Phase 3 — Step 1
Inspect the audited flight CSV before constructing targets.

This script:
1. Confirms that the raw flight file exists.
2. Reads only the CSV header and a small sample.
3. Prints the exact column names.
4. Checks whether the required target fields exist.
5. Prints sample values for those fields.

No target file is created in this step.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FLIGHT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "flights_kaggle"
    / "flights_sample_3m.csv"
)

REQUIRED_TARGET_COLUMNS = {
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
}

SAMPLE_SIZE = 10


def main() -> None:
    """Inspect the source schema and required target columns."""

    if not RAW_FLIGHT_FILE.exists():
        print(f"ERROR: Source file was not found:\n{RAW_FLIGHT_FILE}")
        sys.exit(1)

    print(f"Reading source file:\n{RAW_FLIGHT_FILE}\n")

    header = pd.read_csv(
        RAW_FLIGHT_FILE,
        nrows=0,
    )

    columns = list(header.columns)

    print(f"Number of columns: {len(columns)}")
    print("\nExact source columns:")
    for index, column in enumerate(columns, start=1):
        print(f"{index:03d}. {column}")

    available_columns = set(columns)
    missing_columns = REQUIRED_TARGET_COLUMNS - available_columns

    print("\nRequired target-column check:")

    for column in sorted(REQUIRED_TARGET_COLUMNS):
        status = "FOUND" if column in available_columns else "MISSING"
        print(f"- {column}: {status}")

    if missing_columns:
        print("\nERROR: Required columns are missing:")
        for column in sorted(missing_columns):
            print(f"- {column}")

        print(
            "\nDo not continue to target construction until the source schema "
            "has been checked."
        )
        sys.exit(1)

    sample_columns = [
        column
        for column in [
            "FL_DATE",
            "AIRLINE",
            "DOT_CODE",
            "FL_NUM",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "DEP_TIME",
            "ARR_TIME",
            "ARR_DELAY",
            "CANCELLED",
            "DIVERTED",
        ]
        if column in available_columns
    ]

    sample = pd.read_csv(
        RAW_FLIGHT_FILE,
        usecols=sample_columns,
        nrows=SAMPLE_SIZE,
    )

    print("\nSample values for target-related fields:")
    print(sample.to_string(index=False))

    print("\nStep 1 completed successfully.")
    print("The source contains the required target fields.")
    print("Do not modify the raw CSV file.")


if __name__ == "__main__":
    main()
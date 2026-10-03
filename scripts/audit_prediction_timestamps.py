"""
Audit the completed prediction-timestamp file.

Checks:
- Three million rows with continuous source-row numbers.
- Every successful conversion has both UTC timestamps.
- Prediction time is exactly two hours before scheduled departure.
- Unsuccessful conversions have no invented UTC timestamps.
- Prints affected airports and DST-ambiguous records.

Does not modify any file.
"""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/flight_prediction_timestamps.csv"

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000

COLUMNS = [
    "source_row_number",
    "FL_DATE",
    "CRS_DEP_TIME",
    "ORIGIN",
    "origin_iana_timezone",
    "scheduled_departure_local",
    "scheduled_departure_utc",
    "prediction_timestamp_utc",
    "timestamp_conversion_status",
]


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"File not found: {INPUT}")

    total_rows = 0
    status_counts = Counter()
    exceptions = []

    reader = pd.read_csv(
        INPUT,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = chunk.reset_index(drop=True)

        expected_ids = np.arange(
            total_rows,
            total_rows + len(chunk),
            dtype=np.int64,
        )
        actual_ids = pd.to_numeric(
            chunk["source_row_number"],
            errors="coerce",
        ).to_numpy()

        if not np.array_equal(actual_ids, expected_ids):
            raise ValueError(
                f"Source-row numbers are misaligned in chunk {chunk_number}."
            )

        status = chunk["timestamp_conversion_status"]
        successful = status.eq("ok")

        departure_utc = pd.to_datetime(
            chunk["scheduled_departure_utc"],
            utc=True,
            errors="coerce",
        )
        prediction_utc = pd.to_datetime(
            chunk["prediction_timestamp_utc"],
            utc=True,
            errors="coerce",
        )

        if (
            departure_utc.loc[successful].isna().any()
            or prediction_utc.loc[successful].isna().any()
        ):
            raise ValueError(
                f"Successful rows lack UTC timestamps in chunk {chunk_number}."
            )

        actual_difference = (
            departure_utc.loc[successful]
            - prediction_utc.loc[successful]
        )
        if not actual_difference.eq(pd.Timedelta(hours=2)).all():
            raise ValueError(
                f"Two-hour prediction horizon failed in chunk {chunk_number}."
            )

        unsuccessful = ~successful
        if (
            departure_utc.loc[unsuccessful].notna().any()
            or prediction_utc.loc[unsuccessful].notna().any()
        ):
            raise ValueError(
                f"Unsuccessful rows have UTC timestamps in chunk "
                f"{chunk_number}."
            )

        status_counts.update(status.value_counts(dropna=False).to_dict())

        if unsuccessful.any():
            exceptions.append(chunk.loc[unsuccessful, COLUMNS].copy())

        total_rows += len(chunk)
        print(f"Audited chunk {chunk_number}: {total_rows:,} rows")

    if total_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; found {total_rows:,}."
        )

    print("\nTimestamp audit passed.")
    print(f"Rows audited: {total_rows:,}")

    print("\nConversion status:")
    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count:,}")

    if exceptions:
        problem_rows = pd.concat(exceptions, ignore_index=True)

        unmapped = problem_rows.loc[
            problem_rows["timestamp_conversion_status"].eq(
                "unmapped_origin_timezone"
            )
        ]
        print("\nUnmapped origin airports and affected flight counts:")
        print(unmapped["ORIGIN"].value_counts(dropna=False).to_string())

        ambiguous = problem_rows.loc[
            problem_rows["timestamp_conversion_status"].eq("dst_ambiguous")
        ]
        print("\nDST-ambiguous flight details:")
        if ambiguous.empty:
            print("None")
        else:
            print(ambiguous.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\nAUDIT FAILED: {error}")
        sys.exit(1)
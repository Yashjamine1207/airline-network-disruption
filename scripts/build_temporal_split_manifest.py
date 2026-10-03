"""Assign audited flight-date splits without reading target values."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = ROOT / "data/features/base/base_predictors.csv"
OUTPUT = ROOT / "data/processed/model_ready/temporal_split_manifest.csv"
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999


def main() -> None:
    if not PREDICTORS.is_file():
        raise FileNotFoundError(f"Predictor file not found: {PREDICTORS}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    total = 0
    counts = {
        "development": 0,
        "validation": 0,
        "final_test_locked": 0,
    }
    previous_id = -1

    try:
        reader = pd.read_csv(
            PREDICTORS,
            usecols=[
                "source_row_number",
                "FL_DATE",
                "prediction_timestamp_utc",
            ],
            chunksize=CHUNK_SIZE,
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            ids = pd.to_numeric(
                chunk["source_row_number"], errors="coerce"
            )

            if ids.isna().any() or ids.duplicated().any():
                raise ValueError(
                    f"Invalid source-row numbers in chunk {chunk_number}"
                )

            if int(ids.iloc[0]) <= previous_id:
                raise ValueError(
                    "Source-row numbers are not strictly increasing."
                )
            if not ids.is_monotonic_increasing:
                raise ValueError(
                    f"Source-row order failed in chunk {chunk_number}"
                )
            previous_id = int(ids.iloc[-1])

            dates = pd.to_datetime(
                chunk["FL_DATE"],
                format="%Y-%m-%d",
                errors="coerce",
            )
            if dates.isna().any():
                raise ValueError(
                    f"Invalid flight dates in chunk {chunk_number}"
                )

            timestamps = pd.to_datetime(
                chunk["prediction_timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            if timestamps.isna().any():
                raise ValueError(
                    f"Missing prediction timestamps in chunk {chunk_number}"
                )

            split = np.select(
                [
                    dates.between(
                        "2019-01-01", "2021-12-31"
                    ),
                    dates.between(
                        "2022-01-01", "2022-12-31"
                    ),
                    dates.between(
                        "2023-01-01", "2023-08-31"
                    ),
                ],
                [
                    "development",
                    "validation",
                    "final_test_locked",
                ],
                default="outside_audited_coverage",
            )

            if (split == "outside_audited_coverage").any():
                raise ValueError(
                    f"Out-of-coverage rows in chunk {chunk_number}"
                )

            manifest = pd.DataFrame({
                "source_row_number": ids.to_numpy(dtype=np.int64),
                "flight_date": chunk["FL_DATE"].to_numpy(),
                "prediction_timestamp_utc": (
                    chunk["prediction_timestamp_utc"].to_numpy()
                ),
                "temporal_split": split,
            })

            manifest.to_csv(
                TEMP,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )

            for name, count in (
                manifest["temporal_split"].value_counts().items()
            ):
                counts[name] += int(count)

            total += len(manifest)
            print(f"Assigned {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; found {total:,}"
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nSplit manifest created.")
    print(f"Development: {counts['development']:,}")
    print(f"Validation: {counts['validation']:,}")
    print(f"Final test (locked): {counts['final_test_locked']:,}")
    print(f"Output: {OUTPUT}")
    print("No labels were read or evaluated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
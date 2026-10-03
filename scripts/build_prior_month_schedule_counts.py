"""
Build past-only historical scheduled-flight counts.

For each prediction, count flights scheduled in the previous completed
UTC calendar month for its carrier, origin airport, and route.

No outcome or label file is read. Current-month and future flights cannot
contribute to a feature value.
"""

from collections import Counter
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/features/base/base_predictors.csv"
OUTPUT = (
    ROOT / "data/features/historical/"
    "prior_month_schedule_counts.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999

ENTITIES = {
    "carrier": "carrier_identifier",
    "origin": "origin_airport",
    "route": "route",
}

INPUT_COLUMNS = [
    "source_row_number",
    "prediction_timestamp_utc",
    "scheduled_departure_utc",
    *ENTITIES.values(),
]


def utc_month_number(values: pd.Series) -> pd.Series:
    """Represent a UTC calendar month as year * 12 + month."""
    timestamps = pd.to_datetime(
        values, utc=True, errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("Missing or invalid UTC timestamp encountered.")

    return (
        timestamps.dt.year * 12 + timestamps.dt.month
    ).astype("int32")


def read_chunks():
    """Read only permitted schedule fields; no labels or outcomes."""
    return pd.read_csv(
        INPUT,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )


def build_monthly_counts() -> dict[str, Counter]:
    """First pass: count scheduled flights by entity and UTC month."""
    counts = {name: Counter() for name in ENTITIES}
    rows = 0

    for chunk_number, chunk in enumerate(read_chunks(), start=1):
        departure_month = utc_month_number(
            chunk["scheduled_departure_utc"]
        )

        for name, column in ENTITIES.items():
            entity = (
                chunk[column]
                .astype("string")
                .fillna("MISSING")
            )

            monthly = (
                pd.DataFrame({
                    "entity": entity,
                    "month": departure_month,
                })
                .groupby(["entity", "month"], dropna=False)
                .size()
            )

            for (entity_value, month), count in monthly.items():
                counts[name][
                    (str(entity_value), int(month))
                ] += int(count)

        rows += len(chunk)
        print(f"Count pass {chunk_number}: {rows:,} rows")

    if rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; found {rows:,}."
        )

    return counts


def write_features(counts: dict[str, Counter]) -> None:
    """Second pass: look up only the preceding UTC month's counts."""
    total = 0
    last_id = -1

    TEMP.unlink(missing_ok=True)

    try:
        for chunk_number, chunk in enumerate(
            read_chunks(), start=1
        ):
            chunk = chunk.reset_index(drop=True)

            ids = pd.to_numeric(
                chunk["source_row_number"],
                errors="coerce",
            )

            if (
                ids.isna().any()
                or not ids.is_monotonic_increasing
                or ids.duplicated().any()
                or int(ids.iloc[0]) <= last_id
            ):
                raise ValueError(
                    f"Invalid row alignment in chunk {chunk_number}."
                )
            last_id = int(ids.iloc[-1])

            prediction_time = pd.to_datetime(
                chunk["prediction_timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            departure_time = pd.to_datetime(
                chunk["scheduled_departure_utc"],
                utc=True,
                errors="coerce",
            )

            if prediction_time.isna().any() or departure_time.isna().any():
                raise ValueError(
                    f"Invalid timestamp in chunk {chunk_number}."
                )

            if not (
                departure_time - prediction_time
            ).eq(pd.Timedelta(hours=2)).all():
                raise ValueError(
                    f"Prediction horizon failed in chunk {chunk_number}."
                )

            # Subtract one MONTH, not a fixed number of days.
            history_month = (
                prediction_time.dt.year * 12
                + prediction_time.dt.month
                - 1
            ).astype("int32")

            result = pd.DataFrame({
                "source_row_number": ids.to_numpy(dtype=np.int64),
                "history_utc_month_number": (
                    history_month.to_numpy()
                ),
            })

            for name, column in ENTITIES.items():
                entity_values = (
                    chunk[column]
                    .astype("string")
                    .fillna("MISSING")
                    .astype(str)
                    .to_numpy()
                )

                feature_name = (
                    f"prior_calendar_month_{name}_"
                    "scheduled_flight_count"
                )

                result[feature_name] = np.fromiter(
                    (
                        counts[name].get(
                            (entity, int(month)),
                            0,
                        )
                        for entity, month in zip(
                            entity_values,
                            history_month.to_numpy(),
                        )
                    ),
                    dtype=np.int32,
                    count=len(chunk),
                )

            result["history_feature_version"] = (
                "prior_month_schedule_v1"
            )

            result.to_csv(
                TEMP,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )

            total += len(result)
            print(f"Feature pass {chunk_number}: {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} output rows; "
                f"found {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nHistorical schedule counts created.")
    print(f"Rows: {total:,}")
    print(f"Output: {OUTPUT}")
    print("No outcomes or labels were read.")


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    counts = build_monthly_counts()
    write_features(counts)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
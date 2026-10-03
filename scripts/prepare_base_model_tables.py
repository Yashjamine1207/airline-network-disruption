"""
Create separate base-predictor and label tables.

Inputs are read in their original source-row order:
- Validated base-feature table
- Audited UTC prediction-timestamp table

Only rows with an unambiguous prediction timestamp are included.
Predictors are selected using an explicit allowlist. Target columns
are written to a different file and cannot enter the predictor file.

This script does not train, tune, or evaluate a model.
"""

from itertools import zip_longest
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE_INPUT = ROOT / "data/features/base/base_features.csv"
TIME_INPUT = ROOT / "data/interim/flight_prediction_timestamps.csv"

PREDICTOR_OUTPUT = (
    ROOT / "data/features/base/base_predictors.csv"
)
LABEL_OUTPUT = (
    ROOT / "data/processed/targets/model_labels.csv"
)

PREDICTOR_TEMP = PREDICTOR_OUTPUT.with_suffix(".csv.tmp")
LABEL_TEMP = LABEL_OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_SOURCE_ROWS = 3_000_000
EXPECTED_EXCLUDED_ROWS = 1

# Explicitly approved as candidate schedule/entity predictors.
# Whether the published schedule reflects the exact T-minus-two-hour
# schedule snapshot is not established by this retrospective source.
PREDICTOR_COLUMNS = [
    "scheduled_departure_year",
    "scheduled_departure_month",
    "scheduled_departure_quarter",
    "scheduled_departure_day_of_month",
    "scheduled_departure_day_of_week",
    "scheduled_departure_week_of_year",
    "scheduled_departure_is_weekend",
    "scheduled_departure_hour_local",
    "scheduled_departure_minute_local",
    "scheduled_arrival_hour_local",
    "scheduled_arrival_minute_local",
    "scheduled_elapsed_time_minutes",
    "distance_miles",
    "origin_airport",
    "destination_airport",
    "route",
    "airline_code",
    "dot_code",
    "carrier_identifier",
    "flight_number",
    "scheduled_departure_time_valid",
    "scheduled_arrival_time_valid",
    "scheduled_date_valid",
]

LABEL_COLUMNS = [
    "cancelled_target",
    "completed_flight",
    "arrival_delay_minutes",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
]

BASE_COLUMNS = [
    "source_row_number",
    "FL_DATE",
    *PREDICTOR_COLUMNS,
    *LABEL_COLUMNS,
]

TIME_COLUMNS = [
    "source_row_number",
    "scheduled_departure_utc",
    "prediction_timestamp_utc",
    "timestamp_conversion_status",
]

FORBIDDEN_PREDICTOR_COLUMNS = set(LABEL_COLUMNS) | {
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DEP_DELAY",
    "DEP_TIME",
    "ARR_TIME",
    "CANCELLATION_CODE",
    "target_exclusion_reason",
}


def main() -> None:
    for path in (BASE_INPUT, TIME_INPUT):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    base_header = set(pd.read_csv(BASE_INPUT, nrows=0).columns)
    time_header = set(pd.read_csv(TIME_INPUT, nrows=0).columns)

    if missing := set(BASE_COLUMNS) - base_header:
        raise ValueError(f"Missing base columns: {sorted(missing)}")
    if missing := set(TIME_COLUMNS) - time_header:
        raise ValueError(f"Missing timestamp columns: {sorted(missing)}")
    if leaked := set(PREDICTOR_COLUMNS) & FORBIDDEN_PREDICTOR_COLUMNS:
        raise ValueError(f"Forbidden predictor columns: {sorted(leaked)}")

    PREDICTOR_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LABEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for temp in (PREDICTOR_TEMP, LABEL_TEMP):
        temp.unlink(missing_ok=True)

    base_reader = pd.read_csv(
        BASE_INPUT,
        usecols=BASE_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    time_reader = pd.read_csv(
        TIME_INPUT,
        usecols=TIME_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    source_rows = 0
    eligible_rows = 0
    excluded_rows = 0

    try:
        for chunk_number, pair in enumerate(
            zip_longest(base_reader, time_reader),
            start=1,
        ):
            if pair[0] is None or pair[1] is None:
                raise ValueError(
                    "Base and timestamp files have different row counts."
                )

            base, times = pair
            base = base.reset_index(drop=True)
            times = times.reset_index(drop=True)

            if len(base) != len(times):
                raise ValueError(
                    f"Chunk {chunk_number}: row counts differ."
                )

            expected_ids = np.arange(
                source_rows,
                source_rows + len(base),
                dtype=np.int64,
            )

            for name, table in (("base", base), ("timestamp", times)):
                actual_ids = pd.to_numeric(
                    table["source_row_number"],
                    errors="coerce",
                ).to_numpy()

                if not np.array_equal(actual_ids, expected_ids):
                    raise ValueError(
                        f"Chunk {chunk_number}: {name} rows are misaligned."
                    )

            eligible = times["timestamp_conversion_status"].eq("ok")

            if times.loc[
                eligible, "prediction_timestamp_utc"
            ].isna().any():
                raise ValueError(
                    f"Chunk {chunk_number}: eligible row lacks "
                    "prediction timestamp."
                )

            predictors = base.loc[
                eligible,
                ["source_row_number", *PREDICTOR_COLUMNS],
            ].copy()

            predictors.insert(
                1,
                "prediction_timestamp_utc",
                times.loc[
                    eligible, "prediction_timestamp_utc"
                ].to_numpy(),
            )

            # These two columns identify the scheduled flight; they
            # are metadata, not automatically model input features.
            predictors.insert(
                2,
                "FL_DATE",
                base.loc[eligible, "FL_DATE"].to_numpy(),
            )
            predictors.insert(
                3,
                "scheduled_departure_utc",
                times.loc[
                    eligible, "scheduled_departure_utc"
                ].to_numpy(),
            )

            labels = base.loc[
                eligible,
                ["source_row_number", "FL_DATE", *LABEL_COLUMNS],
            ].copy()

            if len(predictors) != len(labels):
                raise ValueError(
                    f"Chunk {chunk_number}: predictor/label counts differ."
                )

            if not np.array_equal(
                predictors["source_row_number"].to_numpy(),
                labels["source_row_number"].to_numpy(),
            ):
                raise ValueError(
                    f"Chunk {chunk_number}: predictor/label IDs differ."
                )

            if set(predictors.columns) & FORBIDDEN_PREDICTOR_COLUMNS:
                raise ValueError(
                    "A target or outcome entered the predictor table."
                )

            predictors.to_csv(
                PREDICTOR_TEMP,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )
            labels.to_csv(
                LABEL_TEMP,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )

            source_rows += len(base)
            eligible_rows += len(predictors)
            excluded_rows += len(base) - len(predictors)

            print(
                f"Chunk {chunk_number}: {source_rows:,} source rows; "
                f"{eligible_rows:,} eligible rows"
            )

        if source_rows != EXPECTED_SOURCE_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_SOURCE_ROWS:,} source rows; "
                f"found {source_rows:,}."
            )
        if excluded_rows != EXPECTED_EXCLUDED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_EXCLUDED_ROWS} excluded row; "
                f"found {excluded_rows:,}."
            )

        os.replace(PREDICTOR_TEMP, PREDICTOR_OUTPUT)
        os.replace(LABEL_TEMP, LABEL_OUTPUT)

    except Exception:
        PREDICTOR_TEMP.unlink(missing_ok=True)
        LABEL_TEMP.unlink(missing_ok=True)
        raise

    print("\nBase modelling tables created.")
    print(f"Eligible rows: {eligible_rows:,}")
    print(f"Excluded ambiguous-timestamp rows: {excluded_rows:,}")
    print(f"Predictors: {PREDICTOR_OUTPUT}")
    print(f"Labels: {LABEL_OUTPUT}")
    print(
        "No model has been trained and the final-test labels "
        "have not been inspected."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
"""Join base predictors with tested, past-only schedule counts.

No target, cancellation, arrival-delay, or candidate outcome-time
file is read. Input rows must remain aligned by source_row_number.
"""

from itertools import zip_longest
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data/features/base/base_predictors.csv"
HISTORY = (
    ROOT / "data/features/historical/"
    "prior_month_schedule_counts.csv"
)
OUTPUT = (
    ROOT / "data/features/tabular/"
    "base_plus_prior_schedule.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999

HISTORY_COLUMNS = [
    "source_row_number",
    "history_utc_month_number",
    "prior_calendar_month_carrier_scheduled_flight_count",
    "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count",
]

FORBIDDEN = {
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DEP_DELAY",
    "candidate_gate_arrival_utc",
    "arrival_delay_minutes",
    "cancelled_target",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
    "completed_flight",
}


def main() -> None:
    for path in (BASE, HISTORY):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    base_reader = pd.read_csv(
        BASE, chunksize=CHUNK_SIZE, low_memory=False
    )
    history_reader = pd.read_csv(
        HISTORY,
        usecols=HISTORY_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total = 0
    previous_id = -1

    try:
        for number, pair in enumerate(
            zip_longest(base_reader, history_reader),
            start=1,
        ):
            if pair[0] is None or pair[1] is None:
                raise ValueError("Input files have unequal row counts.")

            base, history = pair
            base = base.reset_index(drop=True)
            history = history.reset_index(drop=True)

            if len(base) != len(history):
                raise ValueError(
                    f"Chunk {number}: input chunk lengths differ."
                )

            base_ids = pd.to_numeric(
                base["source_row_number"], errors="coerce"
            )
            history_ids = pd.to_numeric(
                history["source_row_number"], errors="coerce"
            )

            if (
                base_ids.isna().any()
                or not base_ids.equals(history_ids)
                or not base_ids.is_monotonic_increasing
                or base_ids.duplicated().any()
                or int(base_ids.iloc[0]) <= previous_id
            ):
                raise ValueError(
                    f"Chunk {number}: source rows are misaligned."
                )

            previous_id = int(base_ids.iloc[-1])

            prediction_time = pd.to_datetime(
                base["prediction_timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            if prediction_time.isna().any():
                raise ValueError(
                    f"Chunk {number}: invalid prediction timestamp."
                )

            expected_history_month = (
                prediction_time.dt.year * 12
                + prediction_time.dt.month
                - 1
            )
            actual_history_month = pd.to_numeric(
                history["history_utc_month_number"],
                errors="coerce",
            )

            if not actual_history_month.eq(
                expected_history_month
            ).all():
                raise ValueError(
                    f"Chunk {number}: history month is not "
                    "the previous completed UTC month."
                )

            count_columns = [
                col for col in HISTORY_COLUMNS
                if col.startswith("prior_calendar_month_")
            ]
            counts = history[count_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            if (
                counts.isna().any().any()
                or counts.lt(0).any().any()
                or not np.isfinite(
                    counts.to_numpy(dtype="float64")
                ).all()
            ):
                raise ValueError(
                    f"Chunk {number}: invalid historical counts."
                )

            result = pd.concat(
                [base, history[count_columns]],
                axis=1,
            )

            if set(result.columns) & FORBIDDEN:
                raise ValueError(
                    "An outcome or target entered the predictor table."
                )

            result.to_csv(
                TEMP,
                mode="w" if number == 1 else "a",
                header=(number == 1),
                index=False,
            )

            total += len(result)
            print(f"Joined {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; got {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nApproved base-plus-schedule table created.")
    print(f"Rows: {total:,}")
    print(f"Output: {OUTPUT}")
    print("No labels or completed-flight outcomes were read.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
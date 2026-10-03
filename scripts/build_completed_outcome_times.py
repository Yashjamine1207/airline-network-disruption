"""
Reconstruct a candidate gate-arrival UTC time for completed flights.

Gate arrival = scheduled departure UTC + departure delay
               + actual elapsed time.

Only rows with internally consistent delay/elapsed fields receive
a candidate time. This is an event-time proxy, NOT proof of when
the published dataset made the outcome available.

The locked 2023 final-test period receives no reconstructed outcome.
No output from this script may be used as a predicted flight's feature.
"""

from itertools import zip_longest
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"
TIMES = ROOT / "data/interim/flight_prediction_timestamps.csv"
OUTPUT = ROOT / "data/interim/completed_outcome_times.csv"
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000

RAW_COLUMNS = [
    "FL_DATE",
    "CANCELLED",
    "DIVERTED",
    "ARR_DELAY",
    "DEP_DELAY",
    "ELAPSED_TIME",
    "CRS_ELAPSED_TIME",
]
TIME_COLUMNS = [
    "source_row_number",
    "scheduled_departure_utc",
    "timestamp_conversion_status",
]


def process_chunk(
    raw: pd.DataFrame,
    times: pd.DataFrame,
    offset: int,
) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    times = times.reset_index(drop=True)

    if len(raw) != len(times):
        raise ValueError("Raw and timestamp chunk lengths differ.")

    ids = pd.to_numeric(
        times["source_row_number"], errors="coerce"
    ).to_numpy()
    expected = np.arange(offset, offset + len(raw))

    if not np.array_equal(ids, expected):
        raise ValueError("Raw and timestamp rows are misaligned.")

    flight_date = pd.to_datetime(
        raw["FL_DATE"], format="%Y-%m-%d", errors="coerce"
    )
    if flight_date.isna().any():
        raise ValueError("Invalid source flight date.")

    cancelled = pd.to_numeric(raw["CANCELLED"], errors="coerce")
    diverted = pd.to_numeric(raw["DIVERTED"], errors="coerce")
    arrival_delay = pd.to_numeric(raw["ARR_DELAY"], errors="coerce")
    departure_delay = pd.to_numeric(raw["DEP_DELAY"], errors="coerce")
    elapsed = pd.to_numeric(raw["ELAPSED_TIME"], errors="coerce")
    scheduled_elapsed = pd.to_numeric(
        raw["CRS_ELAPSED_TIME"], errors="coerce"
    )

    scheduled_utc = pd.to_datetime(
        times["scheduled_departure_utc"],
        utc=True,
        errors="coerce",
    )

    development_or_validation = flight_date.lt(
        pd.Timestamp("2023-01-01")
    )
    completed = cancelled.eq(0) & diverted.eq(0)

    finite_fields = (
        arrival_delay.notna()
        & departure_delay.notna()
        & elapsed.notna()
        & scheduled_elapsed.notna()
        & np.isfinite(arrival_delay)
        & np.isfinite(departure_delay)
        & np.isfinite(elapsed)
        & np.isfinite(scheduled_elapsed)
    )
    plausible_duration = (
        elapsed.gt(0)
        & elapsed.le(1440)
        & scheduled_elapsed.gt(0)
    )

    identity_error = (
        arrival_delay
        - (
            departure_delay
            + elapsed
            - scheduled_elapsed
        )
    ).abs()

    status = pd.Series(
        "usable_candidate",
        index=raw.index,
        dtype="string",
    )
    status.loc[cancelled.eq(1)] = "cancelled_no_arrival"
    status.loc[diverted.eq(1) & ~cancelled.eq(1)] = (
        "diverted_excluded"
    )
    status.loc[
        completed & (~finite_fields | ~plausible_duration)
    ] = "invalid_or_missing_timing"
    status.loc[
        completed
        & finite_fields
        & plausible_duration
        & (
            scheduled_utc.isna()
            | times["timestamp_conversion_status"].ne("ok")
        )
    ] = "unresolved_scheduled_timestamp"
    status.loc[
        completed
        & finite_fields
        & plausible_duration
        & scheduled_utc.notna()
        & identity_error.gt(1)
    ] = "inconsistent_timing_fields"

    # Preserve the final test lock regardless of its outcome fields.
    status.loc[~development_or_validation] = "final_test_locked"

    usable = status.eq("usable_candidate")
    candidate_arrival = (
        scheduled_utc
        + pd.to_timedelta(departure_delay, unit="m")
        + pd.to_timedelta(elapsed, unit="m")
    ).where(usable)

    if candidate_arrival.loc[usable].isna().any():
        raise ValueError("Usable row has no candidate arrival time.")

    return pd.DataFrame({
        "source_row_number": ids.astype("int64"),
        "candidate_gate_arrival_utc": (
            candidate_arrival.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "outcome_time_status": status,
        "outcome_time_code_version": "phase3_candidate_v1",
    })


def main() -> None:
    for path in (RAW, TIMES):
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    raw_reader = pd.read_csv(
        RAW, usecols=RAW_COLUMNS, chunksize=CHUNK_SIZE
    )
    time_reader = pd.read_csv(
        TIMES, usecols=TIME_COLUMNS, chunksize=CHUNK_SIZE
    )

    total = 0
    statuses = {}

    try:
        for number, pair in enumerate(
            zip_longest(raw_reader, time_reader), start=1
        ):
            if pair[0] is None or pair[1] is None:
                raise ValueError("Input files have unequal row counts.")

            result = process_chunk(pair[0], pair[1], total)
            result.to_csv(
                TEMP,
                mode="w" if number == 1 else "a",
                header=(number == 1),
                index=False,
            )

            for name, count in result[
                "outcome_time_status"
            ].value_counts().items():
                statuses[name] = statuses.get(name, 0) + int(count)

            total += len(result)
            print(f"Processed {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; found {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nCandidate outcome-time table created.")
    for name, count in sorted(statuses.items()):
        print(f"{name}: {count:,}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
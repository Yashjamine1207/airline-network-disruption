"""Join tested network features to the label-free tabular table."""

from itertools import zip_longest
from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data/features/tabular/"
    "base_plus_prior_schedule.csv"
)
NETWORK = (
    ROOT / "data/features/network/"
    "prior_month_airport_network.csv"
)
OUTPUT = (
    ROOT / "data/features/tabular/"
    "phase3_schedule_network_features.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999

NETWORK_FEATURES = [
    "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree",
    "prior_month_destination_in_degree",
    "prior_month_destination_weighted_in_degree",
]

NETWORK_COLUMNS = [
    "source_row_number",
    "network_history_utc_month_number",
    *NETWORK_FEATURES,
]

PROHIBITED = {
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DEP_DELAY",
    "arrival_delay_minutes",
    "cancelled_target",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
    "candidate_gate_arrival_utc",
}


def main() -> None:
    for path in (BASE, NETWORK):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    base_reader = pd.read_csv(
        BASE, chunksize=CHUNK_SIZE, low_memory=False
    )
    network_reader = pd.read_csv(
        NETWORK,
        usecols=NETWORK_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total = 0
    previous_id = -1

    try:
        for number, pair in enumerate(
            zip_longest(base_reader, network_reader),
            start=1,
        ):
            if pair[0] is None or pair[1] is None:
                raise ValueError("Input files have unequal row counts.")

            base, network = pair
            base = base.reset_index(drop=True)
            network = network.reset_index(drop=True)

            if len(base) != len(network):
                raise ValueError(
                    f"Chunk {number}: unequal chunk lengths."
                )

            base_ids = pd.to_numeric(
                base["source_row_number"], errors="coerce"
            )
            network_ids = pd.to_numeric(
                network["source_row_number"], errors="coerce"
            )

            if (
                base_ids.isna().any()
                or not base_ids.equals(network_ids)
                or base_ids.duplicated().any()
                or not base_ids.is_monotonic_increasing
                or int(base_ids.iloc[0]) <= previous_id
            ):
                raise ValueError(
                    f"Chunk {number}: row IDs are misaligned."
                )
            previous_id = int(base_ids.iloc[-1])

            prediction = pd.to_datetime(
                base["prediction_timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            if prediction.isna().any():
                raise ValueError(
                    f"Chunk {number}: invalid prediction timestamp."
                )

            expected_month = (
                prediction.dt.year * 12
                + prediction.dt.month
                - 1
            )
            actual_month = pd.to_numeric(
                network["network_history_utc_month_number"],
                errors="coerce",
            )

            if not actual_month.eq(expected_month).all():
                raise ValueError(
                    f"Chunk {number}: network graph is not "
                    "from the previous UTC month."
                )

            values = network[NETWORK_FEATURES].apply(
                pd.to_numeric, errors="coerce"
            )
            if (
                values.isna().any().any()
                or values.lt(0).any().any()
            ):
                raise ValueError(
                    f"Chunk {number}: invalid network metric."
                )

            combined = pd.concat(
                [base, values],
                axis=1,
            )

            if set(combined.columns) & PROHIBITED:
                raise ValueError(
                    "Target or outcome column entered predictors."
                )

            combined.to_csv(
                TEMP,
                mode="w" if number == 1 else "a",
                header=(number == 1),
                index=False,
            )

            total += len(combined)
            print(f"Joined {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; got {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nPhase 3 tabular feature table created.")
    print(f"Rows: {total:,}")
    print(f"Output: {OUTPUT}")
    print("No labels or flight outcomes were read.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
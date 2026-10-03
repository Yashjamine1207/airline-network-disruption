"""Validate the combined schedule-only Phase 3 feature table.

Checks alignment with label IDs and the frozen split manifest without
reading label values or evaluating final-test outcomes.
"""

from itertools import zip_longest
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    ROOT / "data/features/tabular/"
    "phase3_schedule_network_features.csv"
)
LABELS = ROOT / "data/processed/targets/model_labels.csv"
SPLITS = (
    ROOT / "data/processed/model_ready/"
    "temporal_split_manifest.csv"
)

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999

HISTORICAL_COLUMNS = [
    "prior_calendar_month_carrier_scheduled_flight_count",
    "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count",
    "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree",
    "prior_month_destination_in_degree",
    "prior_month_destination_weighted_in_degree",
]

PROHIBITED = {
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DEP_TIME",
    "DEP_DELAY",
    "ARR_TIME",
    "CANCELLATION_CODE",
    "candidate_gate_arrival_utc",
    "arrival_delay_minutes",
    "cancelled_target",
    "completed_flight",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
}


def main() -> None:
    for path in (FEATURES, LABELS, SPLITS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    columns = set(pd.read_csv(FEATURES, nrows=0).columns)

    if leaked := columns & PROHIBITED:
        raise ValueError(f"Outcome columns in features: {sorted(leaked)}")

    if missing := set(HISTORICAL_COLUMNS) - columns:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    feature_reader = pd.read_csv(
        FEATURES, chunksize=CHUNK_SIZE, low_memory=False
    )
    label_id_reader = pd.read_csv(
        LABELS,
        usecols=["source_row_number"],
        chunksize=CHUNK_SIZE,
    )
    split_reader = pd.read_csv(
        SPLITS,
        usecols=[
            "source_row_number",
            "prediction_timestamp_utc",
            "temporal_split",
        ],
        chunksize=CHUNK_SIZE,
    )

    total = 0
    previous_id = -1
    split_counts = {
        "development": 0,
        "validation": 0,
        "final_test_locked": 0,
    }

    for number, triple in enumerate(
        zip_longest(
            feature_reader,
            label_id_reader,
            split_reader,
        ),
        start=1,
    ):
        if any(table is None for table in triple):
            raise ValueError("Input tables have unequal row counts.")

        features, label_ids, splits = (
            table.reset_index(drop=True) for table in triple
        )

        if not (
            len(features) == len(label_ids) == len(splits)
        ):
            raise ValueError(
                f"Chunk {number}: unequal chunk lengths."
            )

        ids = pd.to_numeric(
            features["source_row_number"], errors="coerce"
        )
        expected_label_ids = pd.to_numeric(
            label_ids["source_row_number"], errors="coerce"
        )
        expected_split_ids = pd.to_numeric(
            splits["source_row_number"], errors="coerce"
        )

        if (
            ids.isna().any()
            or not ids.equals(expected_label_ids)
            or not ids.equals(expected_split_ids)
            or ids.duplicated().any()
            or not ids.is_monotonic_increasing
            or int(ids.iloc[0]) <= previous_id
        ):
            raise ValueError(
                f"Chunk {number}: feature, label, and split IDs differ."
            )
        previous_id = int(ids.iloc[-1])

        if not features[
            "prediction_timestamp_utc"
        ].equals(splits["prediction_timestamp_utc"]):
            raise ValueError(
                f"Chunk {number}: prediction timestamps differ."
            )

        prediction = pd.to_datetime(
            features["prediction_timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        if prediction.isna().any():
            raise ValueError(
                f"Chunk {number}: invalid prediction timestamp."
            )

        numeric_features = features[
            HISTORICAL_COLUMNS
        ].apply(pd.to_numeric, errors="coerce")

        if (
            numeric_features.isna().any().any()
            or numeric_features.lt(0).any().any()
            or not np.isfinite(
                numeric_features.to_numpy(dtype="float64")
            ).all()
        ):
            raise ValueError(
                f"Chunk {number}: invalid historical feature value."
            )

        observed_splits = splits[
            "temporal_split"
        ].value_counts()

        for name, count in observed_splits.items():
            if name not in split_counts:
                raise ValueError(f"Unexpected split: {name}")
            split_counts[name] += int(count)

        total += len(features)
        print(f"Validated {total:,} rows")

    if total != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; found {total:,}."
        )

    print("\nSchedule-only feature table validation passed.")
    print(f"Rows: {total:,}")
    for name, count in split_counts.items():
        print(f"{name}: {count:,}")
    print("No target values were read.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nVALIDATION FAILED: {error}")
        sys.exit(1)
"""Train base-feature Ridge regression for completed-flight arrival delay.

Fit the preprocessor and model on development rows only.
Evaluate on validation rows only. The locked test split is absent.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.pipeline import Pipeline

from airline_disruption.models.tabular_preprocessing import (
    BASE_FEATURE_COLUMNS,
    build_base_preprocessor,
    select_base_features,
)
from scripts.train_phase4_cancellation_logistic import (
    INPUT_PATH,
    ROOT,
    sha256_file,
)


OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_ridge_regression_validation.csv"
)
ALPHA = 10.0  # Initial fixed setting; not selected using validation results.


def load_completed_split(
    connection: duckdb.DuckDBPyConnection,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return allowlisted X and arrival-delay y for completed flights."""
    if split not in {"development", "validation"}:
        raise ValueError(f"Disallowed split: {split}")

    feature_sql = ", ".join(
        f'"{column}"' for column in BASE_FEATURE_COLUMNS
    )

    frame = connection.execute(
        f"""
        SELECT
            source_row_number,
            {feature_sql},
            arrival_delay_minutes
        FROM read_parquet(?)
        WHERE temporal_split = ?
          AND completed_flight = 1
          AND arrival_delay_minutes IS NOT NULL
        ORDER BY source_row_number
        """,
        [str(INPUT_PATH), split],
    ).fetch_df()

    if frame.empty:
        raise ValueError(f"No completed arrival-delay rows in {split}.")

    if frame["source_row_number"].duplicated().any():
        raise ValueError(f"Duplicate source-row IDs in {split}.")

    y = frame["arrival_delay_minutes"].to_numpy(dtype=np.float64)

    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite arrival-delay targets in {split}.")

    # An early arrival can have a negative delay; do not clip it.
    X = select_base_features(frame.loc[:, BASE_FEATURE_COLUMNS])

    if len(X) != len(y):
        raise ValueError(f"X/y row mismatch in {split}.")

    return X, y


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        print("Loading development regression cohort...", flush=True)
        X_train, y_train = load_completed_split(connection, "development")

        print("Loading validation regression cohort...", flush=True)
        X_valid, y_valid = load_completed_split(connection, "validation")
    finally:
        connection.close()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_base_preprocessor()),
            (
                "regressor",
                Ridge(
                    alpha=ALPHA,
                    solver="lsqr",
                    tol=1e-3,
                    max_iter=1000,
                ),
            ),
        ]
    )

    print(
        f"Fitting Ridge on {len(y_train):,} completed development flights...",
        flush=True,
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)

    if not np.isfinite(predictions).all():
        raise ValueError("Ridge produced non-finite validation predictions.")

    severe_mask = y_valid >= 120
    severe_count = int(severe_mask.sum())

    result = {
        "experiment_version": "phase4_arrival_delay_base_ridge_v1",
        "target": "arrival_delay_minutes",
        "model": "ridge_regression",
        "feature_set": "base_schedule_carrier_airport_route",
        "training_split": "development_2019_2021",
        "evaluation_split": "validation_2022",
        "training_rows": len(y_train),
        "validation_rows": len(y_valid),
        "mae_minutes": float(mean_absolute_error(y_valid, predictions)),
        "rmse_minutes": float(
            np.sqrt(mean_squared_error(y_valid, predictions))
        ),
        "median_absolute_error_minutes": float(
            median_absolute_error(y_valid, predictions)
        ),
        "mean_prediction_error_minutes": float(
            np.mean(predictions - y_valid)
        ),
        "severe_validation_rows": severe_count,
        "severe_subset_mae_minutes": (
            float(
                mean_absolute_error(
                    y_valid[severe_mask],
                    predictions[severe_mask],
                )
            )
            if severe_count
            else np.nan
        ),
        "alpha": ALPHA,
        "solver": "lsqr",
        "sklearn_version": sklearn.__version__,
        "input_sha256": sha256_file(INPUT_PATH),
        "model_parameters": json.dumps(
            {
                "alpha": ALPHA,
                "solver": "lsqr",
                "tol": 1e-3,
                "max_iter": 1000,
            },
            sort_keys=True,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved validation result: {OUTPUT_PATH}")
    print(f"Validation MAE: {result['mae_minutes']:.4f} minutes")
    print(f"Validation RMSE: {result['rmse_minutes']:.4f} minutes")
    print(
        "Severe-delay subset MAE: "
        f"{result['severe_subset_mae_minutes']:.4f} minutes"
    )
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
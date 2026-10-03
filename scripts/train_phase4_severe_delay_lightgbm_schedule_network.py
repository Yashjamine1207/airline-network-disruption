"""Train LightGBM with added schedule-count and network features.

Uses the same severe-delay target and chronological splits as the base
LightGBM experiment. Adds seven prior-month predictors to test their
incremental value. Does not access the locked 2023 split.
"""

import json

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
)
from scripts.train_phase4_cancellation_logistic import (
    ROOT,
    precision_recall_at_fraction,
    sha256_file,
)
from scripts.train_phase4_severe_delay_lightgbm import (
    set_training_categories,
)


INPUT_PATH = (
    ROOT
    / "data/processed/model_ready/"
    "phase4_schedule_network_dev_validation.parquet"
)
OUTPUT_PATH = (
    ROOT
    / "reports/tables/"
    "phase4_severe_delay_lightgbm_schedule_network_validation.csv"
)
RANDOM_SEED = 42

ADDITIONAL_FEATURES = (
    "prior_calendar_month_carrier_scheduled_flight_count",
    "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count",
    "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree",
    "prior_month_destination_in_degree",
    "prior_month_destination_weighted_in_degree",
)


def load_completed_split(
    connection: duckdb.DuckDBPyConnection,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return completed-flight severe-delay labels and predictors."""
    if split not in {"development", "validation"}:
        raise ValueError(f"Disallowed split: {split}")

    feature_columns = [
        '"source_row_number"',
        *[f'"{name}"' for name in BASE_CATEGORICAL_COLUMNS],
        *[f'"{name}"' for name in ADDITIONAL_FEATURES],
        '"completed_flight"',
        '"severe_delay_120"',
    ]

    frame = connection.execute(
        f"""
        SELECT {", ".join(feature_columns)}
        FROM read_parquet(?)
        WHERE temporal_split = ?
        ORDER BY source_row_number
        """,
        [str(INPUT_PATH), split],
    ).fetch_df()

    if frame.empty:
        raise ValueError(f"No rows found for {split}.")

    if frame["source_row_number"].duplicated().any():
        raise ValueError(f"Duplicate source-row IDs in {split}.")

    eligible = frame["completed_flight"].eq(1) & frame[
        "severe_delay_120"
    ].notna()

    excluded_count = int((~eligible).sum())
    print(
        f"{split}: {int(eligible.sum()):,} completed labelled flights; "
        f"{excluded_count:,} excluded.",
        flush=True,
    )

    cohort = frame.loc[eligible]

    if not cohort["severe_delay_120"].isin([0, 1]).all():
        raise ValueError(f"Invalid severe-delay labels in {split}.")

    y = cohort["severe_delay_120"].to_numpy(dtype=np.int8)

    # Exclude ID and outcome columns from the feature matrix.
    X = cohort.loc[
        :,
        list(BASE_CATEGORICAL_COLUMNS) + list(ADDITIONAL_FEATURES),
    ].copy()

    if len(X) != len(y):
        raise ValueError(f"X/y row mismatch in {split}.")

    return X, y


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        print("Loading development completed-flight cohort...", flush=True)
        X_train, y_train = load_completed_split(connection, "development")

        print("Loading validation completed-flight cohort...", flush=True)
        X_valid, y_valid = load_completed_split(connection, "validation")
    finally:
        connection.close()

    if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
        raise ValueError("Both splits must contain both severe-delay classes.")

    set_training_categories(X_train, X_valid)

    parameters = {
        "objective": "binary",
        "metric": "average_precision",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
        "verbosity": -1,
    }

    model = lgb.LGBMClassifier(**parameters)

    print(
        f"Fitting LightGBM on {len(y_train):,} development flights...",
        flush=True,
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric="average_precision",
        categorical_feature=list(BASE_CATEGORICAL_COLUMNS),
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    probabilities = model.predict_proba(X_valid)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("LightGBM produced non-finite probabilities.")

    precision_1pct, recall_1pct, reviewed_1pct = (
        precision_recall_at_fraction(y_valid, probabilities, 0.01)
    )

    result = {
        "experiment_version": (
            "phase4_severe120_schedule_network_lightgbm_v1"
        ),
        "target": "severe_delay_120",
        "model": "lightgbm",
        "feature_set": "base_plus_schedule_counts_and_network",
        "training_split": "development_2019_2021",
        "evaluation_split": "validation_2022",
        "training_rows": len(y_train),
        "validation_rows": len(y_valid),
        "training_event_rate": float(y_train.mean()),
        "validation_event_rate": float(y_valid.mean()),
        "average_precision": float(
            average_precision_score(y_valid, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "brier_score": float(brier_score_loss(y_valid, probabilities)),
        "mean_predicted_probability": float(probabilities.mean()),
        "precision_at_1pct": precision_1pct,
        "recall_at_1pct": recall_1pct,
        "reviewed_at_1pct": reviewed_1pct,
        "best_iteration": int(model.best_iteration_),
        "random_seed": RANDOM_SEED,
        "lightgbm_version": lgb.__version__,
        "input_sha256": sha256_file(INPUT_PATH),
        "model_parameters": json.dumps(parameters, sort_keys=True),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved validation result: {OUTPUT_PATH}")
    print(f"Best iteration: {result['best_iteration']}")
    print(f"Validation PR-AUC: {result['average_precision']:.6f}")
    print(f"Validation Brier score: {result['brier_score']:.6f}")
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
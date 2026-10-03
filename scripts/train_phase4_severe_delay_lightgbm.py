"""Train base-feature LightGBM for severe arrival delay.

Fit on eligible completed development flights. Use 2022 validation
for early stopping and evaluation. Do not access the locked 2023 split.
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
    INPUT_PATH,
    ROOT,
    precision_recall_at_fraction,
    sha256_file,
)
from scripts.train_phase4_severe_delay_logistic import (
    load_completed_split,
)


OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_severe_delay_lightgbm_validation.csv"
)
RANDOM_SEED = 42


def set_training_categories(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> None:
    """Use development categories for both splits; never fit on validation."""
    for column in BASE_CATEGORICAL_COLUMNS:
        X_train[column] = X_train[column].astype("category")
        X_valid[column] = pd.Categorical(
            X_valid[column],
            categories=X_train[column].cat.categories,
        )


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        print("Loading development completed-flight cohort...", flush=True)
        X_train, y_train = load_completed_split(
            connection, "development"
        )

        print("Loading validation completed-flight cohort...", flush=True)
        X_valid, y_valid = load_completed_split(
            connection, "validation"
        )
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
        eval_set=[(X_valid, y_valid)],
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
        "experiment_version": "phase4_severe120_base_lightgbm_v1",
        "target": "severe_delay_120",
        "model": "lightgbm",
        "feature_set": "base_schedule_carrier_airport_route",
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
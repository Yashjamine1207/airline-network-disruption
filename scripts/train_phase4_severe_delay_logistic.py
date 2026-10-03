"""Train base-feature Logistic Regression for severe arrival delay.

Target: project-defined arrival delay >= 120 minutes.
Cohort: completed flights with an observed severe-delay label.
Fit: development only. Evaluate: 2022 validation only.
"""

import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
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
    precision_recall_at_fraction,
    sha256_file,
)


OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_severe_delay_logistic_validation.csv"
)
RANDOM_SEED = 42


def load_completed_split(
    connection: duckdb.DuckDBPyConnection,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load approved predictors and severe-delay labels for completed flights."""
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
            completed_flight,
            severe_delay_120
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

    # Cancelled, diverted, and missing-outcome flights must not be
    # interpreted as non-severe completed flights.
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

    if cohort.empty or not cohort["severe_delay_120"].isin([0, 1]).all():
        raise ValueError(f"Invalid or empty severe-delay cohort in {split}.")

    y = cohort["severe_delay_120"].to_numpy(dtype=np.int8)

    # This allowlist leaves IDs, completed-flight flags, and labels out of X.
    X = select_base_features(cohort.loc[:, BASE_FEATURE_COLUMNS])

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
        raise ValueError("Both cohorts must contain both severe-delay classes.")

    classifier = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=150,
        tol=1e-3,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_base_preprocessor()),
            ("classifier", classifier),
        ]
    )

    print(
        f"Fitting on {len(y_train):,} development flights...",
        flush=True,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)

    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught_warnings
    )

    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    precision_1pct, recall_1pct, reviewed_1pct = (
        precision_recall_at_fraction(y_valid, probabilities, 0.01)
    )

    result = {
        "experiment_version": "phase4_severe120_base_logistic_v1",
        "target": "severe_delay_120",
        "model": "logistic_regression_unweighted",
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
        "convergence_warning": convergence_warning,
        "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "input_sha256": sha256_file(INPUT_PATH),
        "model_parameters": json.dumps(
            {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "class_weight": None,
                "max_iter": 150,
                "tol": 1e-3,
            },
            sort_keys=True,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved validation result: {OUTPUT_PATH}")
    print(f"Validation PR-AUC: {result['average_precision']:.6f}")
    print(f"Validation Brier score: {result['brier_score']:.6f}")
    print(
        "Mean predicted probability: "
        f"{result['mean_predicted_probability']:.6f}"
    )
    print(f"Convergence warning: {convergence_warning}")
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
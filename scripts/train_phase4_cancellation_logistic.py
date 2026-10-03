"""Train and validate base-feature Logistic Regression for cancellation.

Development: fit preprocessing and model.
Validation: evaluate without refitting or choosing a final threshold.
Locked final test: absent from the input Parquet table.
"""

import hashlib
import json
import warnings
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT / "data/processed/model_ready/phase4_base_dev_validation.parquet"
)
OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_cancellation_logistic_validation.csv"
)

TARGET = "cancelled_target"
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    """Record the exact derived data version used by this experiment."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_split(
    connection: duckdb.DuckDBPyConnection,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one approved split using only allowlisted X columns and y."""
    if split not in {"development", "validation"}:
        raise ValueError(f"Disallowed split: {split}")

    selected_columns = [
        '"source_row_number"',
        *[f'"{name}"' for name in BASE_FEATURE_COLUMNS],
        f'"{TARGET}"',
    ]
    column_sql = ", ".join(selected_columns)

    frame = connection.execute(
        f"""
        SELECT {column_sql}
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

    # Cancellation is defined for scheduled flights. If a target is
    # unavailable, exclude it explicitly rather than treating it as zero.
    eligible = frame[TARGET].notna()
    excluded_count = int((~eligible).sum())

    if excluded_count:
        print(f"{split}: excluded {excluded_count:,} missing {TARGET} labels.")

    frame = frame.loc[eligible]

    if frame.empty or not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f"Invalid or empty cancellation labels in {split}.")

    y = frame[TARGET].to_numpy(dtype=np.int8)

    # Never hand ID, target, or split columns to the preprocessor.
    predictors = frame.loc[:, BASE_FEATURE_COLUMNS]
    X = select_base_features(predictors)

    if len(X) != len(y):
        raise ValueError(f"X/y row mismatch in {split}.")

    return X, y


def precision_recall_at_fraction(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
) -> tuple[float, float, int]:
    """Evaluate a ranked review capacity without selecting a threshold."""
    count = max(1, int(np.ceil(len(y_true) * fraction)))
    ranked_indices = np.argsort(-probabilities, kind="stable")[:count]
    captured_events = int(y_true[ranked_indices].sum())
    total_events = int(y_true.sum())

    precision = captured_events / count
    recall = captured_events / total_events if total_events else 0.0

    return precision, recall, count


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        print("Loading development cancellation cohort...", flush=True)
        X_train, y_train = load_split(connection, "development")

        print("Loading validation cancellation cohort...", flush=True)
        X_valid, y_valid = load_split(connection, "validation")
    finally:
        connection.close()

    if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
        raise ValueError("Both splits must contain cancelled and non-cancelled flights.")

    estimator = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=150,
        tol=1e-3,
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_base_preprocessor()),
            ("classifier", estimator),
        ]
    )

    print(
        f"Fitting on {len(y_train):,} development flights. "
        "This may take several minutes...",
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

    model = pipeline.named_steps["classifier"]

    result = {
        "experiment_version": "phase4_cancellation_base_logistic_v1",
        "target": TARGET,
        "model": "logistic_regression",
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
        "precision_at_1pct": precision_1pct,
        "recall_at_1pct": recall_1pct,
        "reviewed_at_1pct": reviewed_1pct,
        "convergence_warning": convergence_warning,
        "iterations": int(model.n_iter_[0]),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "input_sha256": sha256_file(INPUT_PATH),
        "model_parameters": json.dumps(
            {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
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
    print(f"Convergence warning: {convergence_warning}")
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
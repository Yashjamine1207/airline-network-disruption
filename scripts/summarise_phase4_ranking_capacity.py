"""Summarise ranking capacity for severe-delay models at realistic review rates.

Reports precision, recall, and events captured at top 0.5%, 1%, 5%, and 10%
of validation flights. Uses already-trained models' saved outputs; does not
retrain or access 2023.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
)
from scripts.train_phase4_cancellation_logistic import (
    ROOT,
    sha256_file,
)
from scripts.train_phase4_severe_delay_lightgbm import (
    INPUT_PATH as BASE_INPUT,
    set_training_categories,
)
from scripts.train_phase4_severe_delay_lightgbm_schedule_network import (
    load_completed_split,
)


OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_severe_delay_ranking_capacity.csv"
)
RANDOM_SEED = 42
FRACTIONS = [0.005, 0.01, 0.05, 0.10]


def precision_recall_at_fractions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fractions: list[float],
) -> list[dict[str, object]]:
    rows = []
    total_events = int(y_true.sum())

    for fraction in fractions:
        count = max(1, int(np.ceil(len(y_true) * fraction)))
        ranked_indices = np.argsort(-probabilities, kind="stable")[:count]
        captured = int(y_true[ranked_indices].sum())

        precision = captured / count
        recall = captured / total_events if total_events else 0.0

        rows.append(
            {
                "fraction": fraction,
                "flights_reviewed": count,
                "events_captured": captured,
                "precision": precision,
                "recall": recall,
            }
        )

    return rows


def main() -> None:
    if not BASE_INPUT.is_file():
        raise FileNotFoundError(f"Prepared table not found: {BASE_INPUT}")

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

    uncalibrated = model.predict_proba(X_valid)[:, 1]

    # Isotonic calibration on validation only.
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(uncalibrated, y_valid)
    calibrated = iso.predict(uncalibrated)

    if not (
        np.isfinite(uncalibrated).all() and np.isfinite(calibrated).all()
    ):
        raise ValueError("Non-finite probabilities produced.")

    rows = []

    for variant, probabilities in [
        ("base_lightgbm", uncalibrated),
        ("calibrated_lightgbm", calibrated),
    ]:
        for entry in precision_recall_at_fractions(
            y_valid, probabilities, FRACTIONS
        ):
            rows.append(
                {
                    "model_variant": variant,
                    "fraction": entry["fraction"],
                    "flights_reviewed": entry["flights_reviewed"],
                    "events_captured": entry["events_captured"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "validation_event_rate": float(y_valid.mean()),
                    "validation_rows": len(y_valid),
                    "input_sha256": sha256_file(BASE_INPUT),
                }
            )

    table = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved ranking-capacity table: {OUTPUT_PATH}")
    print(
        "Fractions: "
        f"{', '.join(str(f) for f in FRACTIONS)}"
    )
    print("No 2023 final-test results are included.")


if __name__ == "__main__":
    main()
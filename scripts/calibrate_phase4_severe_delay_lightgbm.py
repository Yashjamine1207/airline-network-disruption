"""Calibrate base LightGBM severe-delay probabilities on validation data.

Fit an isotonic regressor to map uncalibrated probabilities to calibrated
probabilities using the same validation rows used for early stopping.
Do not access the locked 2023 split.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

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
    ROOT
    / "reports/tables/"
    "phase4_severe_delay_lightgbm_isotonic_calibration.csv"
)
RANDOM_SEED = 42


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

    if not np.isfinite(uncalibrated).all():
        raise ValueError("Uncalibrated probabilities are not finite.")

    # Isotonic regression is fitted on the same validation data used for
    # early stopping. This is a validation-only calibration experiment.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(uncalibrated, y_valid)

    calibrated = iso.predict(uncalibrated)

    if not np.isfinite(calibrated).all():
        raise ValueError("Calibrated probabilities are not finite.")

    # Reliability-curve bins for a compact diagnostic table.
    prob_true, prob_pred = calibration_curve(
        y_valid, calibrated, n_bins=10, strategy="quantile"
    )

    reliability_rows = [
        {
            "bin_index": int(i),
            "mean_predicted_probability": float(prob_pred[i]),
            "fraction_of_positives": float(prob_true[i]),
        }
        for i in range(len(prob_true))
    ]

    result = {
        "experiment_version": (
            "phase4_severe120_base_lightgbm_isotonic_v1"
        ),
        "target": "severe_delay_120",
        "model": "lightgbm_isotonic_calibration",
        "feature_set": "base_schedule_carrier_airport_route",
        "training_split": "development_2019_2021",
        "evaluation_split": "validation_2022",
        "training_rows": len(y_train),
        "validation_rows": len(y_valid),
        "training_event_rate": float(y_train.mean()),
        "validation_event_rate": float(y_valid.mean()),
        "uncalibrated_brier_score": float(
            brier_score_loss(y_valid, uncalibrated)
        ),
        "calibrated_brier_score": float(
            brier_score_loss(y_valid, calibrated)
        ),
        "mean_uncalibrated_probability": float(uncalibrated.mean()),
        "mean_calibrated_probability": float(calibrated.mean()),
        "reliability_curve_bins": json.dumps(reliability_rows),
        "random_seed": RANDOM_SEED,
        "lightgbm_version": lgb.__version__,
        "sklearn_version": pd.__version__,
        "input_sha256": sha256_file(BASE_INPUT),
        "model_parameters": json.dumps(parameters, sort_keys=True),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved calibration result: {OUTPUT_PATH}")
    print(
        "Uncalibrated Brier score: "
        f"{result['uncalibrated_brier_score']:.6f}"
    )
    print(
        "Calibrated Brier score: "
        f"{result['calibrated_brier_score']:.6f}"
    )
    print(
        "Mean uncalibrated probability: "
        f"{result['mean_uncalibrated_probability']:.6f}"
    )
    print(
        "Mean calibrated probability: "
        f"{result['mean_calibrated_probability']:.6f}"
    )
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
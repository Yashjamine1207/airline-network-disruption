"""Summarise key Phase 4 validation results across classification and regression.

Reads the already-produced validation tables and writes one compact summary.
Does not retrain models or access the locked 2023 split.
"""

import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "cancellation_logistic_unweighted": (
        "reports/tables/"
        "phase4_cancellation_logistic_unweighted_validation.csv"
    ),
    "severe_delay_lightgbm_base": (
        "reports/tables/"
        "phase4_severe_delay_lightgbm_validation.csv"
    ),
    "severe_delay_lightgbm_calibrated": (
        "reports/tables/"
        "phase4_severe_delay_lightgbm_isotonic_calibration.csv"
    ),
    "arrival_delay_ridge": (
        "reports/tables/phase4_ridge_regression_validation.csv"
    ),
}

OUTPUT_PATH = ROOT / "reports/tables/phase4_validation_summary.csv"


def load_table(key: str, path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.is_file():
        raise FileNotFoundError(f"Expected table not found: {full_path}")

    frame = pd.read_csv(full_path)
    frame["experiment_key"] = key
    return frame


def main() -> None:
    parts = []

    for key, path in TABLES.items():
        table = load_table(key, path)
        parts.append(table)

    summary = pd.concat(parts, ignore_index=True)

    # Keep only the most relevant columns for a compact overview.
    columns_to_keep = [
        "experiment_key",
        "experiment_version",
        "target",
        "model",
        "feature_set",
        "training_rows",
        "validation_rows",
        "training_event_rate",
        "validation_event_rate",
        "average_precision",
        "roc_auc",
        "brier_score",
        "mae_minutes",
        "rmse_minutes",
        "median_absolute_error_minutes",
        "mean_predicted_probability",
        "precision_at_1pct",
        "recall_at_1pct",
    ]

    available_columns = [
        c for c in columns_to_keep if c in summary.columns
    ]
    summary = summary.loc[:, available_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved Phase 4 validation summary: {OUTPUT_PATH}")
    print(
        "Experiments included: "
        f"{', '.join(summary['experiment_key'].unique())}"
    )
    print("No 2023 final-test results are included.")


if __name__ == "__main__":
    main()
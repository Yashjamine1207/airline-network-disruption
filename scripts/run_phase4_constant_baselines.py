"""Fit Phase 4 constant baselines on development; evaluate on validation.

Uses only the Phase 3 split manifest and separated label table.
No final-test rows enter fitting, evaluation, or saved results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)

from airline_disruption.models.baselines import (
    fit_prevalence_baseline,
    predict_event_probability,
)
from airline_disruption.models.regression_baselines import (
    fit_global_delay_baseline,
    predict_global_delay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = (
    PROJECT_ROOT
    / "data/processed/model_ready/temporal_split_manifest.csv"
)
LABEL_PATH = PROJECT_ROOT / "data/processed/targets/model_labels.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports/tables/phase4_constant_baselines_validation.csv"
)

CHUNK_SIZE = 250_000
ALLOWED_SPLITS = {"development", "validation"}
LABEL_COLUMNS = [
    "source_row_number",
    "cancelled_target",
    "completed_flight",
    "arrival_delay_minutes",
    "severe_delay_120",
]


def load_development_and_validation_rows() -> pd.DataFrame:
    """Join labels to approved split rows, excluding the locked test split."""
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Split manifest not found: {SPLIT_PATH}")
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(f"Model labels not found: {LABEL_PATH}")

    split_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        SPLIT_PATH,
        usecols=["source_row_number", "temporal_split"],
        dtype={"source_row_number": "int64", "temporal_split": "string"},
        chunksize=CHUNK_SIZE,
    ):
        # Do not carry locked-test rows into the modelling table.
        selected = chunk.loc[
            chunk["temporal_split"].isin(ALLOWED_SPLITS)
        ].copy()
        split_parts.append(selected)

    splits = pd.concat(split_parts, ignore_index=True)

    if splits.empty or splits["source_row_number"].duplicated().any():
        raise ValueError("Development/validation split IDs are empty or duplicated.")

    if set(splits["temporal_split"].unique()) != ALLOWED_SPLITS:
        raise ValueError("Both development and validation splits are required.")

    allowed_ids = set(splits["source_row_number"].to_numpy())
    label_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        LABEL_PATH,
        usecols=LABEL_COLUMNS,
        dtype={
            "source_row_number": "int64",
            "cancelled_target": "float64",
            "completed_flight": "float64",
            "arrival_delay_minutes": "float64",
            "severe_delay_120": "float64",
        },
        chunksize=CHUNK_SIZE,
    ):
        # The CSV is scanned in chunks, but test-row labels are discarded
        # before any label table is retained or used for evaluation.
        selected = chunk.loc[
            chunk["source_row_number"].isin(allowed_ids)
        ].copy()
        label_parts.append(selected)

    labels = pd.concat(label_parts, ignore_index=True)

    if labels["source_row_number"].duplicated().any():
        raise ValueError("Duplicate source_row_number in selected labels.")

    joined = splits.merge(
        labels,
        on="source_row_number",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        raise ValueError("Some development/validation IDs lack label rows.")

    joined = joined.drop(columns="_merge")

    if not joined["temporal_split"].isin(ALLOWED_SPLITS).all():
        raise ValueError("A locked-test row entered the modelling table.")

    return joined


def binary_cohort(
    data: pd.DataFrame,
    target: str,
    completed_only: bool,
) -> pd.DataFrame:
    """Select eligible, observed binary labels without treating missing as zero."""
    cohort = data

    if completed_only:
        cohort = cohort.loc[cohort["completed_flight"].eq(1)]

    observed = cohort.loc[cohort[target].notna()].copy()

    if not observed[target].isin([0, 1]).all():
        raise ValueError(f"{target} contains a value other than 0 or 1.")

    return observed


def add_classification_result(
    results: list[dict[str, object]],
    data: pd.DataFrame,
    target: str,
    completed_only: bool,
) -> None:
    cohort = binary_cohort(data, target, completed_only)
    train = cohort.loc[cohort["temporal_split"].eq("development")]
    valid = cohort.loc[cohort["temporal_split"].eq("validation")]

    if train.empty or valid.empty:
        raise ValueError(f"{target}: empty development or validation cohort.")

    y_train = train[target].to_numpy(dtype=np.int8)
    y_valid = valid[target].to_numpy(dtype=np.int8)

    if len(np.unique(y_valid)) != 2:
        raise ValueError(f"{target}: validation needs both classes for PR-AUC.")

    model = fit_prevalence_baseline(y_train)
    probabilities = predict_event_probability(model, len(valid))

    results.append(
        {
            "task": "classification",
            "target": target,
            "model": "training_prevalence",
            "feature_set": "none",
            "training_rows": len(train),
            "validation_rows": len(valid),
            "training_event_rate": float(y_train.mean()),
            "validation_event_rate": float(y_valid.mean()),
            "average_precision": float(
                average_precision_score(y_valid, probabilities)
            ),
            "brier_score": float(brier_score_loss(y_valid, probabilities)),
            "mae_minutes": np.nan,
            "rmse_minutes": np.nan,
            "baseline_prediction": float(probabilities[0]),
        }
    )


def add_regression_results(
    results: list[dict[str, object]],
    data: pd.DataFrame,
) -> None:
    cohort = data.loc[
        data["completed_flight"].eq(1)
        & data["arrival_delay_minutes"].notna()
    ].copy()

    if not np.isfinite(
        cohort["arrival_delay_minutes"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Arrival-delay cohort contains non-finite values.")

    train = cohort.loc[cohort["temporal_split"].eq("development")]
    valid = cohort.loc[cohort["temporal_split"].eq("validation")]

    if train.empty or valid.empty:
        raise ValueError("Regression development or validation cohort is empty.")

    y_train = train["arrival_delay_minutes"].to_numpy(dtype=np.float64)
    y_valid = valid["arrival_delay_minutes"].to_numpy(dtype=np.float64)

    for strategy in ("mean", "median"):
        model = fit_global_delay_baseline(y_train, strategy=strategy)
        predictions = predict_global_delay(model, len(valid))

        results.append(
            {
                "task": "regression",
                "target": "arrival_delay_minutes",
                "model": f"global_{strategy}",
                "feature_set": "none",
                "training_rows": len(train),
                "validation_rows": len(valid),
                "training_event_rate": np.nan,
                "validation_event_rate": np.nan,
                "average_precision": np.nan,
                "brier_score": np.nan,
                "mae_minutes": float(mean_absolute_error(y_valid, predictions)),
                "rmse_minutes": float(
                    np.sqrt(mean_squared_error(y_valid, predictions))
                ),
                "baseline_prediction": float(predictions[0]),
            }
        )


def main() -> None:
    data = load_development_and_validation_rows()
    results: list[dict[str, object]] = []

    add_classification_result(
        results, data, target="cancelled_target", completed_only=False
    )
    add_classification_result(
        results, data, target="severe_delay_120", completed_only=True
    )
    add_regression_results(results, data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved four validation baseline results: {OUTPUT_PATH}")
    print(f"Development/validation rows joined: {len(data):,}")
    print("Locked final-test results: not computed.")


if __name__ == "__main__":
    main()
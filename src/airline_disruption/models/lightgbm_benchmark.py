"""LightGBM tabular benchmark for severe arrival delay (>= 120 minutes).

Phase 6, Step 4. This module is the single code path for loading rows, assigning
training roles and fitting LightGBM. The LSTM, GRU and Transformer runs reuse
``load_model_frame`` and ``assign_roles`` so every model sees the same rows.

Protocol ``phase6`` (default)
    fit          development rows with prediction time before 2021-07-01 (UTC)
    early_stop   development rows from 2021-07-01 on. Chooses the number of trees.
    validation   2022. Untouched by early stopping. Reported and used for model selection.
    final test   never loaded here.

Protocol ``phase4_reproduce``
    Phase 4's protocol: fit on all development, stop early on 2022, report on 2022.
    Early stopping and reporting share the same rows, which flatters the number a
    little. Use it only to check this pipeline against the Phase 4 result.

Rules enforced in code
    * The final-test split cannot be loaded unless ``allow_final_test=True``.
    * Embargoed rows and rows without a valid label are dropped.
    * Category levels come from the FIT rows only. A level seen later becomes missing.
    * Prior-month features are set to missing for the first month of data (see
      ``mask_unavailable_history``).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from airline_disruption.features.feature_sets import (
    HISTORY_COLUMNS,
    ID_COLUMN,
    NETWORK_COLUMNS,
    TIMESTAMP_COLUMN,
    categorical_columns_in,
    columns_for,
)
from airline_disruption.validation.temporal_splits import SPLIT_DEVELOPMENT, SPLIT_FINAL_TEST, SPLIT_VALIDATION

TARGET = "severe_delay_120"
ELIGIBILITY_COLUMN = "eligible_severe_delay"

COHORT_PATH = "data/processed/phase6/phase6_cohort_v1.parquet"
PREDICTORS_PATH = "data/processed/phase6/phase6_predictors_v1.parquet"

# Development rows from this instant on are used to choose the number of trees.
EARLY_STOP_START_UTC = pd.Timestamp("2021-07-01", tz="UTC")

# The Phase 3 prior-month features are stored as 0, not missing, for January 2019,
# because the file has no December 2018. Rows scheduled before this instant get
# missing values instead, so the model is not told "no traffic last month".
HISTORY_AVAILABLE_FROM_UTC = pd.Timestamp("2019-02-01", tz="UTC")

ROLE_FIT = "fit"
ROLE_EARLY_STOP = "early_stop"
ROLE_VALIDATION = "validation"
ROLE_FINAL_TEST = "final_test"

PROTOCOL_PHASE6 = "phase6"
PROTOCOL_PHASE4 = "phase4_reproduce"

# Same settings as Phase 4 (train_phase4_severe_delay_lightgbm.py) apart from n_jobs.
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "reg_lambda": 1.0,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50


@dataclass
class ModelFrame:
    """Rows for one feature set. ``X``, ``y`` and ``meta`` share the same row order."""

    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame  # id, prediction and scheduled times, regime, phase6_split, role, day


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model_frame(
    root: str | Path,
    feature_set: str,
    splits: tuple[str, ...] = (SPLIT_DEVELOPMENT, SPLIT_VALIDATION),
    target: str = TARGET,
    eligibility_column: str = ELIGIBILITY_COLUMN,
    allow_final_test: bool = False,
) -> ModelFrame:
    """Load features, labels and metadata for the rows of ``splits``.

    Rows are sorted by prediction time (UTC), then id. Only rows that are not
    embargoed and are eligible for the target are returned.
    """
    if SPLIT_FINAL_TEST in splits and not allow_final_test:
        raise ValueError("The final-test split is locked. Pass allow_final_test=True only in the final evaluation.")

    root = Path(root)
    feature_columns = list(columns_for(feature_set))

    cohort = pd.read_parquet(
        root / COHORT_PATH,
        columns=[ID_COLUMN, "prediction_timestamp_utc", "regime", "phase6_split", eligibility_column, target],
    )
    keep = cohort["phase6_split"].isin(splits) & cohort[eligibility_column].astype(bool)
    cohort = cohort.loc[keep].drop(columns=eligibility_column)
    if cohort[target].isna().any():
        raise ValueError(f"Missing {target} labels among eligible rows")
    # Sort the small cohort table by time. An inner merge keeps the left table's order,
    # so the big feature table never needs sorting (which would copy it).
    cohort = cohort.sort_values(["prediction_timestamp_utc", ID_COLUMN], kind="stable").reset_index(drop=True)

    predictors = pd.read_parquet(root / PREDICTORS_PATH, columns=[ID_COLUMN, TIMESTAMP_COLUMN, *feature_columns])
    merged = cohort.merge(predictors, on=ID_COLUMN, how="inner", validate="one_to_one")
    if len(merged) != len(cohort):
        raise ValueError(f"{len(cohort) - len(merged):,} cohort rows have no predictors")
    del predictors

    meta = merged[[ID_COLUMN, "prediction_timestamp_utc", TIMESTAMP_COLUMN, "regime", "phase6_split"]].copy()
    meta["phase6_split"] = meta["phase6_split"].astype("category")
    meta["day"] = meta["prediction_timestamp_utc"].dt.floor("D")
    y = merged[target].to_numpy(dtype="int8")

    X = merged[feature_columns]
    del merged
    # float32 halves memory. Counts and degrees are whole numbers far below 2**24, so they stay exact.
    wide = [c for c in X.columns if X[c].dtype == np.float64]
    if wide:
        X[wide] = X[wide].astype("float32")
    return ModelFrame(X=X, y=y, meta=meta)


# ---------------------------------------------------------------------------
# Roles, categories, history mask
# ---------------------------------------------------------------------------
def assign_roles(meta: pd.DataFrame, protocol: str = PROTOCOL_PHASE6) -> pd.Series:
    """Label every row fit, early_stop, validation or final_test.

    Roles follow UTC prediction time. Under the ``phase6`` protocol the last six
    months of development choose the number of trees. Under ``phase4_reproduce``
    all development rows are fit rows.
    """
    if protocol not in (PROTOCOL_PHASE6, PROTOCOL_PHASE4):
        raise ValueError(f"Unknown protocol {protocol!r}")
    split = meta["phase6_split"]
    stamps = meta["prediction_timestamp_utc"]
    development = split == SPLIT_DEVELOPMENT
    late_development = development & (stamps >= EARLY_STOP_START_UTC)

    conditions = [split == SPLIT_VALIDATION, split == SPLIT_FINAL_TEST, development]
    choices = [ROLE_VALIDATION, ROLE_FINAL_TEST, ROLE_FIT]
    roles = np.select(conditions, choices, default="unassigned")
    if protocol == PROTOCOL_PHASE6:
        roles = np.where(late_development, ROLE_EARLY_STOP, roles)
    if (roles == "unassigned").any():
        raise ValueError("Rows with an unexpected split label (embargoed rows must be dropped before this point)")
    return pd.Series(roles, index=meta.index, name="role")


def check_role_chronology(meta: pd.DataFrame, roles: pd.Series) -> None:
    """Raise ValueError unless fit < early_stop < validation in UTC prediction time.

    Only roles that are present are compared. Two rows never share a role
    boundary: the latest row of one role must be strictly earlier than the
    earliest row of the next.
    """
    stamps = meta["prediction_timestamp_utc"]
    previous_name, previous_max = None, None
    for name in (ROLE_FIT, ROLE_EARLY_STOP, ROLE_VALIDATION):
        mask = (roles == name).to_numpy()
        if not mask.any():
            continue
        earliest, latest = stamps[mask].min(), stamps[mask].max()
        if previous_max is not None and not previous_max < earliest:
            raise ValueError(f"Role {previous_name} ends at {previous_max} but role {name} starts at {earliest}")
        previous_name, previous_max = name, latest


def apply_fit_categories(X: pd.DataFrame, fit_mask: np.ndarray) -> list[str]:
    """Restrict every categorical column to the levels seen in the FIT rows.

    A level that appears only later becomes missing (NaN). LightGBM treats
    missing as its own branch, so an unseen airport or route is scored without
    the model ever having learned anything about it. Modifies ``X`` in place and
    returns the categorical column names.
    """
    columns = categorical_columns_in(X.columns)
    for column in columns:
        fit_levels = X.loc[fit_mask, column].cat.remove_unused_categories().cat.categories
        X[column] = X[column].cat.set_categories(fit_levels)
    return columns


def mask_unavailable_history(X: pd.DataFrame, meta: pd.DataFrame) -> int:
    """Set prior-month features to NaN for rows scheduled before 2019-02-01 UTC.

    Returns the number of rows masked. Does nothing if the feature set has no
    prior-month columns.
    """
    columns = [c for c in (*HISTORY_COLUMNS, *NETWORK_COLUMNS) if c in X.columns]
    if not columns:
        return 0
    unavailable = (meta[TIMESTAMP_COLUMN] < HISTORY_AVAILABLE_FROM_UTC).to_numpy()
    X.loc[unavailable, columns] = np.nan
    return int(unavailable.sum())


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    model: object
    best_iteration: int
    seconds: float
    params: dict


def fit_lightgbm(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_stop: pd.DataFrame,
    y_stop: np.ndarray,
    n_jobs: int = 8,
    seed: int = 42,
    params: dict | None = None,
) -> FitResult:
    """Fit LightGBM with early stopping on the stop set (metric: PR-AUC)."""
    import lightgbm as lgb  # imported here so modules that only load data do not need LightGBM

    # LightGBM's own error for a one-class set is cryptic ("unseen labels"). Say what is wrong.
    for name, labels in (("fit", y_fit), ("early-stopping", y_stop)):
        if len(np.unique(labels)) < 2:
            raise ValueError(f"The {name} rows contain only one class ({len(labels):,} rows). "
                             "A smoke-run subsample may be too small for a rare event.")

    settings = {**(params or LIGHTGBM_PARAMS), "random_state": seed, "n_jobs": n_jobs}
    categorical = categorical_columns_in(X_fit.columns)
    model = lgb.LGBMClassifier(**settings)

    started = time.perf_counter()
    with warnings.catch_warnings():
        # LightGBM 4.7 prefers eval_X/eval_y, but eval_set works in every 4.x release.
        warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_stop, y_stop)],
            eval_metric="average_precision",
            categorical_feature=categorical if categorical else "auto",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
        )
    seconds = time.perf_counter() - started
    return FitResult(model=model, best_iteration=int(model.best_iteration_), seconds=seconds, params=settings)


def predict_scores(fit: FitResult, X: pd.DataFrame) -> np.ndarray:
    """Probability of the event, using the best iteration."""
    scores = fit.model.predict_proba(X, num_iteration=fit.best_iteration)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("LightGBM produced non-finite probabilities")
    return scores

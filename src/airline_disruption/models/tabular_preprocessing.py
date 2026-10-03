"""Explicit, leakage-conscious preprocessing for Phase 4 base features.

This module defines the schedule/carrier/airport/route feature set.
Historical counts and network features will be added in later ablations.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_NUMERIC_COLUMNS = (
    "scheduled_departure_year",
    "scheduled_departure_month",
    "scheduled_departure_day_of_week",
    "scheduled_departure_hour_local",
    "scheduled_departure_minute_local",
    "scheduled_arrival_hour_local",
    "scheduled_arrival_minute_local",
    "scheduled_elapsed_time_minutes",
    "distance_miles",
)

BASE_CATEGORICAL_COLUMNS = (
    "origin_airport",
    "destination_airport",
    "route",
    "carrier_identifier",
)

BASE_FEATURE_COLUMNS = BASE_NUMERIC_COLUMNS + BASE_CATEGORICAL_COLUMNS

# These must never be present in a DataFrame handed to this selector.
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "cancelled_target",
        "completed_flight",
        "arrival_delay_minutes",
        "severe_delay_60",
        "severe_delay_90",
        "severe_delay_120",
        "severe_delay_180",
        "ARR_DELAY",
        "DEP_DELAY",
        "CANCELLATION_CODE",
        "DELAY_DUE_CARRIER",
        "DELAY_DUE_WEATHER",
        "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY",
        "DELAY_DUE_LATE_AIRCRAFT",
    }
)


def select_base_features(predictors: pd.DataFrame) -> pd.DataFrame:
    """Return only approved base predictors in a fixed column order.

    Pass the Phase 3 predictor table, not a predictors-plus-labels join.
    IDs, timestamps, outcomes, and later feature families are excluded.
    """
    forbidden = FORBIDDEN_OUTCOME_COLUMNS.intersection(predictors.columns)
    if forbidden:
        raise ValueError(
            f"Outcome columns found in predictor input: {sorted(forbidden)}"
        )

    missing = set(BASE_FEATURE_COLUMNS).difference(predictors.columns)
    if missing:
        raise ValueError(f"Required base features missing: {sorted(missing)}")

    selected = predictors.loc[:, BASE_FEATURE_COLUMNS].copy()

    for column in BASE_NUMERIC_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors="raise")

        non_missing = selected[column].dropna().to_numpy(dtype=np.float64)
        if not np.isfinite(non_missing).all():
            raise ValueError(f"Non-finite values found in {column}.")

    # Normalise missing categories before sklearn sees them. The missing
    # marker is not derived from validation or test data.
    for column in BASE_CATEGORICAL_COLUMNS:
        selected[column] = selected[column].fillna("__MISSING__").astype(str)

    return selected


def build_base_preprocessor() -> ColumnTransformer:
    """Build an unfitted transformer; fit it on development rows only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(BASE_NUMERIC_COLUMNS)),
            (
                "categorical",
                categorical_pipeline,
                list(BASE_CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
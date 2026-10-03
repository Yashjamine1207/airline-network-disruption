"""Corrected final evaluation on 2023 only.

Train all models on 2019–2022 (development + validation).
Evaluate once on 2023. No fitting, tuning, or calibration uses 2023 labels.
"""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_FEATURE_COLUMNS,
    build_base_preprocessor,
    select_base_features,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT / "data/processed/model_ready/phase4_base_full_splits.parquet"
)
TRAIN_SPLITS = ("development", "validation")
TEST_SPLIT = "final_test_locked"
RANDOM_SEED = 42

OUTPUT_PATH = ROOT / "reports/tables/phase4_final_test_2023_results.csv"


def load_split(
    connection: duckdb.DuckDBPyConnection,
    splits: tuple[str, ...],
    target: str,
    regression: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load predictors and labels for one or more splits."""
    if not set(splits).issubset({"development", "validation", TEST_SPLIT}):
        raise ValueError(f"Disallowed splits: {splits}")

    placeholders = ", ".join("?" for _ in splits)

    if regression:
        feature_sql = ", ".join(
            f'"{column}"' for column in BASE_FEATURE_COLUMNS
        )

        frame = connection.execute(
            f"""
            SELECT
                source_row_number,
                {feature_sql},
                completed_flight,
                arrival_delay_minutes
            FROM read_parquet(?)
            WHERE temporal_split IN ({placeholders})
              AND completed_flight = 1
              AND arrival_delay_minutes IS NOT NULL
            ORDER BY source_row_number
            """,
            [str(INPUT_PATH), *splits],
        ).fetch_df()

        if frame.empty:
            raise ValueError(f"No regression rows for splits {splits}.")

        if frame["source_row_number"].duplicated().any():
            raise ValueError(f"Duplicate IDs in regression for {splits}.")

        y = frame["arrival_delay_minutes"].to_numpy(dtype=np.float64)

        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite targets in {splits} regression.")

        X = select_base_features(frame.loc[:, BASE_FEATURE_COLUMNS])

    else:
        feature_sql = ", ".join(
            f'"{column}"' for column in BASE_FEATURE_COLUMNS
        )

        frame = connection.execute(
            f"""
            SELECT
                source_row_number,
                {feature_sql},
                completed_flight,
                cancelled_target,
                severe_delay_120
            FROM read_parquet(?)
            WHERE temporal_split IN ({placeholders})
            ORDER BY source_row_number
            """,
            [str(INPUT_PATH), *splits],
        ).fetch_df()

        if frame.empty:
            raise ValueError(f"No rows for splits {splits}.")

        if frame["source_row_number"].duplicated().any():
            raise ValueError(f"Duplicate IDs for {splits}.")

        if target == "cancelled_target":
            eligible = frame["cancelled_target"].notna()
            y = frame.loc[eligible, "cancelled_target"].to_numpy(dtype=np.int8)
            X = select_base_features(
                frame.loc[eligible, BASE_FEATURE_COLUMNS]
            )

        elif target == "severe_delay_120":
            eligible = (
                frame["completed_flight"].eq(1)
                & frame["severe_delay_120"].notna()
            )
            y = frame.loc[eligible, "severe_delay_120"].to_numpy(dtype=np.int8)
            X = select_base_features(
                frame.loc[eligible, BASE_FEATURE_COLUMNS]
            )

        else:
            raise ValueError(f"Unknown classification target: {target}")

    if len(X) != len(y):
        raise ValueError(f"X/y mismatch for {splits} and {target}.")

    return X, y


def load_split_lightgbm(
    connection: duckdb.DuckDBPyConnection,
    splits: tuple[str, ...],
    target: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load categorical + numeric features for LightGBM without one-hot."""
    if not set(splits).issubset({"development", "validation", TEST_SPLIT}):
        raise ValueError(f"Disallowed splits: {splits}")

    placeholders = ", ".join("?" for _ in splits)

    feature_columns = [
        '"source_row_number"',
        *[f'"{name}"' for name in BASE_CATEGORICAL_COLUMNS],
        '"completed_flight"',
        '"severe_delay_120"',
    ]

    frame = connection.execute(
        f"""
        SELECT {", ".join(feature_columns)}
        FROM read_parquet(?)
        WHERE temporal_split IN ({placeholders})
        ORDER BY source_row_number
        """,
        [str(INPUT_PATH), *splits],
    ).fetch_df()

    if frame.empty:
        raise ValueError(f"No rows for splits {splits}.")

    if frame["source_row_number"].duplicated().any():
        raise ValueError(f"Duplicate IDs for {splits}.")

    eligible = (
        frame["completed_flight"].eq(1) & frame["severe_delay_120"].notna()
    )
    cohort = frame.loc[eligible]

    if not cohort["severe_delay_120"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels for {splits}.")

    y = cohort["severe_delay_120"].to_numpy(dtype=np.int8)
    X = cohort.loc[:, BASE_CATEGORICAL_COLUMNS].copy()

    if len(X) != len(y):
        raise ValueError(f"X/y mismatch for {splits}.")

    return X, y


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        # Confirm 2023 rows exist.
        counts = dict(
            connection.execute(
                """
                SELECT temporal_split, COUNT(*)
                FROM read_parquet(?)
                GROUP BY temporal_split
                """,
                [str(INPUT_PATH)],
            ).fetchall()
        )

        if TEST_SPLIT not in counts or counts[TEST_SPLIT] == 0:
            raise ValueError("No 2023 final-test rows found in the Parquet.")

        print("2023 final-test rows present:", counts[TEST_SPLIT])

        # ---------- Cancellation: Logistic Regression ----------
        print("Loading cancellation data...", flush=True)
        X_train_c, y_train_c = load_split(
            connection, TRAIN_SPLITS, "cancelled_target"
        )
        X_test_c, y_test_c = load_split(
            connection, (TEST_SPLIT,), "cancelled_target"
        )

        print("Training cancellation Logistic Regression...", flush=True)
        clf_c = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            class_weight=None,
            max_iter=150,
            tol=1e-3,
            random_state=RANDOM_SEED,
        )
        pipe_c = Pipeline(
            steps=[
                ("preprocessor", build_base_preprocessor()),
                ("classifier", clf_c),
            ]
        )
        pipe_c.fit(X_train_c, y_train_c)

        prob_c = pipe_c.predict_proba(X_test_c)[:, 1]

        # ---------- Severe delay: LightGBM + isotonic ----------
        print("Loading severe-delay data...", flush=True)
        X_train_s, y_train_s = load_split_lightgbm(
            connection, TRAIN_SPLITS, "severe_delay_120"
        )
        X_test_s, y_test_s = load_split_lightgbm(
            connection, (TEST_SPLIT,), "severe_delay_120"
        )

        # Learn categories from training only.
        for col in BASE_CATEGORICAL_COLUMNS:
            X_train_s[col] = X_train_s[col].astype("category")
            X_test_s[col] = pd.Categorical(
                X_test_s[col],
                categories=X_train_s[col].cat.categories,
            )

        print("Training severe-delay LightGBM...", flush=True)
        lgb_params = {
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

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train_s,
            y_train_s,
            categorical_feature=list(BASE_CATEGORICAL_COLUMNS),
        )

        # Fit isotonic calibrator on TRAINING predictions, not on 2023.
        print("Calibrating on training predictions...", flush=True)
        prob_train_uncal = lgb_model.predict_proba(X_train_s)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_train_uncal, y_train_s)

        prob_test_uncal = lgb_model.predict_proba(X_test_s)[:, 1]
        prob_s = iso.predict(prob_test_uncal)

        # ---------- Arrival delay: Ridge ----------
        print("Loading arrival-delay data...", flush=True)
        X_train_r, y_train_r = load_split(
            connection, TRAIN_SPLITS, "arrival_delay_minutes", regression=True
        )
        X_test_r, y_test_r = load_split(
            connection, (TEST_SPLIT,), "arrival_delay_minutes", regression=True
        )

        print("Training arrival-delay Ridge...", flush=True)
        ridge = Ridge(alpha=10.0, solver="lsqr", max_iter=1000, tol=1e-3)
        pipe_r = Pipeline(
            steps=[
                ("preprocessor", build_base_preprocessor()),
                ("regressor", ridge),
            ]
        )
        pipe_r.fit(X_train_r, y_train_r)

        pred_r = pipe_r.predict(X_test_r)

        # ---------- Assemble results ----------
        results = []

        # Cancellation
        results.append(
            {
                "target": "cancelled_target",
                "model": "logistic_regression_unweighted",
                "feature_set": "base",
                "split": "final_test_2023",
                "test_rows": len(y_test_c),
                "test_event_rate": float(y_test_c.mean()),
                "average_precision": float(
                    average_precision_score(y_test_c, prob_c)
                ),
                "roc_auc": float(roc_auc_score(y_test_c, prob_c)),
                "brier_score": float(brier_score_loss(y_test_c, prob_c)),
                "mae_minutes": np.nan,
                "rmse_minutes": np.nan,
                "input_sha256": "",
            }
        )

        # Severe delay
        results.append(
            {
                "target": "severe_delay_120",
                "model": "lightgbm_isotonic",
                "feature_set": "base",
                "split": "final_test_2023",
                "test_rows": len(y_test_s),
                "test_event_rate": float(y_test_s.mean()),
                "average_precision": float(
                    average_precision_score(y_test_s, prob_s)
                ),
                "roc_auc": float(roc_auc_score(y_test_s, prob_s)),
                "brier_score": float(brier_score_loss(y_test_s, prob_s)),
                "mae_minutes": np.nan,
                "rmse_minutes": np.nan,
                "input_sha256": "",
            }
        )

        # Arrival delay
        severe_mask = y_test_r >= 120
        severe_count = int(severe_mask.sum())

        results.append(
            {
                "target": "arrival_delay_minutes",
                "model": "ridge_regression",
                "feature_set": "base",
                "split": "final_test_2023",
                "test_rows": len(y_test_r),
                "test_event_rate": np.nan,
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "brier_score": np.nan,
                "mae_minutes": float(mean_absolute_error(y_test_r, pred_r)),
                "rmse_minutes": float(
                    np.sqrt(mean_squared_error(y_test_r, pred_r))
                ),
                "severe_test_rows": severe_count,
                "severe_subset_mae_minutes": (
                    float(
                        mean_absolute_error(
                            y_test_r[severe_mask],
                            pred_r[severe_mask],
                        )
                    )
                    if severe_count
                    else np.nan
                ),
                "input_sha256": "",
            }
        )

        table = pd.DataFrame(results)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_PATH, index=False)

        print(f"Saved final-test results: {OUTPUT_PATH}")
        print("No fitting used 2023 labels; 2023 is a pure evaluation split.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
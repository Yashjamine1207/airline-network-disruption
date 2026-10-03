"""Corrected severe-delay evaluation with disjoint calibration.

Model: LightGBM on full base features (categoricals encoded as integer codes).
Training: 2019–2021.
Calibration: 2022 predictions (disjoint from training).
Final test: 2023 only, no fitting on 2023 labels.
"""

from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_FEATURE_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    select_base_features,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT / "data/processed/model_ready/phase4_base_full_splits.parquet"
)
TRAIN_SPLIT = "development"       # 2019–2021
CALIB_SPLIT = "validation"        # 2022
TEST_SPLIT = "final_test_locked"  # 2023
RANDOM_SEED = 42

OUTPUT_PATH = (
    ROOT / "reports/tables/phase4_severe_delay_2023_corrected.csv"
)


def load_severe_split(
    connection: duckdb.DuckDBPyConnection,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load full base features and severe-delay labels for one split."""
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
        raise ValueError(f"No rows for split {split}.")

    if frame["source_row_number"].duplicated().any():
        raise ValueError(f"Duplicate IDs in split {split}.")

    eligible = (
        frame["completed_flight"].eq(1)
        & frame["severe_delay_120"].notna()
    )
    cohort = frame.loc[eligible]

    if not cohort["severe_delay_120"].isin([0, 1]).all():
        raise ValueError(f"Invalid severe-delay labels in {split}.")

    y = cohort["severe_delay_120"].to_numpy(dtype=np.int8)

    # Numeric features as float32.
    X_num = cohort.loc[:, BASE_NUMERIC_COLUMNS].astype("float32")

    # Categorical features as string, then encoded outside.
    X_cat = cohort.loc[:, BASE_CATEGORICAL_COLUMNS].fillna("__MISSING__").astype(str)

    return X_num, X_cat, y


def encode_all_splits(
    X_cat_train: pd.DataFrame,
    X_cat_calib: pd.DataFrame,
    X_cat_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode categoricals using training categories only, as integer codes."""
    encoded_parts_train = []
    encoded_parts_calib = []
    encoded_parts_test = []

    for col in BASE_CATEGORICAL_COLUMNS:
        # Learn categories from training split only.
        categories = pd.Categorical(X_cat_train[col]).categories

        cat_train = pd.Categorical(X_cat_train[col], categories=categories)
        cat_calib = pd.Categorical(X_cat_calib[col], categories=categories)
        cat_test = pd.Categorical(X_cat_test[col], categories=categories)

        encoded_parts_train.append(cat_train.codes.astype("int32"))
        encoded_parts_calib.append(cat_calib.codes.astype("int32"))
        encoded_parts_test.append(cat_test.codes.astype("int32"))

    def make_df(parts):
        return pd.DataFrame(
            dict(zip(BASE_CATEGORICAL_COLUMNS, parts)),
            columns=BASE_CATEGORICAL_COLUMNS,
        )

    return make_df(encoded_parts_train), make_df(encoded_parts_calib), make_df(encoded_parts_test)


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Prepared table not found: {INPUT_PATH}")

    connection = duckdb.connect()
    try:
        print("Loading 2019–2021 training data...", flush=True)
        X_num_train, X_cat_train, y_train = load_severe_split(
            connection, TRAIN_SPLIT
        )

        print("Loading 2022 calibration data...", flush=True)
        X_num_calib, X_cat_calib, y_calib = load_severe_split(
            connection, CALIB_SPLIT
        )

        print("Loading 2023 test data...", flush=True)
        X_num_test, X_cat_test, y_test = load_severe_split(
            connection, TEST_SPLIT
        )

        # Encode categoricals consistently.
        X_cat_train_enc, X_cat_calib_enc, X_cat_test_enc = encode_all_splits(
            X_cat_train, X_cat_calib, X_cat_test
        )

        # Combine numeric + encoded categorical.
        X_train = pd.concat(
            [X_num_train.reset_index(drop=True), X_cat_train_enc],
            axis=1,
        ).astype("float32")

        X_calib = pd.concat(
            [X_num_calib.reset_index(drop=True), X_cat_calib_enc],
            axis=1,
        ).astype("float32")

        X_test = pd.concat(
            [X_num_test.reset_index(drop=True), X_cat_test_enc],
            axis=1,
        ).astype("float32")

        # Train LightGBM on 2019–2021 only.
        print("Training LightGBM on 2019–2021...", flush=True)
        params = {
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

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Calibrate on 2022 predictions (disjoint from training).
        print("Fitting isotonic calibration on 2022 predictions...", flush=True)
        prob_calib_uncal = model.predict_proba(X_calib)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob_calib_uncal, y_calib)

        # Evaluate on 2023 only.
        print("Evaluating on 2023 test data...", flush=True)
        prob_test_uncal = model.predict_proba(X_test)[:, 1]
        prob_test = iso.predict(prob_test_uncal)

        if not np.isfinite(prob_test).all():
            raise ValueError("Non-finite calibrated probabilities on 2023.")

        result = {
            "target": "severe_delay_120",
            "model": "lightgbm_isotonic_disjoint_calibration",
            "feature_set": "base_numeric_categorical_encoded",
            "training_split": "2019_2021",
            "calibration_split": "2022",
            "test_split": "2023_final_test_locked",
            "test_rows": len(y_test),
            "test_event_rate": float(y_test.mean()),
            "average_precision": float(
                average_precision_score(y_test, prob_test)
            ),
            "roc_auc": float(roc_auc_score(y_test, prob_test)),
            "brier_score": float(brier_score_loss(y_test, prob_test)),
            "mean_predicted_probability": float(prob_test.mean()),
            "random_seed": RANDOM_SEED,
            "lightgbm_version": lgb.__version__,
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)

        print(f"Saved corrected severe-delay 2023 result: {OUTPUT_PATH}")
        print("No fitting used 2023 labels; 2023 is a pure evaluation split.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
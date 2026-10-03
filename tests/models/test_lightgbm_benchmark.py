"""Tests for airline_disruption.models.lightgbm_benchmark (Phase 6, Step 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.features.feature_sets import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    HISTORY_COLUMNS,
    NETWORK_COLUMNS,
    columns_for,
)
from airline_disruption.models import lightgbm_benchmark as lb


def _meta(splits: list[str], times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase6_split": splits,
            "prediction_timestamp_utc": pd.to_datetime(times, utc=True),
            "scheduled_departure_utc": pd.to_datetime(times, utc=True) + pd.Timedelta(hours=2),
        }
    )


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
def test_phase6_roles_split_development_at_2021_07_01() -> None:
    meta = _meta(
        ["development", "development", "development", "validation", "final_test_locked"],
        ["2019-03-01 10:00", "2021-06-30 23:59", "2021-07-01 00:00", "2022-05-01 10:00", "2023-03-01 10:00"],
    )
    roles = lb.assign_roles(meta, lb.PROTOCOL_PHASE6).tolist()
    assert roles == [lb.ROLE_FIT, lb.ROLE_FIT, lb.ROLE_EARLY_STOP, lb.ROLE_VALIDATION, lb.ROLE_FINAL_TEST]


def test_phase4_protocol_uses_all_development_as_fit() -> None:
    meta = _meta(["development", "development", "validation"], ["2019-03-01 10:00", "2021-09-01 10:00", "2022-05-01 10:00"])
    assert lb.assign_roles(meta, lb.PROTOCOL_PHASE4).tolist() == [lb.ROLE_FIT, lb.ROLE_FIT, lb.ROLE_VALIDATION]


def test_embargoed_rows_cannot_get_a_role() -> None:
    meta = _meta(["embargoed_boundary"], ["2022-01-01 03:00"])
    with pytest.raises(ValueError, match="unexpected split"):
        lb.assign_roles(meta)


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown protocol"):
        lb.assign_roles(_meta(["development"], ["2019-03-01 10:00"]), "random_split")


def test_chronology_guard_accepts_ordered_roles_and_rejects_overlap() -> None:
    meta = _meta(
        ["development", "development", "validation"],
        ["2019-03-01 10:00", "2021-09-01 10:00", "2022-05-01 10:00"],
    )
    roles = lb.assign_roles(meta)
    lb.check_role_chronology(meta, roles)  # no error

    overlapping = pd.Series([lb.ROLE_FIT, lb.ROLE_EARLY_STOP, lb.ROLE_VALIDATION])
    bad_meta = _meta(["development"] * 3, ["2021-09-01 10:00", "2021-08-01 10:00", "2022-05-01 10:00"])
    with pytest.raises(ValueError, match="fit ends"):
        lb.check_role_chronology(bad_meta, overlapping)

    # Two roles may not touch: a shared boundary instant is an overlap too.
    touching_meta = _meta(["development", "development"], ["2021-06-30 12:00", "2021-06-30 12:00"])
    with pytest.raises(ValueError, match="fit ends"):
        lb.check_role_chronology(touching_meta, pd.Series([lb.ROLE_FIT, lb.ROLE_EARLY_STOP]))


# ---------------------------------------------------------------------------
# Categories and the history mask
# ---------------------------------------------------------------------------
def test_categories_come_from_fit_rows_only() -> None:
    X = pd.DataFrame(
        {
            "origin_airport": pd.Categorical(["ATL", "JFK", "LAX", "ATL"]),
            "distance_miles": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fit_mask = np.array([True, True, False, False])
    columns = lb.apply_fit_categories(X, fit_mask)
    assert columns == ["origin_airport"]
    assert set(X["origin_airport"].cat.categories) == {"ATL", "JFK"}
    assert pd.isna(X.loc[2, "origin_airport"])  # LAX was never seen in the fit rows
    assert X.loc[3, "origin_airport"] == "ATL"


def test_history_is_masked_before_february_2019_only() -> None:
    columns = [*HISTORY_COLUMNS, *NETWORK_COLUMNS]
    X = pd.DataFrame({c: [0.0, 0.0, 5.0] for c in columns})
    X["distance_miles"] = [100.0, 200.0, 300.0]
    meta = pd.DataFrame(
        {
            "scheduled_departure_utc": pd.to_datetime(
                ["2018-12-31 23:15", "2019-01-31 23:59", "2019-02-01 00:00"], utc=True
            )
        }
    )
    masked = lb.mask_unavailable_history(X, meta)
    assert masked == 2
    assert X.loc[:1, columns].isna().all().all()
    assert X.loc[2, columns].tolist() == [5.0] * len(columns)
    assert X["distance_miles"].tolist() == [100.0, 200.0, 300.0]  # other columns untouched


def test_history_mask_does_nothing_without_history_columns() -> None:
    X = pd.DataFrame({"distance_miles": [1.0]})
    meta = pd.DataFrame({"scheduled_departure_utc": pd.to_datetime(["2019-01-05"], utc=True)})
    assert lb.mask_unavailable_history(X, meta) == 0


# ---------------------------------------------------------------------------
# Loading rows
# ---------------------------------------------------------------------------
@pytest.fixture()
def project(tmp_path):
    """A tiny project tree with a cohort table and a predictor table."""
    times = pd.to_datetime(
        [
            "2021-12-31 23:00",  # 1 development, eligible (latest development time, lowest id)
            "2019-03-01 11:00",  # 2 development, cancelled (not eligible)
            "2019-03-01 10:00",  # 3 development, eligible (earliest time)
            "2022-01-01 06:00",  # 4 embargoed
            "2022-06-01 10:00",  # 5 validation, eligible
            "2023-02-01 10:00",  # 6 final test, eligible
        ],
        utc=True,
    )
    splits = ["development", "development", "development", "embargoed_boundary", "validation", "final_test_locked"]
    eligible = [True, False, True, False, True, True]
    labels = pd.array([0, pd.NA, 1, pd.NA, 1, 0], dtype="Int8")  # by row id 1..6
    cohort = pd.DataFrame(
        {
            "source_row_number": [1, 2, 3, 4, 5, 6],
            "prediction_timestamp_utc": times,
            "regime": pd.Categorical(["2019_pre_covid"] * 2 + ["2021_2022_transition"] * 3 + ["2023_partial_final"]),
            "phase6_split": pd.Categorical(splits),
            "eligible_severe_delay": eligible,
            "severe_delay_120": labels,
            # columns that must never reach the model
            "cancelled": [0, 1, 0, 0, 0, 0],
            "arrival_delay_minutes": [5.0, np.nan, 130.0, np.nan, 200.0, 3.0],
        }
    )
    predictors = pd.DataFrame({"source_row_number": [6, 5, 4, 3, 2, 1]})  # deliberately shuffled
    predictors["scheduled_departure_utc"] = predictors["source_row_number"].map(
        dict(zip([1, 2, 3, 4, 5, 6], times + pd.Timedelta(hours=2)))
    )
    for i, column in enumerate(BASE_NUMERIC_COLUMNS):
        predictors[column] = np.float32(i) + predictors["source_row_number"].astype("float32")
    for column in BASE_CATEGORICAL_COLUMNS:
        predictors[column] = pd.Categorical([f"{column[:3]}{n}" for n in predictors["source_row_number"]])
    for column in (*HISTORY_COLUMNS, *NETWORK_COLUMNS):
        predictors[column] = predictors["source_row_number"].astype("float64") * 10  # stored as float64, like the real file
    (tmp_path / "data/processed/phase6").mkdir(parents=True)
    cohort.to_parquet(tmp_path / lb.COHORT_PATH, index=False)
    predictors.to_parquet(tmp_path / lb.PREDICTORS_PATH, index=False)
    return tmp_path


def test_loader_keeps_only_eligible_non_embargoed_rows_in_time_order(project) -> None:
    frame = lb.load_model_frame(project, "base")
    # Row 2 is ineligible, 4 embargoed, 6 final test. Row 3 has the earliest time, so it comes first.
    assert frame.meta["source_row_number"].tolist() == [3, 1, 5]
    assert frame.y.tolist() == [1, 0, 1]
    assert frame.meta["phase6_split"].tolist() == ["development", "development", "validation"]
    assert frame.meta["prediction_timestamp_utc"].is_monotonic_increasing


def test_loader_returns_exactly_the_feature_set_columns(project) -> None:
    frame = lb.load_model_frame(project, "base")
    assert list(frame.X.columns) == list(columns_for("base"))
    forbidden = {"severe_delay_120", "cancelled", "arrival_delay_minutes", "source_row_number", "phase6_split"}
    assert not forbidden & set(frame.X.columns)


def test_loader_aligns_predictors_by_id_not_by_position(project) -> None:
    frame = lb.load_model_frame(project, "base")
    # The predictor value for the first numeric column is its row number (offset 0).
    first_numeric = BASE_NUMERIC_COLUMNS[0]
    assert frame.X[first_numeric].tolist() == [3.0, 1.0, 5.0]


def test_final_test_is_locked_by_default(project) -> None:
    with pytest.raises(ValueError, match="locked"):
        lb.load_model_frame(project, "base", splits=("development", "validation", "final_test_locked"))
    with pytest.raises(ValueError, match="locked"):
        lb.load_model_frame(project, "base", splits=("final_test_locked",))


def test_final_test_loads_only_when_explicitly_allowed(project) -> None:
    frame = lb.load_model_frame(project, "base", splits=("final_test_locked",), allow_final_test=True)
    assert frame.meta["source_row_number"].tolist() == [6]


def test_default_loader_never_returns_final_test_rows(project) -> None:
    frame = lb.load_model_frame(project, "base")
    assert "final_test_locked" not in set(frame.meta["phase6_split"])


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def test_fit_and_predict_learn_a_real_signal() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame(
        {
            "distance_miles": rng.normal(size=n).astype("float32"),
            "origin_airport": pd.Categorical(rng.choice(["ATL", "JFK", "LAX", "ORD"], n)),
        }
    )
    logit = 1.8 * X["distance_miles"] + np.where(X["origin_airport"] == "JFK", 1.0, 0.0) - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    fit_rows, stop_rows, test_rows = slice(0, 3500), slice(3500, 4500), slice(4500, n)
    params = {**lb.LIGHTGBM_PARAMS, "n_estimators": 200, "min_child_samples": 20}

    result = lb.fit_lightgbm(X.iloc[fit_rows], y[fit_rows], X.iloc[stop_rows], y[stop_rows], n_jobs=1, seed=1, params=params)
    scores = lb.predict_scores(result, X.iloc[test_rows])

    assert result.best_iteration >= 1
    assert result.params["random_state"] == 1 and result.params["n_jobs"] == 1
    assert scores.min() >= 0 and scores.max() <= 1
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[test_rows], scores) > 0.75


def test_fit_is_reproducible_for_a_fixed_seed() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(4)
    n = 3000
    X = pd.DataFrame({"distance_miles": rng.normal(size=n).astype("float32")})
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["distance_miles"] - 1.0)))).astype("int8")
    params = {**lb.LIGHTGBM_PARAMS, "n_estimators": 50, "min_child_samples": 20}
    a = lb.fit_lightgbm(X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:], n_jobs=1, seed=3, params=params)
    b = lb.fit_lightgbm(X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:], n_jobs=1, seed=3, params=params)
    assert np.allclose(lb.predict_scores(a, X), lb.predict_scores(b, X))


def test_predictions_use_the_recorded_best_iteration() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(8)
    n = 4000
    X = pd.DataFrame({"distance_miles": rng.normal(size=n).astype("float32")})
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X["distance_miles"] - 1.5)))).astype("int8")
    params = {**lb.LIGHTGBM_PARAMS, "n_estimators": 60, "min_child_samples": 20}
    fit = lb.fit_lightgbm(X.iloc[:3000], y[:3000], X.iloc[3000:], y[3000:], n_jobs=1, seed=2, params=params)

    early = lb.FitResult(model=fit.model, best_iteration=3, seconds=0.0, params=fit.params)
    late = lb.FitResult(model=fit.model, best_iteration=40, seconds=0.0, params=fit.params)
    assert np.allclose(lb.predict_scores(early, X), fit.model.predict_proba(X, num_iteration=3)[:, 1])
    assert np.allclose(lb.predict_scores(late, X), fit.model.predict_proba(X, num_iteration=40)[:, 1])
    assert not np.allclose(lb.predict_scores(early, X), lb.predict_scores(late, X))


@pytest.mark.parametrize("empty_side", ["fit", "stop"])
def test_fit_rejects_a_one_class_set_with_a_readable_message(empty_side: str) -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"distance_miles": rng.normal(size=400).astype("float32")})
    mixed = (np.arange(400) % 2).astype("int8")
    zeros = np.zeros(400, dtype="int8")
    y_fit, y_stop = (zeros, mixed) if empty_side == "fit" else (mixed, zeros)
    label = "fit" if empty_side == "fit" else "early-stopping"
    with pytest.raises(ValueError, match=f"The {label} rows contain only one class"):
        lb.fit_lightgbm(X, y_fit, X, y_stop, n_jobs=1)


def test_loader_stores_wide_floats_as_float32_without_changing_values(project) -> None:
    frame = lb.load_model_frame(project, "base_history_network")
    for column in (*HISTORY_COLUMNS, *NETWORK_COLUMNS):
        assert frame.X[column].dtype == np.float32
    # Row ids [3, 1, 5] in time order, stored value = id * 10.
    assert frame.X[HISTORY_COLUMNS[0]].tolist() == [30.0, 10.0, 50.0]

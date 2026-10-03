"""Tests for SHAP helpers and the error breakdown (Phase 7A, Step 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.explain import errors, shap_tools
from airline_disruption.features.feature_sets import BASE_NO_YEAR_COLUMNS


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def small_model(n: int = 6_000, seed: int = 0):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "signal": rng.normal(size=n),
        "noise": rng.normal(size=n),
        "airport": pd.Categorical(rng.choice(["AAA", "BBB", "CCC", "DDD"], n)),
    })
    logit = 1.5 * X["signal"] + np.where(X["airport"] == "BBB", 1.0, 0.0) - 2.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = lgb.LGBMClassifier(n_estimators=40, num_leaves=8, learning_rate=0.1, verbose=-1, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model, X, y


def test_shap_values_add_up_to_the_model_output_including_a_categorical_feature() -> None:
    model, X, _ = small_model()
    contrib, base = shap_tools.contributions(model, X)
    gap = shap_tools.check_additivity(model, X, contrib, base)
    assert gap < 1e-6 and list(contrib.columns) == list(X.columns)


def test_additivity_check_fails_when_the_values_are_wrong() -> None:
    model, X, _ = small_model()
    contrib, base = shap_tools.contributions(model, X)
    contrib["signal"] += 0.01
    with pytest.raises(ValueError, match="do not add up"):
        shap_tools.check_additivity(model, X, contrib, base)


def test_the_informative_feature_has_the_largest_mean_absolute_shap() -> None:
    model, X, _ = small_model()
    contrib, _ = shap_tools.contributions(model, X)
    table = shap_tools.global_importance(contrib)
    assert table["feature"].iloc[0] == "signal" and table["feature"].iloc[-1] == "noise"
    assert table["share_of_total"].sum() == pytest.approx(1.0)
    assert (table["mean_abs_shap"].diff().dropna() <= 0).all()  # sorted, largest first


def test_the_numeric_effect_leans_the_right_way_and_the_high_risk_level_is_found() -> None:
    model, X, _ = small_model()
    contrib, _ = shap_tools.contributions(model, X)
    effect = shap_tools.numeric_effect(X["signal"], contrib["signal"], n_bins=5)
    assert effect["mean_shap"].is_monotonic_increasing and effect["rows"].sum() == len(X)
    levels = shap_tools.categorical_effect(X["airport"], contrib["airport"], min_rows=100, top=1)
    assert levels[levels["direction"] == "highest"]["level"].iloc[0] == "BBB"


def test_categorical_effect_reports_unseen_levels_and_skips_small_levels() -> None:
    values = pd.Series(pd.Categorical(["A"] * 600 + ["B"] * 600 + [None] * 600 + ["C"] * 10), name="airport")
    shap = pd.Series(np.r_[np.full(600, 0.5), np.full(600, -0.5), np.full(600, 0.1), np.full(10, 9.0)])
    out = shap_tools.categorical_effect(values, shap, min_rows=500, top=5)
    assert set(out["level"]) == {"A", "B", "unseen"}  # level C has only 10 rows


def test_every_base_no_year_feature_has_a_named_family_and_top_contributions_are_ordered() -> None:
    assert all(shap_tools.family_of(c) != "other" for c in BASE_NO_YEAR_COLUMNS)
    contrib = pd.Series({"a": 0.1, "b": -0.9, "c": 0.4})
    values = pd.Series({"a": 1, "b": "JFK", "c": 3.5})
    top = shap_tools.top_contributions(contrib, values, k=2)
    assert [t[0] for t in top] == ["b", "c"] and top[0][1] == "JFK" and top[0][2] == -0.9


def test_family_importance_sums_the_features_of_each_family() -> None:
    imp = pd.DataFrame({"feature": ["origin_airport", "destination_airport", "route"], "mean_abs_shap": [0.2, 0.1, 0.3]})
    imp["family"] = imp["feature"].map(shap_tools.family_of)
    fam = shap_tools.family_importance(imp).set_index("family")
    assert fam.loc["airport", "mean_abs_shap"] == pytest.approx(0.3) and fam.loc["route", "mean_abs_shap"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Error breakdown
# ---------------------------------------------------------------------------
def test_top_fraction_cutoffs_pick_the_score_at_the_edge_of_the_top_group() -> None:
    s = np.arange(1, 101) / 100.0
    cut = errors.top_fraction_cutoffs(s, (0.05, 0.10))
    assert cut[0.05] == pytest.approx(0.96) and cut[0.10] == pytest.approx(0.91)


def make_frame():
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.6, 0.5, 0.4, 0.3])
    return pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "score": score, "probability": score / 2,
                         "label": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]})


def test_group_metrics_match_a_hand_calculation_with_a_global_cutoff() -> None:
    frame = make_frame()
    cut = errors.top_fraction_cutoffs(frame["score"], (0.3,))  # top 3 rows overall: 0.9, 0.8, 0.7 -> cutoff 0.7
    table = errors.group_metrics(frame, "test", cut, "g", min_events=1).set_index("group")
    a = table.loc["a"]
    assert (a["rows"], a["events"]) == (5, 3) and a["prevalence"] == pytest.approx(0.6)
    assert a["alert_rate_top30pct"] == pytest.approx(3 / 5) and a["precision_top30pct"] == pytest.approx(2 / 3)
    assert a["recall_top30pct"] == pytest.approx(2 / 3)
    b = table.loc["b"]
    assert b["alert_rate_top30pct"] == 0.0 and b["recall_top30pct"] == 0.0 and np.isnan(b["precision_top30pct"])
    assert a["mean_probability"] == pytest.approx(np.mean([0.45, 0.4, 0.35, 0.1, 0.05]))
    assert a["probability_to_observed"] == pytest.approx(a["mean_probability"] / 0.6)


def test_small_groups_get_no_pr_auc_and_are_flagged() -> None:
    frame = make_frame()
    table = errors.group_metrics(frame, "test", {0.3: 0.7}, "g", min_events=30)
    assert table["pr_auc"].isna().all() and table["small_sample"].all()


def test_pr_auc_is_reported_for_groups_with_enough_events() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    s = rng.random(n)
    frame = pd.DataFrame({"g": rng.choice(["x", "y"], n), "score": s, "probability": s, "label": (rng.random(n) < s * 0.3).astype(int)})
    table = errors.group_metrics(frame, "test", errors.top_fraction_cutoffs(s, (0.1,)), "g")
    assert (~table["small_sample"]).all() and table["pr_auc"].between(0, 1).all() and (table["lift"] > 0).all()


def test_unseen_group_values_are_labelled_unseen() -> None:
    frame = make_frame()
    frame["g"] = pd.Categorical(["a", "a", "a", None, None, "b", "b", "b", "b", "b"])
    table = errors.group_metrics(frame, "test", {0.3: 0.7}, "g", min_events=1)
    assert "unseen" in set(table["group"])


def test_severity_table_shows_the_share_of_each_delay_band_caught() -> None:
    frame = pd.DataFrame({
        "score": [0.9, 0.8, 0.2, 0.1, 0.05, 0.3],
        "label": [1, 1, 1, 1, 1, 0],
        "arrival_delay_minutes": [130, 200, 150, 400, 700, 10],
    })
    table = errors.severity_table(frame, {0.1: 0.8}).set_index("group")
    assert table["events"].sum() == 5 and table["share_of_events"].sum() == pytest.approx(1.0)
    assert table.loc["[120.0, 180.0)", "recall_top10pct"] == pytest.approx(0.5)  # 130 caught (0.9), 150 missed (0.2)
    assert table.loc["[180.0, 300.0)", "recall_top10pct"] == pytest.approx(1.0)
    assert table.loc["[600.0, inf)", "recall_top10pct"] == 0.0


def test_severity_table_and_group_metrics_stack_into_one_clean_boolean_small_sample_column() -> None:
    """The real run crashed here: the severity table had no small_sample column, so stacking made it NaN."""
    rng = np.random.default_rng(3)
    n = 4000
    score = rng.random(n)
    label = (rng.random(n) < 0.05 + 0.1 * score).astype(int)
    delay = np.where(label == 1, 120 + rng.exponential(120, n), rng.integers(-10, 120, n))
    frame = pd.DataFrame({"label": label, "score": score, "probability": score, "arrival_delay_minutes": delay, "g": "a"})
    cutoffs = errors.top_fraction_cutoffs(score, (0.01, 0.10))
    groups = errors.group_metrics(frame, "overall", cutoffs, "g")
    severity = errors.severity_table(frame, cutoffs)
    assert len(severity) > 1 and severity["small_sample"].dtype == bool
    stacked = pd.concat([groups, severity], ignore_index=True)
    assert stacked["small_sample"].dtype == bool  # stays boolean, so ~stacked["small_sample"] works
    assert (~stacked["small_sample"]).sum() >= 1
    tiny = frame.iloc[:200].copy()
    tiny.loc[tiny.index[:5], ["label", "arrival_delay_minutes"]] = [1, 400]
    assert errors.severity_table(tiny, cutoffs)["small_sample"].all()  # fewer than 30 events in every band


def test_half_percent_capacity_gets_its_own_column_name() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    score = rng.random(n)
    frame = pd.DataFrame({"label": (rng.random(n) < 0.05).astype(int), "score": score, "probability": score, "g": "a"})
    cutoffs = errors.top_fraction_cutoffs(score, (0.005, 0.01, 0.05, 0.10))
    table = errors.group_metrics(frame, "overall", cutoffs, "g")
    for name in ("top0.5pct", "top1pct", "top5pct", "top10pct"):
        assert f"alert_rate_{name}" in table.columns and f"precision_{name}" in table.columns and f"recall_{name}" in table.columns
    assert table.loc[0, "alert_rate_top0.5pct"] < table.loc[0, "alert_rate_top1pct"] < table.loc[0, "alert_rate_top5pct"]
    severity = errors.severity_table(frame.assign(arrival_delay_minutes=np.where(frame["label"] == 1, 150.0, 10.0)), cutoffs)
    assert "recall_top0.5pct" in severity.columns and "recall_top10pct" in severity.columns

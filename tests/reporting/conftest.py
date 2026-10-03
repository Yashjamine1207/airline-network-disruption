"""A small synthetic project folder with every table the report builder reads.

The numbers are made up and simple on purpose, so a test can say exactly what a report must contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

LOCK_HASH = "ab" * 32
SEVERE, CANCEL = "severe_delay_120", "cancelled"
CAPACITIES = ("0.5%", "1%", "5%", "10%")
FRACTIONS = {"0.5%": 0.005, "1%": 0.01, "5%": 0.05, "10%": 0.10}


def write_csv(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def capacity_rows(periods: dict[str, tuple[int, int]], lift_whole: float = 2.0, lift_day: float = 1.8) -> pd.DataFrame:
    """periods maps a period name to (flights, events)."""
    rows = []
    for period, (flights, events) in periods.items():
        prevalence = events / flights
        for scheme, lift in (("whole_year", lift_whole), ("by_month", lift_day), ("by_day", lift_day)):
            for cap in CAPACITIES:
                reviewed = max(1, int(round(FRACTIONS[cap] * flights)))
                precision = min(0.9, prevalence * lift)
                captured = int(round(reviewed * precision))
                rows.append({
                    "target": "t", "period": period, "eval_year": 2023 if "2023" in period else 2022, "scheme": scheme, "capacity": cap, "fraction": FRACTIONS[cap],
                    "flights": flights, "events": events, "prevalence": prevalence, "reviewed": reviewed, "events_captured": captured,
                    "false_alerts": reviewed - captured, "events_missed": events - captured, "precision": precision, "recall": captured / events,
                    "lift": lift, "break_even_ratio": (reviewed - captured) / max(captured, 1), "precision_low": precision * 0.9, "precision_high": precision * 1.1,
                    "recall_low": 0.0, "recall_high": 1.0, "n_days": 100, "n_boot": 20,
                })
    return pd.DataFrame(rows)


def group_table(dimension: str, specs: list[tuple]) -> pd.DataFrame:
    """specs: (group, rows, events, alert_rate_top1pct, precision_top1pct, recall_top10pct, alert_rate_top10pct, pr_auc)."""
    rows = []
    for group, n, events, a1, p1, r10, a10, pr in specs:
        rows.append({"target": "t", "dimension": dimension, "group": group, "rows": n, "share_of_rows": 0.0, "events": events, "prevalence": events / n,
                     "alert_rate_top1pct": a1, "precision_top1pct": p1, "recall_top1pct": 0.01, "alert_rate_top10pct": a10, "precision_top10pct": 0.05,
                     "recall_top10pct": r10, "pr_auc": pr, "lift": 1.5, "small_sample": False})
    return pd.DataFrame(rows)


def final_by_group(skew_months: bool = True) -> pd.DataFrame:
    months = [("2023-01", 10000, 200, 0.001, 0.02, 0.05, 0.02, 0.03), ("2023-02", 10000, 200, 0.001, 0.02, 0.05, 0.02, 0.03),
              ("2023-03", 10000, 300, 0.028 if skew_months else 0.01, 0.10, 0.40, 0.40 if skew_months else 0.10, 0.06),
              ("2023-04", 10000, 200, 0.0005, 0.02, 0.05, 0.02, 0.03)]
    hours = [("00-05", 3000, 50, 0.0, float("nan"), 0.01, 0.0, 0.02), ("06-09", 7000, 100, 0.0, float("nan"), 0.01, 0.01, 0.02),
             ("10-13", 6000, 120, 0.001, 0.05, 0.05, 0.03, 0.03), ("14-17", 7000, 200, 0.015, 0.10, 0.30, 0.20, 0.05), ("18-23", 7000, 230, 0.02, 0.11, 0.33, 0.30, 0.06)]
    carriers = [("C1", 15000, 200, 0.0, float("nan"), 0.02, 0.0, 0.02), ("C2", 10000, 200, 0.01, 0.1, 0.3, 0.2, 0.05), ("C3", 5000, 300, 0.02, 0.1, 0.4, 0.3, 0.06)]
    airports = [("AAA", 4000, 100, 0.10, 0.1, 0.5, 0.3, 0.05), ("BBB", 4000, 100, 0.0, float("nan"), 0.02, 0.0, 0.03), ("CCC", 4000, 100, 0.01, 0.05, 0.2, 0.1, 0.04),
                ("DDD", 4000, 100, 0.0, float("nan"), 0.03, 0.0, 0.03), ("EEE", 4000, 100, 0.02, 0.06, 0.25, 0.1, 0.04), ("FFF", 4000, 100, 0.03, 0.06, 0.28, 0.1, 0.04)]
    volume = [("1-99", 20000, 500, 0.01, 0.1, 0.2, 0.1, 0.04), ("unseen route (0 fit-period flights)", 10000, 200, 0.01, 0.1, 0.1, 0.1, 0.04)]
    weekday = [("0", 15000, 350, 0.01, 0.1, 0.2, 0.1, 0.04), ("1", 15000, 350, 0.01, 0.1, 0.2, 0.1, 0.04)]
    missing = [("complete", 28000, 650, 0.01, 0.1, 0.2, 0.1, 0.04), ("unseen category", 2000, 50, 0.001, 0.2, 0.2, 0.1, 0.04)]
    deciles = [(f"decile {i:02d}" + (" (lowest)" if i == 1 else " (highest)" if i == 10 else ""), 3000, 20 + 8 * i, 0.0 if i < 10 else 0.1, float("nan") if i < 10 else 0.1,
                0.0 if i < 10 else 1.0, 0.0 if i < 10 else 1.0, 0.01 * i) for i in range(1, 11)]
    frames = [group_table("month_2023", months), group_table("scheduled_departure_local_hours", hours), group_table("carrier", carriers),
              group_table("origin_airport_top25", airports), group_table("route_volume_in_fit_period", volume), group_table("day_of_week", weekday),
              group_table("missing_or_unseen_values", missing), group_table("score_decile", deciles)]
    return pd.concat(frames, ignore_index=True)


def claims(c1=True, c2=True, c3=True) -> dict:
    return {
        "C1_skill_over_chance": {"statement": "lift rule", "lift": 1.8, "lift_low": 1.7, "supported": c1},
        "C2_beats_history": {"statement": "history rule", "baseline": "baseline_carrier_rate", "difference": 0.015 if c2 else -0.003,
                             "difference_low": 0.012 if c2 else -0.005, "difference_high": 0.019 if c2 else -0.001, "supported": c2},
        "C3_ranking_not_only_seasonal": {"statement": "day rule", "by_day_precision_at_10pct": 0.07, "by_day_precision_at_10pct_low": 0.06, "event_rate_2023": 0.03, "supported": c3},
    }


def calibration_rows(ratio22: float, ratio23: float, prev22: float, prev23: float) -> pd.DataFrame:
    rows = []
    for period, prev, ratio in (("validation_2022", prev22, ratio22), ("final_2023", prev23, ratio23)):
        for variant, r in (("uncalibrated", ratio * 0.8), ("platt fitted on early_stop (locked primary)", ratio), ("platt fitted on validation_2022 (diagnostic, not the locked choice)", ratio * 1.05)):
            rows.append({"target": "t", "period": period, "variant": variant, "rows": 100000, "events": int(prev * 100000), "prevalence": prev, "mean_predicted": prev * r,
                         "predicted_to_observed": r, "brier": 0.025, "brier_skill": 0.01, "calibration_slope": 0.97, "calibration_in_the_large": 0.1, "ece": 0.005,
                         "pr_auc": 0.05 if period == "final_2023" else 0.045, "roc_auc": 0.65, "distinct_scores": 1000})
    return pd.DataFrame(rows)


def build_root(root: Path, *, ratio22=0.93, ratio23=0.83, prev_fit=0.024, prev22=0.0267, prev23=0.0319, severe_claims=(True, True, True), cancel_claims=(False, False, True),
               smoke=False, skew_months=True, ten_pct_alert_2023=0.128) -> Path:
    """Write every table. Returns the root."""
    # ---- lock, marker, decision, calibration choice
    lock = {"lock_version": 1, "smoke": smoke, "feature_set": "base_no_year", "feature_version": "phase6_features_v1", "seed": 42,
            "prediction_time": "scheduled departure UTC minus 2 hours",
            "params": {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50, "reg_lambda": 50.0},
            "files": ["data/processed/phase6/phase6_cohort_v1.parquet", "configs/lightgbm_benchmark_selected.json"],
            "not_in_this_test": ["regression (MAE): no locked regression model is part of this lock", "airport recovery episodes", "sequence models"],
            "periods": {
        "fit": "2019-01 to 2021-06", "early_stop": "2021-07 to 2021-12", "validation": "2022", "final_test": "2023-01-01 to 2023-08-31"},
        "model_policy": {"refit_on_2022_before_final_test": False, "reason": "The validated model is the tested model."},
        "targets": {SEVERE: {"label": "Severe delay (arrival delay of 120 minutes or more)", "best_iteration": 53, "n_features": 12, "early_stop_rows": 337156,
                             "validation_rows": 667616, "validation_events": 17820, "fit_rows": 1454352, "calibrator_primary": {"method": "platt", "fit_rows": 337156, "fit_events": 8154, "intercept": 0.509, "slope": 1.078,
                                                                         "fit_event_rate": prev_fit, "fit_period": "2021-07 to 2021-12"},
                             "calibrator_diagnostic": {"method": "platt", "fit_rows": 667616, "fit_events": 17820, "intercept": 0.229, "slope": 0.983}},
                    CANCEL: {"fit_rows": 1504223}},
        "rejected_challengers": {"lstm": "did not beat the MLP control (DEC-020)", "transformer": "overfits (DEC-022)"}, "lock_sha256": LOCK_HASH}
    lock_name = "reports/final/final_protocol_lock_smoke.json" if smoke else "reports/final/final_protocol_lock.json"
    write_json(root, lock_name, lock)
    write_json(root, "reports/final/final_test_consumed.json", {"lock_sha256": LOCK_HASH, "first_run_utc": "2026-10-01T09:00:00+00:00"})
    write_json(root, "configs/calibration_selected.json", {"methods_compared": ["uncalibrated", "platt", "isotonic"], "rule_1": {"chosen_method": "platt", "reason": "Simpler."},
                                                           "rule_2": {"final_test_fit_source": "same_as_here", "reason": "Not clearly better."}})
    boot = {"primary_vs_uncalibrated": {"difference": -1.15e-4, "ci_low": -1.5e-4, "ci_high": -8e-5}, "diagnostic_vs_primary": {"difference": -1.6e-5, "ci_low": -2.5e-5, "ci_high": -8e-6}}
    decision = {"lock_sha256": LOCK_HASH, "final_test_used": True, "limits": ["a single period (Jan to Aug 2023): one regime"],
                "hardware": {"python": "3.13.5", "cpu_logical_cores": 16}, "packages": {"lightgbm": "4.7.0", "pandas": "2.2.3"}, "targets": {
        SEVERE: {"rows": 454413, "events": 14484, "event_rate": prev23, "trees": 53, "claims": claims(*severe_claims), "calibration_bootstrap": boot, "stray_rows_on_2023_09_01_utc": 175},
        CANCEL: {"rows": 463484, "events": 7809, "event_rate": 0.0169, "trees": 59, "claims": claims(*cancel_claims), "stray_rows_on_2023_09_01_utc": 178}}}
    write_json(root, "reports/tables/phase7b_final_decision.json", decision)

    # ---- final tables, severe delay
    def summary(target: str, prev22_: float, prev23_: float, lgb22: float, lgb23: float, base22: float, base23: float) -> pd.DataFrame:
        rows = []
        for period, prev, lgb, base in (("validation_2022", prev22_, lgb22, base22), ("final_2023", prev23_, lgb23, base23)):
            for scorer, pr in (("lightgbm_locked", lgb), ("baseline_prevalence", prev), ("baseline_carrier_rate", base)):
                rows.append({"target": target, "period": period, "rows": 100000, "events": int(prev * 100000), "prevalence": prev, "scorer": scorer, "pr_auc": pr,
                             "pr_auc_low": pr * 0.9, "pr_auc_high": pr * 1.1, "lift": pr / prev, "lift_low": 1.0, "lift_high": 2.0, "difference_vs_reference": 0.0,
                             "difference_low": 0.0, "difference_high": 0.0, "share_better": 1.0, "roc_auc": 0.65})
        return pd.DataFrame(rows)

    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_summary.csv", summary(SEVERE, prev22, prev23, 0.048, 0.059, 0.037, 0.043))
    caps = capacity_rows({"validation_2022": (667616, int(prev22 * 667616)), "final_2023": (454413, int(prev23 * 454413))})
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_capacity.csv", caps)
    util = []
    for r in caps.itertuples():
        for ratio in (1, 10, 20):
            net = ratio * r.events_captured - r.false_alerts
            util.append({"target": SEVERE, "period": r.period, "eval_year": r.eval_year, "scheme": r.scheme, "capacity": r.capacity, "fraction": r.fraction,
                         "miss_to_false_alert_ratio": ratio, "net_reduction_units": net, "break_even_ratio": r.break_even_ratio, "pays_off_at_this_ratio": bool(net > 0)})
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_utility.csv", pd.DataFrame(util))
    ops = []
    for period, alert10 in (("validation_2022", 0.10), ("final_2023", ten_pct_alert_2023)):
        for name, alert in (("capacity_0.5pct", 0.005), ("capacity_1pct", 0.01), ("capacity_5pct", 0.05), ("capacity_10pct", alert10), ("f1_optimal", 0.09)):
            ops.append({"target": SEVERE, "period": period, "operating_point": name, "cutoff_score": 0.05, "alert_rate": alert, "precision": 0.07, "recall": 0.2, "f1": 0.1, "cutoff_probability": 0.045})
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_operating_points.csv", pd.DataFrame(ops))
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_calibration.csv", calibration_rows(ratio22, ratio23, prev22, prev23))
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_reliability.csv", pd.DataFrame(
        [{"variant": "platt fitted on early_stop (locked primary)", "bin": i, "rows": 1000, "events": 20 + i, "mean_predicted": 0.02, "observed_rate": 0.03} for i in range(1, 4)]))
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_by_group.csv", final_by_group(skew_months))
    write_csv(root, f"reports/tables/phase7b_final_{SEVERE}_severity.csv", pd.DataFrame([
        {"target": SEVERE, "dimension": "event_delay_band_minutes", "group": g, "events": e, "share_of_events": e / 1000, "small_sample": False, "recall_top0.5pct": 0.02,
         "recall_top1pct": 0.03, "recall_top5pct": 0.12, "recall_top10pct": r} for g, e, r in (("[120.0, 180.0)", 500, 0.22), ("[180.0, 300.0)", 300, 0.24), ("[300.0, inf)", 200, 0.20))]))

    # ---- final tables, cancellation
    write_csv(root, f"reports/tables/phase7b_final_{CANCEL}_summary.csv", summary(CANCEL, 0.0268, 0.0169, 0.030, 0.0176, 0.0343, 0.0208))
    write_csv(root, f"reports/tables/phase7b_final_{CANCEL}_capacity.csv", capacity_rows({"validation_2022": (687710, 18444), "final_2023": (463484, 7809)}, lift_whole=0.9, lift_day=1.8))
    write_csv(root, f"reports/tables/phase7b_final_{CANCEL}_by_group.csv", final_by_group(True).query("dimension == 'month_2023'"))

    # ---- earlier steps
    write_csv(root, "reports/tables/calibration_summary.csv", pd.DataFrame([
        {"experiment": "A", "method": m, "period": p, "mean_predicted": 0.02, "prevalence": 0.0267, "predicted_to_observed": 0.8, "brier": 0.025, "brier_skill": 0.01,
         "calibration_slope": 0.98, "calibration_in_the_large": 0.1, "ece": 0.003}
        for m in ("uncalibrated", "platt", "isotonic") for p in ("2022 first half (selection)", "2022 second half (confirmation)", "2022 full year", "2022 Q1", "2022 Q2", "2022 Q3", "2022 Q4")]))
    write_csv(root, "reports/tables/phase6_calibration_comparisons.csv", pd.DataFrame([
        {"comparison": "A: calibrator vs uncalibrated", "challenger": "platt", "reference": "uncalibrated", "period": "2022 second half (confirmation)", "difference": -1e-5, "ci_low": -2e-5, "ci_high": -5e-6}]))
    write_csv(root, "reports/tables/phase7a_shap_global.csv", pd.DataFrame([
        {"feature": "carrier_identifier", "mean_abs_shap": 0.15, "share_of_total": 0.25, "family": "carrier"}, {"feature": "scheduled_departure_hour_local", "mean_abs_shap": 0.15, "share_of_total": 0.3, "family": "calendar and clock"},
        {"feature": "scheduled_departure_month", "mean_abs_shap": 0.1, "share_of_total": 0.17, "family": "calendar and clock"}, {"feature": "origin_airport", "mean_abs_shap": 0.1, "share_of_total": 0.28, "family": "airport"}]))
    write_csv(root, "reports/tables/phase7a_shap_family.csv", pd.DataFrame([
        {"family": "calendar and clock", "mean_abs_shap": 0.25, "share_of_total": 0.47}, {"family": "carrier", "mean_abs_shap": 0.15, "share_of_total": 0.25}, {"family": "airport", "mean_abs_shap": 0.1, "share_of_total": 0.28}]))
    write_csv(root, "reports/tables/phase7a_shap_numeric_effects.csv", pd.DataFrame([
        {"feature": "scheduled_departure_hour_local", "bin": i, "value_low": i * 4.0, "value_high": i * 4.0 + 4, "rows": 1000, "mean_shap": -0.2 + 0.1 * i} for i in range(5)]))
    write_csv(root, "reports/tables/phase7a_shap_categorical_effects.csv", pd.DataFrame(
        [{"feature": "carrier_identifier", "direction": d, "level": f"C{i}", "rows": 1000, "mean_shap": (0.3 - 0.05 * i) * (1 if d == "highest" else -1)} for d in ("highest", "lowest") for i in range(7)]))
    cases = []
    for category in ("true_positive", "false_positive", "false_negative", "low_risk_correct"):
        for pick in ("typical_q25", "typical_q50", "typical_q75"):
            cases.append({"category": category, "pick": pick, "source_row_number": 1, "origin": "AAA", "destination": "BBB", "carrier": "C1", "local_departure_hour": 20.0, "score": 0.05,
                          "score_rank_in_h2": 100, "calibrated_probability": 0.05, "event": 1, "arrival_delay_minutes": 150.0, "top_features": "carrier_identifier=C1 (+0.300)"})
    for category, pick in (("high_confidence_false_positive", "highest_scoring_non_event"), ("high_confidence_false_negative", "lowest_scoring_event")):
        cases.append({"category": category, "pick": pick, "source_row_number": 1, "origin": "AAA", "destination": "BBB", "carrier": "C1", "local_departure_hour": 20.0, "score": 0.05,
                      "score_rank_in_h2": 1, "calibrated_probability": 0.05, "event": 0, "arrival_delay_minutes": 10.0, "top_features": "x"})
    write_csv(root, "reports/tables/phase7a_case_studies.csv", pd.DataFrame(cases))
    write_json(root, "reports/tables/phase7a_explain_decision.json", {"feature_set": "base_no_year", "seed": 42, "best_iteration": 53, "not_evaluable": ["2019 and 2020 regimes"],
                                                                      "shap": {"rows": 300000, "period": "2022 second half", "additivity_gap": 1e-15, "expected_value_log_odds": -3.6}})
    write_csv(root, "reports/tables/error_breakdown.csv", pd.DataFrame([
        {"dimension": "event_delay_band_minutes", "group": "[120.0, 180.0)", "recall_top10pct": 0.20}]))
    write_csv(root, "reports/tables/phase7b_severe_delay_120_backtest_models.csv", pd.DataFrame([
        {"target": SEVERE, "eval_year": y, "regime": r, "fit_rows": 1000, "trees": t, "eval_rows": 5000, "events": 100, "prevalence": 0.02, "pr_auc": 0.03, "pr_auc_lift": lift, "roc_auc": 0.6}
        for y, r, t, lift in ((2020, "2020 shock", 12, 1.27), (2022, "2022 recovery/transition (locked model)", 53, 1.8))]))
    cap11 = capacity_rows({"validation_2022": (667616, 17820)})
    cap11["eval_year"] = 2022
    write_csv(root, "reports/tables/phase7b_severe_delay_120_review_capacity.csv", cap11)
    write_csv(root, "reports/tables/phase7b_cancelled_review_capacity.csv", cap11)
    write_csv(root, "reports/tables/phase7b_cancelled_backtest_models.csv", pd.DataFrame([
        {"target": CANCEL, "eval_year": 2022, "regime": "2022 recovery/transition (locked model)", "fit_rows": 1000, "trees": 59, "eval_rows": 5000, "events": 100, "prevalence": 0.02, "pr_auc": 0.03, "pr_auc_lift": 1.1, "roc_auc": 0.56}]))
    q22 = group_table("quarter_2022", [(f"2022 Q{q}", 150000, 3000, 0.01, 0.09, 0.2, 0.1, 0.04) for q in (1, 2, 3, 4)])
    h22 = group_table("scheduled_departure_local_hours", [("00-05", 3000, 50, 0.0, float("nan"), 0.01, 0.0, 0.02), ("06-09", 7000, 100, 0.0, float("nan"), 0.01, 0.01, 0.02)])
    write_csv(root, "reports/tables/phase7b_severe_delay_120_review_by_group_2022.csv", pd.concat([q22, h22], ignore_index=True))
    write_csv(root, "reports/tables/phase7b_cancelled_review_by_group_2022.csv", q22)
    for name in ("shap_summary.png", "local_flight_explanation.png", "calibration_curve.png", "final_calibration_curve_severe_delay_120.png", "final_review_capacity_severe_delay_120.png"):
        (root / "reports/figures").mkdir(parents=True, exist_ok=True)
        (root / "reports/figures" / name).write_bytes(b"png")
    return root


@pytest.fixture
def report_root(tmp_path) -> Path:
    return build_root(tmp_path / "project")


@pytest.fixture
def make_root(tmp_path):
    """A factory: make_root(name, **options) builds another synthetic project folder."""
    def factory(name: str = "variant", **options) -> Path:
        return build_root(tmp_path / name, **options)
    return factory


@pytest.fixture
def lock_hash() -> str:
    return LOCK_HASH


@pytest.fixture
def write_json_file():
    return write_json

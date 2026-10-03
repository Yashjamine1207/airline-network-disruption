#!/usr/bin/env python
"""Phase 7B, Step 12b: the one-time final test on 2023-01-01 to 2023-08-31. Needs the lock from Step 12a.

Save as:   scripts/run_phase7b_final_test.py
Run from the project root, once the lock is read and committed:

    python scripts/run_phase7b_final_test.py --confirm USE-2023-ONCE

What it does
------------
1. Checks the lock: sealed and unedited, from a real (not smoke) run, and every file it fingerprinted is unchanged.
2. Refits the locked LightGBM (and the cancellation baseline) exactly as locked and proves it is the same model:
   same tree count, same 2022 PR-AUC, same sum of 2022 scores. It stops here if not. 2023 has not been read yet.
3. Only then loads the 2023 rows and scores them. The scores are saved at once to
   data/processed/phase7/final/ (never committed). Nothing is chosen, tuned or refitted on 2023.
4. Reports 2022 and 2023 side by side, from the SAME code:
   * PR-AUC, lift, ROC-AUC with day-level bootstrap intervals; the history baselines; paired differences
   * Precision@K and Recall@K at 0.5%, 1%, 5%, 10%: whole period, within each month, within each day
   * the operating points learned on 2022, applied unchanged: confusion matrices, precision, recall, F1
   * calibration (severe delay only): uncalibrated, the locked primary calibrator, the labelled diagnostic
   * where the alerts land (month, carrier, origin airport, departure hour, weekday, unseen values, score decile)
   * an ILLUSTRATIVE penalty table (assumed weights, not airline costs)
   * the three pre-registered claims, each marked SUPPORTED or NOT SUPPORTED
5. Writes the marker reports/final/final_test_consumed.json.

Running it again
----------------
You can rerun it to fix a reporting bug. A rerun refits, scores 2023 again and must reproduce the stored 2023 scores
(largest difference at most 1e-9) or it stops. A rerun cannot change the model or the protocol, only the reporting.

Cancellation is an untuned baseline: ranking only, no calibration, no thresholds.
Regression (MAE) is not part of this test; there is no locked regression model in the lock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.calibration.metrics import evaluate_probabilities, paired_day_bootstrap_brier, reliability_table
    from airline_disruption.calibration.methods import make_calibrator
    from airline_disruption.evaluation.classification import binary_metrics
    from airline_disruption.explain import errors, plots
    from airline_disruption.explain.dimensions import build_dimensions
    from airline_disruption.features.feature_sets import ID_COLUMN, categorical_columns_in
    from airline_disruption.final import lock as lk
    from airline_disruption.final import protocol as proto
    from airline_disruption.final.bootstrap import DayBootstrap
    from airline_disruption.final.operating_points import operating_point_table
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import refit as refit_module
    from airline_disruption.models.rate_baselines import build_rate_baselines
    from airline_disruption.policy import review
    from airline_disruption.utils.run_info import hardware_context, package_versions
    from airline_disruption.validation.temporal_splits import SPLIT_FINAL_TEST
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.calibration.metrics import evaluate_probabilities, paired_day_bootstrap_brier, reliability_table  # noqa: E402
    from airline_disruption.calibration.methods import make_calibrator  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics  # noqa: E402
    from airline_disruption.explain import errors, plots  # noqa: E402
    from airline_disruption.explain.dimensions import build_dimensions  # noqa: E402
    from airline_disruption.features.feature_sets import ID_COLUMN, categorical_columns_in  # noqa: E402
    from airline_disruption.final import lock as lk  # noqa: E402
    from airline_disruption.final import protocol as proto  # noqa: E402
    from airline_disruption.final.bootstrap import DayBootstrap  # noqa: E402
    from airline_disruption.final.operating_points import operating_point_table  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import refit as refit_module  # noqa: E402
    from airline_disruption.models.rate_baselines import build_rate_baselines  # noqa: E402
    from airline_disruption.policy import review  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402
    from airline_disruption.validation.temporal_splits import SPLIT_FINAL_TEST  # noqa: E402

FINAL_START = pd.Timestamp("2023-01-01", tz="UTC")
SCORE_TOLERANCE = 1e-9
fmt = lambda v: f"{v:.4f}"  # noqa: E731


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Scoring 2023 (the only place 2023 rows are read)
# ---------------------------------------------------------------------------
def load_and_score_final_rows(root: Path, lock: dict, target: str, refit) -> tuple[lb.ModelFrame, np.ndarray]:
    cfg = proto.TARGETS[target]
    test = lb.load_model_frame(root, lock["feature_set"], splits=(SPLIT_FINAL_TEST,), target=target,
                               eligibility_column=cfg["eligibility"], allow_final_test=True)
    meta = test.meta
    if not (meta["phase6_split"].astype(str) == SPLIT_FINAL_TEST).all():
        fail("a row outside the final-test split was loaded")
    if not (meta["prediction_timestamp_utc"] >= FINAL_START).all():
        fail("a final-test row has a prediction time before 2023-01-01 UTC")
    if not meta["prediction_timestamp_utc"].min().value // 10**9 > int(refit.seconds_val.max()):
        fail("the final-test rows do not come strictly after the validation rows")
    lb.mask_unavailable_history(test.X, meta)
    for column in categorical_columns_in(test.X.columns):  # levels come from the FIT rows only, as in validation
        test.X[column] = test.X[column].cat.set_categories(refit.X_fit[column].cat.categories)
    return test, lb.predict_scores(refit.fit, test.X)


def persist_or_compare_scores(root: Path, lock: dict, target: str, test, scores: np.ndarray) -> dict:
    """First run: save the scores. Later runs: they must reproduce the saved ones."""
    path = root / proto_scores_path(target)
    ids = test.meta[ID_COLUMN].to_numpy()
    if path.exists():
        stored = pd.read_parquet(path)
        if len(stored) != len(ids) or not np.array_equal(stored[ID_COLUMN].to_numpy(), ids):
            fail(f"the stored 2023 scores for {target} cover different flights from this run")
        difference = lk.max_score_difference(stored["score"].to_numpy(), scores)
        if difference > SCORE_TOLERANCE:
            fail(f"this run does not reproduce the stored 2023 scores for {target} (largest difference {difference:.2e}). "
                 "The model or the data changed since the first run.")
        print(f"  reproduced the stored 2023 scores (largest difference {difference:.1e}); this is a reporting rerun")
        return {"first_run": False, "largest_difference_from_stored": difference, "file": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: ids, "score": scores, "label": test.y}).to_parquet(path, index=False)
    print(f"  saved the 2023 scores to {path} (not for Git)")
    return {"first_run": True, "largest_difference_from_stored": 0.0, "file": str(path)}


def proto_scores_path(target: str) -> str:
    return f"{lk.SCORES_DIR}/final_test_scores_{target}.parquet"


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------
def run_target(root: Path, lock: dict, target: str, n_boot: int, tables: Path, figures: Path) -> dict:
    cfg, spec = proto.TARGETS[target], lock["targets"][target]
    seed = int(lock["seed"])
    print(f"\n{'=' * 100}\n{cfg['label']} ({target}): {cfg['role']}\n{'=' * 100}", flush=True)

    refit = refit_module.refit_selected_lightgbm(root, lock["feature_set"], lock["params"], seed, int(lock["n_jobs"]), lock.get("max_fit_rows"),
                                                 target=target, eligibility_column=cfg["eligibility"])
    validation_pr = binary_metrics(refit.y_val, refit.s_val, probabilities=False)["pr_auc"]
    lk.check_reproduces(spec, refit.fit.best_iteration, validation_pr)
    if abs(float(np.sum(refit.s_val)) - spec["validation_score_sum"]) > 1e-6:
        fail("the refit does not reproduce the locked sum of 2022 scores")
    print(f"  the refit is the locked model: {refit.fit.best_iteration} trees, 2022 PR-AUC {validation_pr:.6f}. Now reading 2023.", flush=True)

    test, s_t = load_and_score_final_rows(root, lock, target, refit)
    y_t = test.y
    print(f"  final test: {len(y_t):,} rows from {test.meta['prediction_timestamp_utc'].min():%Y-%m-%d} to {test.meta['prediction_timestamp_utc'].max():%Y-%m-%d} UTC, "
          f"{int(y_t.sum()):,} events ({y_t.mean():.3%})")
    persisted = persist_or_compare_scores(root, lock, target, test, s_t)

    y_v, s_v = refit.y_val, refit.s_val
    days_v, days_t = refit.meta_val["day"].to_numpy(), test.meta["day"].to_numpy()
    months_v = refit.meta_val["prediction_timestamp_utc"].dt.strftime("%Y-%m").to_numpy()
    months_t = test.meta["prediction_timestamp_utc"].dt.strftime("%Y-%m").to_numpy()
    stray = int((months_t == "2023-09").sum())
    if stray:
        print(f"  note: {stray} rows have a UTC prediction time on 2023-09-01 (late flights on 31 August in western time zones); they form their own tiny month group")
    best = spec["best_history_baseline"]

    # ---- discrimination, baselines, paired differences ---------------------------------------------------
    base_v = build_rate_baselines(refit.X_fit, refit.y_fit, refit.X_val)
    base_t = build_rate_baselines(refit.X_fit, refit.y_fit, test.X)
    summaries = []
    for period, y, s, base, days in (("validation_2022", y_v, s_v, base_v, days_v), ("final_2023", y_t, s_t, base_t, days_t)):
        scorers = {"lightgbm_locked": s, **{name: base[name] for name in ("baseline_prevalence", *proto.HISTORY_BASELINES)}}
        table = DayBootstrap(y, days, n_boot, seed, proto.BOOTSTRAP["confidence"]).summarise(scorers, reference=best)
        roc = {name: binary_metrics(y, v, probabilities=False)["roc_auc"] if name != "baseline_prevalence" else 0.5 for name, v in scorers.items()}
        table["roc_auc"] = table["scorer"].map(roc)
        table.insert(0, "period", period)
        table.insert(1, "rows", len(y))
        table.insert(2, "events", int(y.sum()))
        table.insert(3, "prevalence", float(y.mean()))
        summaries.append(table)
    summary = pd.concat(summaries, ignore_index=True)
    summary.insert(0, "target", target)
    summary.to_csv(tables / f"phase7b_final_{target}_summary.csv", index=False)
    print(f"\nPR-AUC with 95% day-bootstrap intervals (difference is against {best}, the best history baseline on 2022):")
    show = ["period", "scorer", "pr_auc", "pr_auc_low", "pr_auc_high", "lift", "lift_low", "lift_high", "roc_auc", "difference_vs_reference", "difference_low", "difference_high"]
    print(summary[show].to_string(index=False, float_format=fmt))

    # ---- review capacity -----------------------------------------------------------------------------------
    capacity_frames = []
    for period, year, y, s, months, days in (("validation_2022", 2022, y_v, s_v, months_v, days_v), ("final_2023", 2023, y_t, s_t, months_t, days_t)):
        table = review.top_k_review(y, s, proto.CAPACITIES, months, days, n_boot=n_boot, seed=seed)
        table.insert(0, "eval_year", year)
        table.insert(0, "period", period)
        table.insert(0, "target", target)
        capacity_frames.append(table)
    capacity = pd.concat(capacity_frames, ignore_index=True)
    capacity.to_csv(tables / f"phase7b_final_{target}_capacity.csv", index=False)
    utility_frames = []
    for (period, year), part in capacity.groupby(["period", "eval_year"], sort=False):
        u = review.utility_table(part)
        u.insert(0, "eval_year", year)
        u.insert(0, "period", period)
        u.insert(0, "target", target)
        utility_frames.append(u)
    utility = pd.concat(utility_frames, ignore_index=True)
    utility.to_csv(tables / f"phase7b_final_{target}_utility.csv", index=False)
    cap_cols = ["capacity", "reviewed", "events_captured", "false_alerts", "precision", "precision_low", "precision_high", "recall", "recall_low", "recall_high", "lift", "break_even_ratio"]
    for period in ("validation_2022", "final_2023"):
        for scheme in review.SCHEMES:
            part = capacity[(capacity["period"] == period) & (capacity["scheme"] == scheme)]
            label = {"whole_year": "whole period"}.get(scheme, scheme.replace("_", " "))
            print(f"\n{period}, {label} capacity ({int(part['events'].iloc[0]):,} events in {int(part['flights'].iloc[0]):,} flights, event rate {part['prevalence'].iloc[0]:.3%}); ranking by the UNCALIBRATED score:")
            print(part[cap_cols].to_string(index=False, float_format=fmt))

    # ---- operating points and calibration (severe delay only) ----------------------------------------------
    p_primary = p_diag = None
    calibration_rows, reliability_rows, calibration_bootstrap = [], [], {}
    if cfg["calibrate"]:
        sources = {"early_stop": (refit.s_stop, refit.y_stop), "validation_2022": (s_v, y_v)}
        calibration = lock["calibration"]
        calibrators = {}
        for label, name in (("primary", calibration["primary_fit_rows"]), ("diagnostic", calibration["diagnostic_fit_rows"])):
            calibrator = make_calibrator(calibration["method"]).fit(*sources[name])
            recorded = spec[f"calibrator_{label}"]
            for key in ("fit_rows", "fit_events", "intercept", "slope"):
                if key in recorded and abs(float(calibrator.describe()[key]) - float(recorded[key])) > 1e-9:
                    fail(f"the {label} calibrator does not match the lock ({key})")
            calibrators[label] = calibrator
        p_primary, p_diag = calibrators["primary"].transform(s_t), calibrators["diagnostic"].transform(s_t)
        reference_rate = float(spec["calibrator_primary"]["fit_event_rate"])
        variants_v = {"uncalibrated": s_v, "primary": calibrators["primary"].transform(s_v), "diagnostic": calibrators["diagnostic"].transform(s_v)}
        variants_t = {"uncalibrated": s_t, "primary": p_primary, "diagnostic": p_diag}
        names = {"uncalibrated": "uncalibrated", "primary": f"{calibration['method']} fitted on {calibration['primary_fit_rows']} (locked primary)",
                 "diagnostic": f"{calibration['method']} fitted on {calibration['diagnostic_fit_rows']} (diagnostic, not the locked choice)"}
        for period, y, variants in (("validation_2022", y_v, variants_v), ("final_2023", y_t, variants_t)):
            for variant, p in variants.items():
                calibration_rows.append({"target": target, "period": period, "variant": names[variant], **evaluate_probabilities(y, p, reference_rate)})
                if period == "final_2023":
                    r = reliability_table(y, p, 10)
                    r.insert(0, "variant", names[variant])
                    reliability_rows.append(r)
        calibration_table = pd.DataFrame(calibration_rows)
        calibration_table.to_csv(tables / f"phase7b_final_{target}_calibration.csv", index=False)
        pd.concat(reliability_rows, ignore_index=True).to_csv(tables / f"phase7b_final_{target}_reliability.csv", index=False)
        calibration_bootstrap = {
            "primary_vs_uncalibrated": paired_day_bootstrap_brier(y_t, s_t, p_primary, days_t, n_boot=n_boot, seed=seed),
            "diagnostic_vs_primary": paired_day_bootstrap_brier(y_t, p_primary, p_diag, days_t, n_boot=n_boot, seed=seed),
        }
        print("\nCalibration (the 2023 rows were never used to fit or choose anything):")
        show_cal = ["period", "variant", "mean_predicted", "prevalence", "predicted_to_observed", "brier", "brier_skill", "calibration_slope", "calibration_in_the_large", "ece", "pr_auc"]
        print(calibration_table[show_cal].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        for name, result in calibration_bootstrap.items():
            print(f"  Brier difference, {name}: {result['difference']:+.6f}  95% interval [{result['ci_low']:+.6f}, {result['ci_high']:+.6f}]  (negative favours the first-named)")

        points = spec["operating_points"]
        pm = calibrators["primary"].transform
        operating = pd.concat([operating_point_table(y_v, s_v, points, "validation_2022", pm), operating_point_table(y_t, s_t, points, "final_2023", pm)], ignore_index=True)
        operating.insert(0, "target", target)
        operating.to_csv(tables / f"phase7b_final_{target}_operating_points.csv", index=False)
        print("\nOperating points learned on 2022 and applied unchanged (confusion matrices):")
        print(operating[["period", "operating_point", "cutoff_score", "cutoff_probability", "alert_rate", "true_positives", "false_positives", "false_negatives", "true_negatives", "precision", "recall", "f1"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- where the alerts land ------------------------------------------------------------------------------
    cutoffs = errors.top_fraction_cutoffs(s_t, proto.CAPACITIES)
    frame_eval = pd.DataFrame({"label": y_t, "score": s_t, "probability": p_primary if p_primary is not None else s_t})
    dims = build_dimensions(test.X, refit.X_fit, s_t, list(months_t), period_name="month_2023")
    groups = [errors.group_metrics(frame_eval.assign(group=pd.Series(labels).to_numpy()), name, cutoffs, "group") for name, labels in dims.items()]
    by_group = pd.concat([errors.group_metrics(frame_eval.assign(group="all 2023 flights"), "overall", cutoffs, "group"), *groups], ignore_index=True)
    if not cfg["calibrate"]:
        by_group = by_group.drop(columns=["mean_probability", "probability_to_observed", "brier_probability"])
    by_group.insert(0, "target", target)
    by_group.to_csv(tables / f"phase7b_final_{target}_by_group.csv", index=False)
    wide = ["group", "rows", "events", "prevalence", "alert_rate_top1pct", "precision_top1pct", "recall_top1pct", "alert_rate_top10pct", "precision_top10pct", "recall_top10pct", "pr_auc"]
    print(f"\nWhere the alerts land in 2023 (one global cutoff per capacity over the whole final period): top-1% cutoff {cutoffs[0.01]:.5f}, top-10% cutoff {cutoffs[0.10]:.5f}")
    for dimension in ("month_2023", "scheduled_departure_local_hours", "missing_or_unseen_values", "score_decile", "carrier"):
        print(f"\n{dimension}:")
        print(by_group[by_group["dimension"] == dimension][wide].to_string(index=False, float_format=fmt))
    airports = by_group[by_group["dimension"] == "origin_airport_top25"]
    print("\norigin airports: highest and lowest share of flights reviewed at the top 1% capacity")
    print(pd.concat([airports.nlargest(4, "alert_rate_top1pct"), airports.nsmallest(4, "alert_rate_top1pct")])[wide].to_string(index=False, float_format=fmt))

    if target == "severe_delay_120":
        delays = pd.read_parquet(root / lb.COHORT_PATH, columns=[ID_COLUMN, "arrival_delay_minutes"])
        severity_frame = frame_eval.assign(**{ID_COLUMN: test.meta[ID_COLUMN].to_numpy()}).merge(delays, on=ID_COLUMN, how="left")
        severity = errors.severity_table(severity_frame, cutoffs)
        severity.insert(0, "target", target)
        severity.to_csv(tables / f"phase7b_final_{target}_severity.csv", index=False)
        print("\nEvents by size of delay (share caught by a global top-K% review; realised delay used only to describe, after the fact):")
        print(severity.to_string(index=False, float_format=fmt))

    # ---- pre-registered claims -------------------------------------------------------------------------------
    final_row = summary[(summary["period"] == "final_2023") & (summary["scorer"] == "lightgbm_locked")].iloc[0]
    by_day10 = capacity[(capacity["period"] == "final_2023") & (capacity["scheme"] == "by_day") & (capacity["capacity"] == "10%")].iloc[0]
    claims = {
        "C1_skill_over_chance": {"statement": proto.CLAIMS["C1_skill_over_chance"], "lift": float(final_row["lift"]), "lift_low": float(final_row["lift_low"]), "supported": bool(final_row["lift_low"] > 1.0)},
        "C2_beats_history": {"statement": proto.CLAIMS["C2_beats_history"], "baseline": best, "difference": float(final_row["difference_vs_reference"]),
                             "difference_low": float(final_row["difference_low"]), "difference_high": float(final_row["difference_high"]), "supported": bool(final_row["difference_low"] > 0)},
        "C3_ranking_not_only_seasonal": {"statement": proto.CLAIMS["C3_ranking_not_only_seasonal"], "by_day_precision_at_10pct": float(by_day10["precision"]),
                                         "by_day_precision_at_10pct_low": float(by_day10["precision_low"]), "event_rate_2023": float(y_t.mean()), "supported": bool(by_day10["precision_low"] > y_t.mean())},
    }
    print(f"\nPre-registered claims for {target} (fixed in the lock before 2023 was read):")
    for name, claim in claims.items():
        print(f"  {name}: {'SUPPORTED' if claim['supported'] else 'NOT SUPPORTED'}")
        print(f"      {claim['statement']}")
        print("      " + ", ".join(f"{k} {v:.5f}" for k, v in claim.items() if isinstance(v, float)))

    # ---- figures -------------------------------------------------------------------------------------------------
    plots.pr_curves(figures / f"final_pr_curve_{target}.png", y_t, {"LightGBM (locked)": s_t, f"best history baseline ({best.replace('baseline_', '')})": base_t[best]},
                    subtitle="final test, 2023-01 to 2023-08", title=f"Precision-recall, {cfg['label'].split(' (')[0].lower()}")
    if cfg["calibrate"]:
        plots.reliability_figure(figures / f"final_calibration_curve_{target}.png", {r["variant"].iloc[0]: r for r in reliability_rows}, "Reliability on the final test", "2023-01 to 2023-08, 10 equal-count bins")
    whole = utility[(utility["period"] == "final_2023") & (utility["scheme"] == "whole_year")]
    plots.review_figure(figures / f"final_review_capacity_{target}.png", capacity, whole, cfg["label"].split(" (")[0], focus_year=2023)

    return {"target": target, "role": cfg["role"], "rows": int(len(y_t)), "events": int(y_t.sum()), "event_rate": float(y_t.mean()), "claims": claims,
            "score_file": persisted, "calibration_bootstrap": calibration_bootstrap, "stray_rows_on_2023_09_01_utc": stray,
            "trees": int(refit.fit.best_iteration), "validation_pr_auc": float(validation_pr), "final_pr_auc": float(final_row["pr_auc"]), "final_lift": float(final_row["lift"])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7B Step 12b: the one-time final test (2023-01-01 to 2023-08-31)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--lock", default=lk.LOCK_PATH)
    parser.add_argument("--confirm", default=None, help=f"must be {lk.CONFIRMATION_PHRASE}")
    parser.add_argument("--bootstrap", type=int, default=None, help="day-level resamples (default: the value in the lock)")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    try:
        lock = lk.read_lock(root / args.lock)
        lk.require_final_lock(lock)
        lk.require_confirmation(args.confirm)
        lk.check_fingerprints(root, lock["files"])
    except ValueError as problem:
        fail(str(problem))
    n_boot = int(args.bootstrap or lock.get("bootstrap", {}).get("resamples", 1000))
    print(f"Lock {lock['lock_sha256'][:16]}... written {lock['created_utc']}. Model: tuned LightGBM, {lock['feature_set']}, seed {lock['seed']}, not refitted on 2022.")
    print(f"Calibration: {lock['calibration']['method']}, primary fitted on {lock['calibration']['primary_fit_rows']} ({lock['calibration']['basis']}).")

    consumed_path = root / lk.CONSUMED_PATH
    consumed = json.loads(consumed_path.read_text(encoding="utf-8")) if consumed_path.exists() else None
    if consumed and consumed["lock_sha256"] != lock["lock_sha256"]:
        fail("the final test was already run under a DIFFERENT lock. It cannot be run again under this one.")
    print("This is a REPORTING RERUN: the final test was already scored once under this lock." if consumed else "This is the first and only scoring of the final test.")

    tables, figures = root / "reports/tables", root / "reports/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for target in lock["targets"]:
            results[target] = run_target(root, lock, target, n_boot, tables, figures)
    except ValueError as problem:
        fail(str(problem))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    marker = consumed or {"lock_sha256": lock["lock_sha256"], "first_run_utc": now, "targets": {}, "reruns": []}
    if consumed:
        marker["reruns"].append({"utc": now, "largest_score_difference": {t: r["score_file"]["largest_difference_from_stored"] for t, r in results.items()}})
    else:
        marker["targets"] = {t: {"rows": r["rows"], "events": r["events"], "score_file": r["score_file"]["file"]} for t, r in results.items()}
    consumed_path.parent.mkdir(parents=True, exist_ok=True)
    consumed_path.write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")

    decision = {"created_utc": now, "lock_sha256": lock["lock_sha256"], "final_test_used": True, "first_run": not consumed,
                "model": "tuned LightGBM, base_no_year, seed 42, not refitted on 2022", "ranking_score": "uncalibrated LightGBM score",
                "bootstrap": {"unit": "whole UTC prediction days", "resamples": n_boot, "interval": "95% percentile"},
                "targets": results, "rejected_challengers_not_scored": lock["rejected_challengers"],
                "limits": ["a single period (Jan to Aug 2023): one regime, no regime comparison inside the final test",
                           "the model saw nothing after 2021-06; the gap to 2023 is 18 to 26 months",
                           "schedule-only features: no same-day airport state, no weather, no aircraft identity",
                           "penalty table values are assumptions, never airline costs",
                           "regression (MAE) and airport recovery episodes are not part of this test"],
                "hardware": hardware_context(), "packages": package_versions(), "total_seconds": round(time.perf_counter() - started, 1)}
    (tables / "phase7b_final_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote tables in {tables}, figures in {figures} and {consumed_path}. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

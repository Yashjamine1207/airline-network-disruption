#!/usr/bin/env python
"""Phase 7B, Step 12a: lock the final protocol. Uses development and 2022 rows only. 2023 is never loaded.

Save as:   scripts/run_phase7b_final_lock.py
Run from the project root:

    # 1. Smoke run on reduced rows (about a minute). Writes a lock marked smoke that the final test refuses.
    python scripts/run_phase7b_final_lock.py --max-fit-rows 300000

    # 2. The real lock (a few minutes). Read what it prints, then commit the file BEFORE the final test:
    python scripts/run_phase7b_final_lock.py
    git add reports/final/final_protocol_lock.json && git commit -m "Lock the final-test protocol"

    # Only if you have accepted DEC-024 (fit the final calibrator on all of 2022):
    python scripts/run_phase7b_final_lock.py --calibration-fit validation_2022

What it locks
-------------
* The model: the tuned LightGBM on base_no_year, seed 42, exactly as validated (fit 2019-01 to 2021-06, early
  stopping 2021-07 to 2021-12). It is NOT refitted on 2022. The thing that was validated is the thing that is tested.
* The calibration method (from configs/calibration_selected.json) and which rows the final calibrator is fitted on:
  the early-stopping rows (DEC-023 Rule 2 as run) unless you pass --calibration-fit validation_2022 (DEC-024).
  The other choice is kept as a labelled diagnostic, so the final report shows both. The primary is fixed here.
* Operating points learned on 2022: score cutoffs for the top 0.5%, 1%, 5%, 10% and the F1-optimal cutoff.
* The only three pass/fail claims, and the history baseline they are judged against.
* SHA-256 fingerprints of every file the result depends on, and the tree count and 2022 PR-AUC of each model,
  so the final test can prove it is scoring the same model.
* Cancellation is locked as an untuned baseline: ranking only.

Writes reports/final/final_protocol_lock.json (small, safe to commit). A smoke run writes final_protocol_lock_smoke.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.calibration import guards
    from airline_disruption.calibration.methods import make_calibrator
    from airline_disruption.calibration.metrics import evaluate_probabilities
    from airline_disruption.evaluation.classification import binary_metrics
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION
    from airline_disruption.final import lock as lk
    from airline_disruption.final import protocol as proto
    from airline_disruption.final.operating_points import learn_cutoffs, operating_point_table
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.models import refit as refit_module
    from airline_disruption.models.rate_baselines import build_rate_baselines
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.calibration import guards  # noqa: E402
    from airline_disruption.calibration.methods import make_calibrator  # noqa: E402
    from airline_disruption.calibration.metrics import evaluate_probabilities  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION  # noqa: E402
    from airline_disruption.final import lock as lk  # noqa: E402
    from airline_disruption.final import protocol as proto  # noqa: E402
    from airline_disruption.final.operating_points import learn_cutoffs, operating_point_table  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.models import refit as refit_module  # noqa: E402
    from airline_disruption.models.rate_baselines import build_rate_baselines  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

import json  # noqa: E402


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def span(seconds: np.ndarray) -> str:
    low, high = pd.to_datetime(int(seconds.min()), unit="s", utc=True), pd.to_datetime(int(seconds.max()), unit="s", utc=True)
    return f"{low:%Y-%m-%d %H:%M} to {high:%Y-%m-%d %H:%M} UTC"


def choose_calibration_source(root: Path, override: str | None) -> tuple[str, str, str]:
    """``(method, fit_rows, basis)``. The pre-registered rules say what to do; an override is recorded as a deviation."""
    method, rule_two = "platt", "same_as_here"
    selected_path = root / proto.CALIBRATION_SELECTED
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        method = selected["rule_1"]["chosen_method"]
        rule_two = selected["rule_2"]["final_test_fit_source"]
        print(f"Calibration decision read from {proto.CALIBRATION_SELECTED}: method {method}, Rule 2 says {rule_two}.")
    else:
        print(f"NOTE: {proto.CALIBRATION_SELECTED} not found; using Platt and the early-stopping rows (the DEC-023 outcome).")
    registered = "validation_2022" if rule_two == "recent_year" else "early_stop"
    if override is None or override == registered:
        return method, registered, "DEC-023 Rule 2 as run (pre-registered)"
    return method, override, "DEC-024 (post-hoc amendment to Rule 2, accepted by the user before this lock was written)"


def lock_target(root: Path, target: str, args, params: dict, method: str, fit_rows: str) -> dict:
    cfg = proto.TARGETS[target]
    print(f"\n=== {cfg['label']} ({target}) ===  {cfg['role']}", flush=True)
    refit = refit_module.refit_selected_lightgbm(root, args.feature_set, params, args.seed, args.n_jobs, args.max_fit_rows,
                                                 target=target, eligibility_column=cfg["eligibility"])
    all_seconds = np.concatenate([refit.seconds_fit, refit.seconds_stop, refit.seconds_val])
    guards.reject_final_test_rows(all_seconds, f"{target} refit")
    y_val, s_val = refit.y_val, refit.s_val
    metrics = binary_metrics(y_val, s_val, probabilities=False)
    print(f"  fit {refit.n_fit:,} rows ({span(refit.seconds_fit)})\n  early stopping {refit.n_stop:,} ({span(refit.seconds_stop)}), {refit.fit.best_iteration} trees\n"
          f"  validation 2022: {len(y_val):,} rows, {int(y_val.sum()):,} events ({y_val.mean():.3%}); PR-AUC {metrics['pr_auc']:.5f} (lift {metrics['pr_auc_lift']:.2f})")

    reproduction = {"checked": False}
    if target == "severe_delay_120" and args.max_fit_rows is None:
        reproduction = refit_module.check_against_stored(refit, root / proto.REFERENCE_PATH, proto.REFERENCE_COLUMN, proto.REPRODUCTION_TOLERANCE)
        print(f"  check against the stored Phase 6 scores: PR-AUC difference {reproduction['pr_auc_difference']:.6f}")

    # Baselines from the FIT rows only, scored on 2022. The best one is the opponent for claim C2.
    baselines = build_rate_baselines(refit.X_fit, refit.y_fit, refit.X_val)
    baseline_pr = {name: binary_metrics(y_val, baselines[name], probabilities=False)["pr_auc"] for name in proto.HISTORY_BASELINES}
    best_baseline = max(baseline_pr, key=baseline_pr.get)
    print("  history baselines, 2022 PR-AUC: " + ", ".join(f"{n.replace('baseline_', '').replace('_rate', '')} {v:.5f}" for n, v in baseline_pr.items()) + f"  -> best: {best_baseline}")

    points = {}
    if cfg["calibrate"]:  # cancellation is ranking only: no thresholds are locked for it
        points = learn_cutoffs(y_val, s_val, proto.CAPACITIES)
        table = operating_point_table(y_val, s_val, points, "validation_2022")
        for name, row in table.set_index("operating_point").iterrows():
            points[name].update({"validation_alert_rate": float(row["alert_rate"]), "validation_precision": float(row["precision"]), "validation_recall": float(row["recall"])})
        print("  operating points learned on 2022 (score cutoff, share of flights flagged, precision, recall):")
        for name, row in table.set_index("operating_point").iterrows():
            print(f"    {name:16s} cutoff {row['cutoff_score']:.5f}  flagged {row['alert_rate']:.2%}  precision {row['precision']:.4f}  recall {row['recall']:.4f}")
        if points["f1_optimal"]["validation_alert_rate"] > 0.5:
            print("    NOTE: the F1-optimal cutoff flags more than half of all flights. For a weak ranking that is what maximising F1 does; it is not a useful alert rule.")

    record = {
        "role": cfg["role"], "label": cfg["label"], "calibrate": cfg["calibrate"], "best_iteration": int(refit.fit.best_iteration),
        "n_features": int(refit.X.shape[1]), "validation_rows": int(len(y_val)), "validation_events": int(y_val.sum()),
        "validation_prevalence": float(y_val.mean()), "validation_pr_auc": float(metrics["pr_auc"]), "validation_roc_auc": float(metrics["roc_auc"]),
        "validation_score_sum": float(np.sum(s_val)),
        "fit_rows": int(refit.n_fit), "fit_span": span(refit.seconds_fit), "early_stop_rows": int(refit.n_stop), "early_stop_span": span(refit.seconds_stop),
        "stored_score_reproduction": reproduction, "history_baselines_validation_pr_auc": baseline_pr, "best_history_baseline": best_baseline,
        "operating_points": points,
    }
    if cfg["calibrate"]:
        sources = {"early_stop": (refit.s_stop, refit.y_stop, refit.seconds_stop, "2021-07-01 to 2021-12-31 UTC (the early-stopping rows)"),
                   "validation_2022": (s_val, y_val, refit.seconds_val, "2022-01-01 to 2022-12-31 UTC (the whole validation year)")}
        other = "validation_2022" if fit_rows == "early_stop" else "early_stop"
        fitted = {}
        for label, name in (("primary", fit_rows), ("diagnostic", other)):
            scores, labels, seconds, description = sources[name]
            guards.reject_final_test_rows(seconds, f"{label} calibration")
            calibrator = make_calibrator(method).fit(scores, labels)
            fitted[label] = (calibrator, name)
            record[f"calibrator_{label}"] = {**calibrator.describe(), "fit_rows_source": name, "fit_period": description,
                                             "fit_event_rate": float(labels.mean())}
        guards.require_calibration_before_evaluation(refit.seconds_stop, refit.seconds_val, "early-stopping calibration rows before 2022")
        calibrator, name = fitted["primary"]
        reference_rate = float(sources[name][1].mean())
        validation_calibration = evaluate_probabilities(y_val, calibrator.transform(s_val), reference_rate)
        validation_calibration["in_sample_for_the_calibrator"] = name == "validation_2022"
        record["validation_calibration_primary"] = validation_calibration
        print(f"  calibrator (primary, {method}, fitted on {name}): {record['calibrator_primary']}")
        print(f"  calibrator (diagnostic, fitted on {other}): {record['calibrator_diagnostic']}")
        print(f"  validation 2022 with the primary calibrator: mean predicted {validation_calibration['mean_predicted']:.4f} against observed {validation_calibration['prevalence']:.4f} "
              f"(ratio {validation_calibration['predicted_to_observed']:.3f}), Brier {validation_calibration['brier']:.5f}"
              + ("  [in sample: the calibrator saw these rows]" if name == "validation_2022" else ""))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7B Step 12a: lock the final-test protocol (2023 is never loaded)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feature-set", default="base_no_year", choices=sorted(FEATURE_SETS))
    parser.add_argument("--params-file", default=proto.PARAMS_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8, help="stored in the lock; the final test refits with the same value")
    parser.add_argument("--calibration-fit", choices=["early_stop", "validation_2022"], default=None,
                        help="default follows DEC-023 Rule 2 as run. validation_2022 is the DEC-024 amendment.")
    parser.add_argument("--max-fit-rows", type=int, default=None, help="smoke runs only")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    smoke = args.max_fit_rows is not None
    tag = args.tag or ("smoke" if smoke else "")
    infix = f"_{tag}" if tag else ""
    if smoke:
        print("SMOKE RUN: reduced fit rows. The lock it writes is marked smoke and the final test will refuse it.")

    needed = [lb.COHORT_PATH, lb.PREDICTORS_PATH, args.params_file, proto.CALIBRATION_CONFIG]
    for relative in needed:
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}")
    params = tune.load_params_file(root / args.params_file)
    method, fit_rows, basis = choose_calibration_source(root, args.calibration_fit)
    print(f"Final calibrator: {method}, fitted on {fit_rows}. Basis: {basis}.")

    targets = {}
    try:
        for target in proto.TARGETS:
            targets[target] = lock_target(root, target, args, params, method, fit_rows)
    except ValueError as problem:
        fail(str(problem))

    fingerprint_paths = [*needed, *([proto.CALIBRATION_SELECTED] if (root / proto.CALIBRATION_SELECTED).exists() else [])]
    print("\nFingerprinting the files the result depends on (the two parquet files take a moment)...", flush=True)
    files = lk.file_fingerprints(root, fingerprint_paths)

    lock = {
        "lock_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "smoke": smoke, "tag": tag,
        "purpose": "Pre-registration of the one-time final test. Written before any 2023 row is read.",
        "prediction_time": "scheduled departure UTC minus 2 hours", "feature_version": FEATURE_VERSION, "feature_set": args.feature_set,
        "seed": args.seed, "n_jobs": args.n_jobs, "max_fit_rows": args.max_fit_rows, "params_file": args.params_file, "params": params,
        "model_policy": {
            "refit_on_2022_before_final_test": False,
            "reason": "The model that was validated is the model that is tested. Refitting on 2019 to 2022 would test a different model whose tree count and calibration were never validated.",
            "consequence": "The model has seen nothing after 2021-06 and the calibrator nothing after 2021-12 (or 2022-12 under DEC-024). The final test is 14 to 26 months later. That gap is a limitation to report.",
        },
        "periods": {"fit": "2019-01 to 2021-06 (UTC prediction time)", "early_stop": "2021-07 to 2021-12", "validation": "2022",
                    "final_test": "2023-01-01 to 2023-08-31 local flight date (UTC prediction time from 2023-01-01)"},
        "calibration": {"method": method, "primary_fit_rows": fit_rows, "basis": basis, "applies_to": [t for t, c in proto.TARGETS.items() if c["calibrate"]],
                        "diagnostic_fit_rows": "validation_2022" if fit_rows == "early_stop" else "early_stop",
                        "note": "The primary calibrator is the only one used for any probability statement. The diagnostic is shown beside it and is never chosen after the final test.",
                        "ranking_score": "uncalibrated LightGBM score. Platt is strictly increasing, so rankings are identical"},
        "capacities": list(proto.CAPACITIES), "bootstrap": proto.BOOTSTRAP,
        "review_schemes": {"whole_year": "top K% of every flight in the period (the final period is Jan to Aug 2023, not a full year)",
                           "by_month": "top K% within each UTC prediction month", "by_day": "top K% within each UTC prediction day"},
        "preregistered_claims": {"claims": proto.CLAIMS, "note": proto.CLAIM_NOTE},
        "rejected_challengers": proto.REJECTED_CHALLENGERS, "challengers_scored_on_final_test": [], "challenger_note": proto.NOT_SCORED_ON_FINAL_TEST,
        "targets": targets, "files": files,
        "not_in_this_test": ["regression (MAE): no locked regression model is part of this lock", "airport recovery episodes", "sequence models"],
        "environment": {"hardware": hardware_context(), "packages": package_versions()},
        "final_test_used": False,
    }
    path = root / (lk.LOCK_PATH.replace(".json", f"{infix}.json"))
    sealed = lk.write_lock(path, lock)
    print(f"\nWrote {path}\nSeal (SHA-256 of the lock): {sealed['lock_sha256']}")
    if smoke:
        print("This was a smoke lock. Run without --max-fit-rows for the real one.")
    else:
        print("\nNext:\n  1. Read the numbers above. If anything is wrong, fix it now: after the final test nothing can change.\n"
              "  2. Commit the lock:  git add reports/final/final_protocol_lock.json ; git commit -m \"Lock the final-test protocol\"\n"
              f"  3. Run the final test once:  python scripts/run_phase7b_final_test.py --confirm {lk.CONFIRMATION_PHRASE}")
    print(f"Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

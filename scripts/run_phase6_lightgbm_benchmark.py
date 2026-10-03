#!/usr/bin/env python
"""Phase 6, Step 4: LightGBM benchmark for severe arrival delay (>= 120 minutes).

Save as:   scripts/run_phase6_lightgbm_benchmark.py
Run from the project root, after build_phase6_cohort.py and build_phase6_predictors.py.

    # 1. Quick smoke run (a few minutes). Writes files tagged "smoke", never the real tables.
    python scripts/run_phase6_lightgbm_benchmark.py --max-fit-rows 300000 --bootstrap 20

    # 2. Full run: base against base_no_year, plus the historical-rate baselines.
    python scripts/run_phase6_lightgbm_benchmark.py

    # 3. Optional pipeline check against the Phase 4 number (PR-AUC about 0.045 on 2022).
    python scripts/run_phase6_lightgbm_benchmark.py --protocol phase4_reproduce --feature-sets base

What it does
------------
* Loads development and 2022 validation rows only. The final-test split is never loaded.
* Fits each requested feature set with the shared protocol (see lightgbm_benchmark.py).
* Scores the historical-rate baselines learned from the same fit rows.
* Reports PR-AUC, lift, ROC-AUC, Brier score, and precision/recall at the top 0.5%, 1%, 5%
  and 10% of flights, all on the 2022 validation rows.
* Compares models with a paired day-level bootstrap of the PR-AUC difference.

Writes (tables are small and safe to commit; predictions stay out of Git):
    reports/tables/phase6_lightgbm_validation_<protocol>.csv
    reports/tables/phase6_lightgbm_ranking_validation_<protocol>.csv
    reports/tables/phase6_lightgbm_calibration_validation_<protocol>.csv
    reports/tables/phase6_lightgbm_pairwise_<protocol>.csv
    reports/tables/phase6_lightgbm_run_manifest_<protocol>.json
    data/processed/phase6/predictions/lightgbm_validation_<protocol>.parquet

These are validation results. Nothing here is final-test evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.evaluation.classification import (
        DEFAULT_FRACTIONS,
        binary_metrics,
        calibration_table,
        paired_day_bootstrap,
        ranking_table,
    )
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models.lightgbm_tuning import load_params_file
    from airline_disruption.models.rate_baselines import build_rate_baselines
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.evaluation.classification import (  # noqa: E402
        DEFAULT_FRACTIONS,
        binary_metrics,
        calibration_table,
        paired_day_bootstrap,
        ranking_table,
    )
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models.lightgbm_tuning import load_params_file  # noqa: E402
    from airline_disruption.models.rate_baselines import build_rate_baselines  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 LightGBM benchmark (validation only)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feature-sets", nargs="+", default=["base", "base_no_year"], choices=sorted(FEATURE_SETS))
    parser.add_argument("--protocol", default=lb.PROTOCOL_PHASE6, choices=[lb.PROTOCOL_PHASE6, lb.PROTOCOL_PHASE4])
    parser.add_argument("--n-jobs", type=int, default=8, help="LightGBM threads (default 8; lower it if RAM is tight)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fit-rows", type=int, default=None, help="random subsample of fit rows, for smoke runs only")
    parser.add_argument("--bootstrap", type=int, default=100, help="day-level bootstrap resamples (0 to skip)")
    parser.add_argument("--tag", default="", help="suffix for output files; smoke runs are tagged automatically")
    parser.add_argument("--params-json", type=Path, default=None,
                        help="settings file written by tune_phase6_lightgbm.py (default: the Phase 4 settings)")
    args = parser.parse_args()
    root = args.root.resolve()

    params = None
    if args.params_json is not None:
        try:
            params = load_params_file(args.params_json)
        except (OSError, ValueError) as problem:
            fail(f"cannot use {args.params_json}: {problem}")

    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2 and Step 3 build scripts first)")

    tag = args.tag or ("smoke" if args.max_fit_rows else "")
    suffix = args.protocol + (f"_{tag}" if tag else "")
    tables = root / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    predictions_dir = root / "data/processed/phase6/predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Protocol: {args.protocol}   feature sets: {args.feature_sets}   tag: {tag or '-'}")
    print(f"LightGBM settings: {args.params_json if params else 'Phase 4 defaults'}")
    if args.max_fit_rows:
        print(f"SMOKE RUN: fit rows limited to {args.max_fit_rows:,}. Numbers are not results.")
    if args.protocol == lb.PROTOCOL_PHASE4:
        print("phase4_reproduce: early stopping uses the 2022 rows that are also reported. Not valid for model selection.")

    metric_rows: list[dict] = []
    ranking_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    predictions = pd.DataFrame()
    y_validation = None
    days = None
    fit_details: dict[str, dict] = {}

    for position, feature_set in enumerate(args.feature_sets):
        print(f"\n=== feature set: {feature_set} ({len(FEATURE_SETS[feature_set])} columns) ===")
        try:
            frame = lb.load_model_frame(root, feature_set)  # development + validation; final test cannot load
            roles = lb.assign_roles(frame.meta, args.protocol)
            lb.check_role_chronology(frame.meta, roles)
        except ValueError as problem:  # data checks raise ValueError with a readable message
            fail(str(problem))

        fit_mask = (roles == lb.ROLE_FIT).to_numpy()
        stop_role = lb.ROLE_EARLY_STOP if args.protocol == lb.PROTOCOL_PHASE6 else lb.ROLE_VALIDATION
        stop_mask = (roles == stop_role).to_numpy()
        validation_mask = (roles == lb.ROLE_VALIDATION).to_numpy()
        counts = {name: int((roles == name).sum()) for name in (lb.ROLE_FIT, lb.ROLE_EARLY_STOP, lb.ROLE_VALIDATION)}
        print(f"  rows by role: {counts}")
        for name in (lb.ROLE_FIT, stop_role):
            mask = (roles == name).to_numpy()
            print(f"  {name:<11} event rate {frame.y[mask].mean():.5f}   "
                  f"{frame.meta.loc[mask, 'prediction_timestamp_utc'].min():%Y-%m-%d} to "
                  f"{frame.meta.loc[mask, 'prediction_timestamp_utc'].max():%Y-%m-%d}")
        print(f"  validation  event rate {frame.y[validation_mask].mean():.5f}")

        masked = lb.mask_unavailable_history(frame.X, frame.meta)
        if masked:
            print(f"  prior-month features set to missing for {masked:,} rows scheduled before 2019-02-01 UTC")

        y_val = frame.y[validation_mask]
        meta_val = frame.meta.loc[validation_mask].reset_index(drop=True)
        if y_validation is None:
            y_validation, days = y_val, meta_val["day"].to_numpy()
            validation_ids = meta_val[ID_COLUMN].to_numpy()
            predictions[ID_COLUMN] = validation_ids
            predictions["prediction_day_utc"] = days
            predictions["label"] = y_val
        elif not np.array_equal(validation_ids, meta_val[ID_COLUMN].to_numpy()):
            fail("validation rows differ between feature sets")

        # Baselines are learned once, from the fit rows of the first feature set.
        if position == 0:
            print("  scoring historical-rate baselines ...")
            baseline_scores = build_rate_baselines(frame.X.loc[fit_mask], frame.y[fit_mask], frame.X.loc[validation_mask])
            for name, scores in baseline_scores.items():
                predictions[name] = scores
                metric_rows.append({"model": name, "feature_set": "n/a", "protocol": args.protocol,
                                    **binary_metrics(y_val, scores, probabilities=True)})
                ranking_frames.append(ranking_table(y_val, scores).assign(model=name))

        fit_levels = lb.apply_fit_categories(frame.X, fit_mask)
        X_fit, y_fit = frame.X.loc[fit_mask], frame.y[fit_mask]
        if args.max_fit_rows and len(y_fit) > args.max_fit_rows:
            chosen = np.sort(np.random.default_rng(args.seed).choice(len(y_fit), args.max_fit_rows, replace=False))
            X_fit, y_fit = X_fit.iloc[chosen], y_fit[chosen]
        X_stop, y_stop = frame.X.loc[stop_mask], frame.y[stop_mask]
        X_val = frame.X.loc[validation_mask]
        frame.X = None  # free the full table; the slices above are all that is needed from here

        print(f"  fitting LightGBM on {len(y_fit):,} rows, early stopping on {len(y_stop):,} rows "
              f"(categorical: {fit_levels}) ...", flush=True)
        fit = lb.fit_lightgbm(X_fit, y_fit, X_stop, y_stop, n_jobs=args.n_jobs, seed=args.seed, params=params)
        scores = lb.predict_scores(fit, X_val)
        print(f"  best iteration {fit.best_iteration}, training time {fit.seconds:,.0f} s")

        name = f"lightgbm_{feature_set}"
        predictions[name] = scores
        metrics = binary_metrics(y_val, scores)
        metric_rows.append({"model": name, "feature_set": feature_set, "protocol": args.protocol,
                            "fit_rows": int(len(y_fit)), "stop_rows": int(len(y_stop)),
                            "best_iteration": fit.best_iteration, "train_seconds": round(fit.seconds, 1), **metrics})
        ranking_frames.append(ranking_table(y_val, scores).assign(model=name))
        calibration_frames.append(calibration_table(y_val, scores).assign(model=name))
        fit_details[name] = {"best_iteration": fit.best_iteration, "train_seconds": fit.seconds,
                             "params": fit.params, "fit_rows": int(len(y_fit)), "stop_rows": int(len(y_stop)),
                             "history_rows_masked": masked}
        print(f"  validation PR-AUC {metrics['pr_auc']:.5f} (prevalence {metrics['prevalence']:.5f}, "
              f"lift {metrics['pr_auc_lift']:.2f}x)  ROC-AUC {metrics['roc_auc']:.4f}")
        del frame, X_fit, X_stop, X_val, roles

    # ------------------------------------------------------------------
    # Results tables
    # ------------------------------------------------------------------
    results = pd.DataFrame(metric_rows)
    ranking = pd.concat(ranking_frames, ignore_index=True)
    top = ranking[ranking["fraction"] == 0.01].set_index("model")
    results["precision_at_1pct"] = results["model"].map(top["precision"])
    results["recall_at_1pct"] = results["model"].map(top["recall"])

    show = ["model", "pr_auc", "pr_auc_lift", "roc_auc", "brier_score", "precision_at_1pct", "recall_at_1pct"]
    print("\nValidation results (2022):")
    print(results[show].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ------------------------------------------------------------------
    # Paired day-level bootstrap
    # ------------------------------------------------------------------
    pairwise = pd.DataFrame()
    lightgbm_names = [f"lightgbm_{s}" for s in args.feature_sets]
    if args.bootstrap > 0:
        baselines = results[results["model"].str.startswith("baseline_") & (results["model"] != "baseline_prevalence")]
        best_baseline = baselines.sort_values("pr_auc", ascending=False)["model"].iloc[0]
        pairs = [(best_baseline, lightgbm_names[0])] + [(lightgbm_names[0], other) for other in lightgbm_names[1:]]
        rows = []
        for reference, challenger in pairs:
            print(f"\nBootstrapping {challenger} minus {reference} ({args.bootstrap} day-level resamples) ...", flush=True)
            outcome = paired_day_bootstrap(y_validation, predictions[reference].to_numpy(),
                                           predictions[challenger].to_numpy(), days, n_boot=args.bootstrap, seed=args.seed)
            rows.append({"reference": reference, "challenger": challenger, **outcome})
        pairwise = pd.DataFrame(rows)
        print(pairwise[["reference", "challenger", "difference", "ci_low", "ci_high", "share_challenger_better"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print("  A difference whose 95% interval includes 0 is not evidence that one model is better.")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    results.to_csv(tables / f"phase6_lightgbm_validation_{suffix}.csv", index=False)
    ranking.to_csv(tables / f"phase6_lightgbm_ranking_validation_{suffix}.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(tables / f"phase6_lightgbm_calibration_validation_{suffix}.csv", index=False)
    if len(pairwise):
        pairwise.to_csv(tables / f"phase6_lightgbm_pairwise_{suffix}.csv", index=False)
    predictions.to_parquet(predictions_dir / f"lightgbm_validation_{suffix}.parquet", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "target": lb.TARGET,
        "protocol": args.protocol,
        "tag": tag,
        "early_stop_start_utc": str(lb.EARLY_STOP_START_UTC),
        "history_available_from_utc": str(lb.HISTORY_AVAILABLE_FROM_UTC),
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "params_source": str(args.params_json) if params else "phase4 defaults (LIGHTGBM_PARAMS)",
        "feature_sets": {s: list(FEATURE_SETS[s]) for s in args.feature_sets},
        "fits": fit_details,
        "review_fractions": list(DEFAULT_FRACTIONS),
        "final_test_used": False,
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / f"phase6_lightgbm_run_manifest_{suffix}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote tables to {tables} (suffix _{suffix}) and predictions to {predictions_dir}")
    print("Final test was not loaded. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

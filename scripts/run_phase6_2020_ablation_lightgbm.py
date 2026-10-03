#!/usr/bin/env python
"""Phase 6, Step 8a: does training on 2020 help or hurt 2022? (LightGBM)

Save as:   scripts/run_phase6_2020_ablation_lightgbm.py
Run from the project root, after the Step 6 tuning run:

    # 1. Smoke run (a few minutes). Tagged "smoke"; numbers are not results.
    python scripts/run_phase6_2020_ablation_lightgbm.py --max-fit-rows 300000 --seeds 1 --bootstrap 20

    # 2. Real run (three variants x three seeds, roughly ten minutes).
    python scripts/run_phase6_2020_ablation_lightgbm.py

Question (DEC-021)
------------------
Scope: "test whether including the 2020 shock improves or harms later generalisation".
Removing 2020 also removes about a third of the fit rows, so a plain "with versus without" test
mixes two effects. Three variants of the fit rows are trained with the same tuned LightGBM settings
(configs/lightgbm_benchmark_selected.json), the same early-stopping rows and the same seeds:

    full      all fit rows (development before 2021-07-01 UTC)
    no2020    the same rows without calendar year 2020 (UTC prediction time)
    matched   a random subset with exactly as many rows as no2020, drawn from all years

All are scored on the 2022 validation rows. The second half of 2022 is the quoted comparison. The rule
that decides whether 2020 is dropped is fixed in DEC-021: no2020 must beat BOTH full and matched, each
with the whole 95% interval above zero. Otherwise 2020 stays in the training data, because the
scope says not to remove 2020 merely because it is unusual.

2023 is never loaded. Nothing here uses an outcome of a validation row for training or stopping.

Writes (tables are small and safe to commit; predictions stay out of Git):
    reports/tables/phase6_ablation2020_lightgbm[_tag]_{fit_composition,runs,summary,comparisons}.csv
    reports/tables/phase6_ablation2020_lightgbm[_tag]_decision.json
    data/processed/phase6/predictions/lightgbm_ablation2020[_tag].parquet
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
    from airline_disruption.deep import protocol as proto
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

PARAMS_PATH = "configs/lightgbm_benchmark_selected.json"
PERIODS = ("2022 second half (confirmation)", "2022 full year")


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_metrics(y, scores)["pr_auc"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 LightGBM 2020 ablation (validation only)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feature-set", default="base_no_year", choices=sorted(FEATURE_SETS))
    parser.add_argument("--params-json", default=PARAMS_PATH, help="settings written by tune_phase6_lightgbm.py")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds per variant (42, 43, ...)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="first seed; its scores are used for the paired comparisons")
    parser.add_argument("--max-fit-rows", type=int, default=None, help="subsample fit rows, smoke runs only")
    parser.add_argument("--bootstrap", type=int, default=200, help="day-level bootstrap resamples (0 to skip)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    if args.seeds < 1:
        fail("--seeds must be at least 1")
    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH, args.params_json):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2, 3 and 6 scripts first)")
    try:
        params = tune.load_params_file(root / args.params_json)
    except (OSError, ValueError) as problem:
        fail(f"cannot use {args.params_json}: {problem}")

    smoke = args.max_fit_rows is not None
    tag = args.tag or ("smoke" if smoke else "")
    suffix = "lightgbm" + (f"_{tag}" if tag else "")
    tables = root / "reports/tables"
    predictions_dir = root / "data/processed/phase6/predictions"
    tables.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    print(f"Feature set: {args.feature_set}   settings: {args.params_json}   seeds: {args.seeds}   tag: {tag or '-'}")
    if smoke:
        print(f"SMOKE RUN: fit rows limited to {args.max_fit_rows:,}. Numbers are not results.")

    # ------------------------------------------------------------------
    # Rows and roles: identical to the benchmark
    # ------------------------------------------------------------------
    try:
        frame = lb.load_model_frame(root, args.feature_set)  # development + validation; the final test cannot load
        roles = lb.assign_roles(frame.meta, lb.PROTOCOL_PHASE6)
        lb.check_role_chronology(frame.meta, roles)
    except ValueError as problem:
        fail(str(problem))
    masked = lb.mask_unavailable_history(frame.X, frame.meta)
    if masked:
        print(f"  prior-month features set to missing for {masked:,} rows scheduled before 2019-02-01 UTC")

    fit_pos = np.flatnonzero((roles == lb.ROLE_FIT).to_numpy())
    stop_pos = np.flatnonzero((roles == lb.ROLE_EARLY_STOP).to_numpy())
    val_pos = np.flatnonzero((roles == lb.ROLE_VALIDATION).to_numpy())
    seconds = ab.to_epoch_seconds(frame.meta["prediction_timestamp_utc"])
    fit_pos = proto.subsample_rows(fit_pos, args.max_fit_rows, args.seed)

    y_stop, y_val = frame.y[stop_pos], frame.y[val_pos]
    meta_val = frame.meta.iloc[val_pos].reset_index(drop=True)
    try:
        selection, confirmation = tune.validation_halves(meta_val["prediction_timestamp_utc"])
    except ValueError as problem:
        fail(str(problem))
    days = meta_val["day"].to_numpy()
    everything = np.ones(len(y_val), dtype=bool)
    print(f"  fit {len(fit_pos):,} rows | early stopping {len(stop_pos):,} | validation {len(val_pos):,}")

    composition = proto.fit_composition(seconds[fit_pos], frame.y[fit_pos])
    print("\nFit rows by UTC year (this is what the ablation removes):")
    print(composition.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    composition.to_csv(tables / f"phase6_ablation2020_{suffix}_fit_composition.csv", index=False)

    # ------------------------------------------------------------------
    # Fit every variant with every seed
    # ------------------------------------------------------------------
    seeds = [args.seed + i for i in range(args.seeds)]
    run_rows: list[dict] = []
    scores: dict[tuple[str, int], np.ndarray] = {}
    for seed in seeds:
        for variant in proto.FIT_VARIANTS:
            try:
                rows = proto.fit_variant_rows(fit_pos, seconds, variant, seed)
            except ValueError as problem:
                fail(str(problem))
            # Category levels come from THIS variant's fit rows only, as in the benchmark.
            order = np.concatenate([rows, stop_pos, val_pos])
            X_variant = frame.X.iloc[order].reset_index(drop=True)
            fit_mask = np.zeros(len(X_variant), dtype=bool)
            fit_mask[: len(rows)] = True
            lb.apply_fit_categories(X_variant, fit_mask)
            n_fit, n_stop = len(rows), len(stop_pos)
            try:
                fit = lb.fit_lightgbm(
                    X_variant.iloc[:n_fit], frame.y[rows], X_variant.iloc[n_fit : n_fit + n_stop], y_stop,
                    n_jobs=args.n_jobs, seed=seed, params=params,
                )
            except ValueError as problem:
                fail(str(problem))
            variant_scores = lb.predict_scores(fit, X_variant.iloc[n_fit + n_stop :])
            del X_variant
            scores[(variant, seed)] = variant_scores
            row = {
                "variant": variant, "seed": seed, "fit_rows": int(n_fit), "fit_events": int(frame.y[rows].sum()),
                "best_iteration": fit.best_iteration, "train_seconds": round(fit.seconds, 1),
                "pr_auc_selection_half": pr_auc(y_val[selection], variant_scores[selection]),
                "pr_auc_confirmation_half": pr_auc(y_val[confirmation], variant_scores[confirmation]),
                "pr_auc_full_year": pr_auc(y_val, variant_scores),
                "roc_auc_full_year": binary_metrics(y_val, variant_scores)["roc_auc"],
            }
            run_rows.append(row)
            print(f"  seed {seed}  {variant:<8} rows {n_fit:>9,}  trees {fit.best_iteration:>4}  "
                  f"PR-AUC selection {row['pr_auc_selection_half']:.5f}  confirmation {row['pr_auc_confirmation_half']:.5f}  "
                  f"({fit.seconds:,.0f} s)", flush=True)
            pd.DataFrame(run_rows).to_csv(tables / f"phase6_ablation2020_{suffix}_runs.csv", index=False)  # keep progress
    runs = pd.DataFrame(run_rows)

    summary = (
        runs.groupby("variant", sort=False)
        .agg(fit_rows=("fit_rows", "mean"), seeds=("seed", "size"),
             confirmation_mean=("pr_auc_confirmation_half", "mean"), confirmation_min=("pr_auc_confirmation_half", "min"),
             confirmation_max=("pr_auc_confirmation_half", "max"),
             full_year_mean=("pr_auc_full_year", "mean"), selection_mean=("pr_auc_selection_half", "mean"))
        .reset_index()
    )
    summary["confirmation_spread"] = summary["confirmation_max"] - summary["confirmation_min"]
    print("\nPR-AUC on 2022, mean over seeds (spread = largest minus smallest confirmation-half value):")
    print(summary[["variant", "fit_rows", "confirmation_mean", "confirmation_spread", "full_year_mean", "selection_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.5f}" if abs(v) < 10 else f"{v:,.0f}"))
    summary.to_csv(tables / f"phase6_ablation2020_{suffix}_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Paired comparisons (first seed) and the DEC-021 rule
    # ------------------------------------------------------------------
    periods = {PERIODS[0]: confirmation, PERIODS[1]: everything}
    pairs = [("no2020", "full"), ("no2020", "matched"), ("matched", "full")]
    comparisons, lower_ends = [], {}
    if args.bootstrap > 0:
        for challenger, reference in pairs:
            for period, mask in periods.items():
                outcome = paired_day_bootstrap(y_val[mask], scores[(reference, seeds[0])][mask], scores[(challenger, seeds[0])][mask],
                                               days[mask], n_boot=args.bootstrap, seed=args.seed)
                sentence = proto.verdict(outcome["difference"], outcome["ci_low"], outcome["ci_high"], challenger, reference)
                comparisons.append({"challenger": challenger, "reference": reference, "period": period, **outcome, "verdict": sentence})
                lower_ends[(challenger, reference, period)] = outcome["ci_low"]
                print(f"\n{challenger} minus {reference}, {period}:\n  {outcome['pr_auc_challenger']:.5f} vs {outcome['pr_auc_reference']:.5f}  "
                      f"difference {outcome['difference']:+.5f}  95% interval [{outcome['ci_low']:+.5f}, {outcome['ci_high']:+.5f}]\n  {sentence}")
        pd.DataFrame(comparisons).to_csv(tables / f"phase6_ablation2020_{suffix}_comparisons.csv", index=False)
        drop = proto.decide_drop_2020(lower_ends[("no2020", "full", PERIODS[0])], lower_ends[("no2020", "matched", PERIODS[0])])
        spread = float(summary.set_index("variant").loc["no2020", "confirmation_spread"])
        print(f"\nDEC-021 rule (second half of 2022): drop 2020 only if no2020 beats BOTH full and matched with the whole interval above zero.")
        print(f"  DECISION: {'drop 2020 from the fit rows' if drop else 'keep 2020 in the fit rows'}")
        print(f"  (seed spread of the no2020 model on this half: {spread:.5f}; a difference of that size or smaller is within seed noise)")
    else:
        drop = False
        print("\nBootstrap skipped: 2020 stays in the fit rows.")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    predictions = pd.DataFrame({ID_COLUMN: meta_val[ID_COLUMN].to_numpy(), "prediction_day_utc": days, "label": y_val})
    for (variant, seed), values in scores.items():
        predictions[f"lightgbm_{variant}_s{seed}"] = values
    predictions.to_parquet(predictions_dir / f"lightgbm_ablation2020{'_' + tag if tag else ''}.parquet", index=False)

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "feature_set": args.feature_set,
        "protocol": lb.PROTOCOL_PHASE6,
        "tag": tag,
        "smoke_run": smoke,
        "params_source": args.params_json,
        "params": params,
        "variants": list(proto.FIT_VARIANTS),
        "shock_year": proto.SHOCK_YEAR,
        "seeds": seeds,
        "comparison_seed": seeds[0],
        "fit_rows_by_utc_year": {int(r.year): int(r.rows) for r in composition.itertuples()},
        "comparisons": comparisons,
        "drop_2020_adopted": bool(drop),
        "rule": "DEC-021: no2020 must beat both full and matched on the second half of 2022, each with the whole 95% interval above zero",
        "final_test_used": False,
        "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / f"phase6_ablation2020_{suffix}_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote tables in {tables} (files phase6_ablation2020_{suffix}_*) and predictions in {predictions_dir}.")
    print(f"Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 6, Step 6: tune the LightGBM benchmark under a pre-registered protocol.

Save as:   scripts/tune_phase6_lightgbm.py
Run from the project root, after the Step 2, 3 and 5 build scripts:

    # 1. Quick smoke run (a few minutes). Tagged "smoke"; writes no selection file.
    #    (A run with --tag also writes no selection file: only the plain full run does.)
    python scripts/tune_phase6_lightgbm.py --n-configs 4 --max-fit-rows 300000 --bootstrap 20

    # 2. Real run: default settings plus 30 random draws.
    python scripts/tune_phase6_lightgbm.py

The protocol is in airline_disruption/models/lightgbm_tuning.py and was fixed before any
result existed:

    SELECT    the challenger = best PR-AUC on the FIRST half of 2022.
    CONFIRM   on the SECOND half of 2022, with a paired day-level bootstrap against the
              default. The challenger is adopted only if the 95% interval is above zero.
    2023      never loaded.

Why two halves: choosing the best of 31 configurations on one set of rows and then quoting
that set's score overstates the gain. The second half played no part in the choice.

Writes:
    reports/tables/phase6_lightgbm_tuning_<feature_set>[_<tag>].csv      all configurations
    reports/tables/phase6_lightgbm_tuning_decision_<feature_set>[_<tag>].json
    configs/lightgbm_benchmark_selected.json      the settings later steps use (plain full run only)
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
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

CHECK_SEEDS = (1, 2)  # extra seeds for the noise check on the default and the challenger


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_metrics(y, scores)["pr_auc"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 LightGBM tuning (select on 2022 H1, confirm on H2)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feature-set", default="base_no_year", choices=sorted(FEATURE_SETS))
    parser.add_argument("--n-configs", type=int, default=30, help="random draws besides the default settings")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fit-rows", type=int, default=None, help="subsample fit rows, smoke runs only")
    parser.add_argument("--bootstrap", type=int, default=200, help="day-level bootstrap resamples for the confirmation")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2 and Step 3 build scripts first)")

    smoke = args.max_fit_rows is not None
    official = not smoke and not args.tag  # only an untagged, full run may write the selection file
    tag = args.tag or ("smoke" if smoke else "")
    suffix = args.feature_set + (f"_{tag}" if tag else "")
    tables = root / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load once. Fit rows, early-stopping rows and 2022 rows, as in Step 4.
    # ------------------------------------------------------------------
    print(f"Feature set: {args.feature_set}   configurations: default + {args.n_configs} draws   tag: {tag or '-'}")
    if smoke:
        print(f"SMOKE RUN: fit rows limited to {args.max_fit_rows:,}. Numbers are not results and no selection file is written.")
    try:
        frame = lb.load_model_frame(root, args.feature_set)
        roles = lb.assign_roles(frame.meta, lb.PROTOCOL_PHASE6)
        lb.check_role_chronology(frame.meta, roles)
    except ValueError as problem:
        fail(str(problem))

    fit_mask = (roles == lb.ROLE_FIT).to_numpy()
    stop_mask = (roles == lb.ROLE_EARLY_STOP).to_numpy()
    validation_mask = (roles == lb.ROLE_VALIDATION).to_numpy()
    masked = lb.mask_unavailable_history(frame.X, frame.meta)
    if masked:
        print(f"  prior-month features set to missing for {masked:,} rows scheduled before 2019-02-01 UTC")
    lb.apply_fit_categories(frame.X, fit_mask)

    X_fit, y_fit = frame.X.loc[fit_mask], frame.y[fit_mask]
    if smoke and len(y_fit) > args.max_fit_rows:
        chosen = np.sort(np.random.default_rng(args.seed).choice(len(y_fit), args.max_fit_rows, replace=False))
        X_fit, y_fit = X_fit.iloc[chosen], y_fit[chosen]
    X_stop, y_stop = frame.X.loc[stop_mask], frame.y[stop_mask]
    X_val, y_val = frame.X.loc[validation_mask], frame.y[validation_mask]
    meta_val = frame.meta.loc[validation_mask].reset_index(drop=True)
    frame.X = None  # free the full table

    try:
        selection, confirmation = tune.validation_halves(meta_val["prediction_timestamp_utc"])
    except ValueError as problem:
        fail(str(problem))
    days = meta_val["day"].to_numpy()
    print(f"  fit {len(y_fit):,} rows | early stopping {len(y_stop):,} | validation {len(y_val):,} "
          f"(selection half {int(selection.sum()):,}, confirmation half {int(confirmation.sum()):,})")
    print(f"  event rate: selection half {y_val[selection].mean():.5f}, confirmation half {y_val[confirmation].mean():.5f}")

    # ------------------------------------------------------------------
    # Fit every configuration
    # ------------------------------------------------------------------
    configs = tune.sample_configs(args.n_configs, args.seed)
    rows: list[dict] = []
    scores_by_config: dict[int, np.ndarray] = {}
    for number, overrides in enumerate(configs):
        label = "default" if number == 0 else f"draw {number}"
        params = tune.params_for(overrides)
        fit = lb.fit_lightgbm(X_fit, y_fit, X_stop, y_stop, n_jobs=args.n_jobs, seed=args.seed, params=params)
        scores = lb.predict_scores(fit, X_val)
        scores_by_config[number] = scores
        row = {
            "config": number,
            "label": label,
            "best_iteration": fit.best_iteration,
            "train_seconds": round(fit.seconds, 1),
            "pr_auc_selection_half": pr_auc(y_val[selection], scores[selection]),
            "pr_auc_confirmation_half": pr_auc(y_val[confirmation], scores[confirmation]),
            "pr_auc_full_year": pr_auc(y_val, scores),
            "overrides": json.dumps(overrides),
        }
        rows.append(row)
        print(f"  [{number:>2}/{args.n_configs}] {label:<8} trees {fit.best_iteration:>4}  "
              f"PR-AUC selection {row['pr_auc_selection_half']:.5f}  confirmation {row['pr_auc_confirmation_half']:.5f}  "
              f"({fit.seconds:,.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(tables / f"phase6_lightgbm_tuning_{suffix}.csv", index=False)  # keep partial progress

    results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Select on the first half, confirm on the second
    # ------------------------------------------------------------------
    best = tune.best_config_index(results["pr_auc_selection_half"].to_numpy())
    print(f"\nChallenger (best selection-half PR-AUC): config {best}  {configs[best]}")
    default_selection, challenger_selection = (results.loc[i, "pr_auc_selection_half"] for i in (0, best))
    print(f"  selection half:    default {default_selection:.5f}   challenger {challenger_selection:.5f}")

    outcome = None
    adopt = False
    if args.bootstrap > 0:
        outcome = paired_day_bootstrap(
            y_val[confirmation],
            scores_by_config[0][confirmation],
            scores_by_config[best][confirmation],
            days[confirmation],
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        adopt = tune.decide_adoption(outcome["ci_low"])
        print(f"  confirmation half: default {outcome['pr_auc_reference']:.5f}   challenger {outcome['pr_auc_challenger']:.5f}   "
              f"difference {outcome['difference']:+.5f}  95% interval [{outcome['ci_low']:+.5f}, {outcome['ci_high']:+.5f}]")
    else:
        print("  bootstrap skipped: the default is kept.")
    print(f"  DECISION: {'adopt the challenger' if adopt else 'keep the default settings'}")

    # ------------------------------------------------------------------
    # Seed noise check: how much does PR-AUC move with the seed alone?
    # ------------------------------------------------------------------
    print("\nSeed noise check (confirmation-half PR-AUC, same settings, different seeds):")
    noise = {}
    for name, index in (("default", 0), ("challenger", best)):
        values = [results.loc[index, "pr_auc_confirmation_half"]]
        for seed in CHECK_SEEDS:
            fit = lb.fit_lightgbm(X_fit, y_fit, X_stop, y_stop, n_jobs=args.n_jobs, seed=seed, params=tune.params_for(configs[index]))
            values.append(pr_auc(y_val[confirmation], lb.predict_scores(fit, X_val)[confirmation]))
        noise[name] = values
        print(f"  {name:<10} seeds {[args.seed, *CHECK_SEEDS]}: " + "  ".join(f"{v:.5f}" for v in values)
              + f"   spread {max(values) - min(values):.5f}")

    if max(noise["default"]) - min(noise["default"]) == 0.0:
        print("  (the default settings use no row or column sampling, so the seed cannot change them)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    chosen_overrides = configs[best] if adopt else {}
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "feature_set": args.feature_set,
        "protocol": lb.PROTOCOL_PHASE6,
        "tag": tag,
        "smoke_run": smoke,
        "n_random_draws": args.n_configs,
        "seed": args.seed,
        "search_space": tune.SEARCH_SPACE,
        "selection_period": "2022, prediction time before 2022-07-01 UTC",
        "confirmation_period": "2022, prediction time from 2022-07-01 UTC",
        "challenger_config": int(best),
        "challenger_overrides": configs[best],
        "challenger_adopted": bool(adopt),
        "confirmation_bootstrap": outcome,
        "seed_noise_confirmation_half": {k: [float(v) for v in values] for k, values in noise.items()},
        "params": tune.params_for(chosen_overrides),
        "final_test_used": False,
        "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / f"phase6_lightgbm_tuning_decision_{suffix}.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    if official:
        (root / "configs").mkdir(exist_ok=True)
        (root / "configs/lightgbm_benchmark_selected.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote configs/lightgbm_benchmark_selected.json (params used by later steps: "
              f"{'tuned challenger' if adopt else 'default settings'})")
    print(f"Wrote tables in {tables} (suffix _{suffix}). Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

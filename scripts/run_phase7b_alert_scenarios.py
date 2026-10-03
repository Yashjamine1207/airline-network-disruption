#!/usr/bin/env python
"""Phase 7B, Step 11: retrospective ranked analyst-review scenarios (validation years only).

Save as:   scripts/run_phase7b_alert_scenarios.py
Run from the project root:

    # 1. Smoke run (about a minute). Tagged "smoke"; numbers are not results.
    python scripts/run_phase7b_alert_scenarios.py --max-fit-rows 300000 --years 2022

    # 2. Severe delay, the real run (a few minutes on 16 cores).
    python scripts/run_phase7b_alert_scenarios.py

    # 3. Cancellation (same model settings, NOT tuned for this target; read the warning it prints).
    python scripts/run_phase7b_alert_scenarios.py --target cancelled

The question
------------
If an analyst could review only the top 0.5%, 1%, 5% or 10% of flights by model score, how many of the events
would be among them, and how many reviewed flights would be false alerts? This is a retrospective reading of
a ranking. It is not a system for running an airline, and nothing here uses 2023.

What it does
------------
* Backtests the locked LightGBM recipe (same settings as the benchmark) in expanding windows, so every
  evaluation year is scored by a model that never saw it:
      2020: fit 2019-01 to 2019-06, early stopping 2019-07 to 2019-12
      2021: fit 2019-01 to 2020-06, early stopping 2020-07 to 2020-12
      2022: fit 2019-01 to 2021-06, early stopping 2021-07 to 2021-12   (the locked model; it must match the
            stored scores for severe delay)
  2019 cannot be evaluated (no earlier data). Each year is a separate model; do not compare the numbers
  as if one model were tested on three years.
* Applies each capacity three ways: over the whole year, within each month, within each day (see
  policy/review.py). The ranking score is the UNCALIBRATED model score. No probability threshold is used.
* Day-level bootstrap intervals for precision and recall.
* For 2022: results by quarter, carrier, origin airport, scheduled departure hour, missing or unseen values
  (global cutoffs, as in the error analysis).
* An ILLUSTRATIVE penalty table: the assumed cost of a missed event relative to a false alert, at several
  ratios, plus the break-even ratio computed from the measured counts. Weights are assumptions, never airline
  costs.

Writes (small tables, safe to commit):
    reports/tables/phase7b_<target>_backtest_models.csv
    reports/tables/phase7b_<target>_review_capacity.csv
    reports/tables/phase7b_<target>_review_by_group_2022.csv
    reports/tables/phase7b_<target>_utility.csv
    reports/tables/phase7b_<target>_decision.json
    reports/figures/review_capacity_<target>.png
(a smoke run adds "_smoke" to every file name)
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
    from airline_disruption.evaluation.classification import binary_metrics
    from airline_disruption.explain import errors, plots
    from airline_disruption.explain.dimensions import build_dimensions
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.policy import review
    from airline_disruption.policy.windows import expanding_window_roles
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics  # noqa: E402
    from airline_disruption.explain import errors, plots  # noqa: E402
    from airline_disruption.explain.dimensions import build_dimensions  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.policy import review  # noqa: E402
    from airline_disruption.policy.windows import expanding_window_roles  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

TARGETS = {
    "severe_delay_120": {"eligibility": "eligible_severe_delay", "label": "Severe delay (arrival delay of 120 minutes or more)"},
    "cancelled": {"eligibility": "eligible_cancellation", "label": "Cancellation"},
}
REGIMES = {2020: "2020 shock", 2021: "2021 recovery", 2022: "2022 recovery/transition (locked model)"}
PARAMS_FILE = "configs/lightgbm_benchmark_selected.json"
REFERENCE_PATH = "data/processed/phase6/predictions/lightgbm_validation_phase6_tuned_ladder.parquet"
REFERENCE_COLUMN = "lightgbm_base_no_year"
REPRODUCTION_TOLERANCE = 0.0005
DROP_FROM_GROUP_TABLE = ["mean_probability", "probability_to_observed", "brier_probability"]  # rankings only in this step


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def fit_window(frame, year: int, params: dict, seed: int, n_jobs: int, max_fit_rows: int | None) -> dict:
    """Fit on everything before the early-stopping months of ``year`` and score ``year``."""
    stamps = frame.meta["prediction_timestamp_utc"]
    roles = expanding_window_roles(stamps, year)
    fit_pos = proto.subsample_rows(np.flatnonzero(roles == "fit"), max_fit_rows, seed)
    stop_pos, eval_pos = np.flatnonzero(roles == "early_stop"), np.flatnonzero(roles == "eval")
    order = np.concatenate([fit_pos, stop_pos, eval_pos])
    X = frame.X.iloc[order].reset_index(drop=True)
    fit_mask = np.zeros(len(X), dtype=bool)
    fit_mask[: len(fit_pos)] = True
    lb.apply_fit_categories(X, fit_mask)
    n_fit, n_stop = len(fit_pos), len(stop_pos)
    fit = lb.fit_lightgbm(X.iloc[:n_fit], frame.y[fit_pos], X.iloc[n_fit : n_fit + n_stop], frame.y[stop_pos],
                          n_jobs=n_jobs, seed=seed, params=params)
    X_eval = X.iloc[n_fit + n_stop :]
    scores = lb.predict_scores(fit, X_eval)
    meta_eval = frame.meta.iloc[eval_pos].reset_index(drop=True)
    return {
        "year": year, "roles": roles, "fit": fit, "X_fit": X.iloc[:n_fit], "X_eval": X_eval.reset_index(drop=True), "scores": scores,
        "y": frame.y[eval_pos], "meta": meta_eval, "n_fit": n_fit, "n_stop": n_stop, "ids": meta_eval["source_row_number"].to_numpy(),
        "fit_span": (stamps.iloc[fit_pos].min(), stamps.iloc[fit_pos].max()), "stop_span": (stamps.iloc[stop_pos].min(), stamps.iloc[stop_pos].max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7B ranked-review scenarios (validation years only)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target", default="severe_delay_120", choices=sorted(TARGETS))
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022], help="evaluation years, each at most 2022")
    parser.add_argument("--feature-set", default="base_no_year", choices=sorted(FEATURE_SETS))
    parser.add_argument("--params-file", default=PARAMS_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500, help="day-level bootstrap resamples")
    parser.add_argument("--max-fit-rows", type=int, default=None, help="subsample fit rows, smoke runs only")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    if any(y > 2022 or y < 2020 for y in args.years):
        fail("evaluation years must be between 2020 and 2022 (2019 has no earlier data; 2023 is the locked final test)")
    target = args.target
    cfg = TARGETS[target]
    smoke = args.max_fit_rows is not None
    tag = args.tag or ("smoke" if smoke else "")
    infix = f"_{tag}" if tag else ""
    tables, figures = root / "reports/tables", root / "reports/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print(f"Target: {cfg['label']} ({target}). Feature set {args.feature_set}, seed {args.seed}, tag: {tag or '-'}")
    if smoke:
        print("SMOKE RUN: reduced fit rows. Numbers are not results.")
    if target != "severe_delay_120":
        print("WARNING: the LightGBM settings were tuned for severe delay, not for this target. Treat these results as a\n"
              "         baseline, not as a tuned cancellation model.")
    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH, args.params_file):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}")
    try:
        params = tune.load_params_file(root / args.params_file)
        frame = lb.load_model_frame(root, args.feature_set, target=target, eligibility_column=cfg["eligibility"])  # development + 2022 only
    except (OSError, ValueError) as problem:
        fail(str(problem))
    lb.mask_unavailable_history(frame.X, frame.meta)
    stamps = frame.meta["prediction_timestamp_utc"]
    if (stamps >= pd.Timestamp("2023-01-01", tz="UTC")).any():
        fail("rows from 2023 are in the frame")
    print(f"  {len(frame.y):,} eligible rows from {stamps.min().date()} to {stamps.max().date()}, event rate {frame.y.mean():.4%}")

    # ------------------------------------------------------------------
    # One model per evaluation year
    # ------------------------------------------------------------------
    results, model_rows = {}, []
    for year in args.years:
        print(f"\n=== evaluation year {year}: {REGIMES[year]} ===", flush=True)
        try:
            outcome = fit_window(frame, year, params, args.seed, args.n_jobs, args.max_fit_rows)
        except ValueError as problem:
            fail(str(problem))
        results[year] = outcome
        y, s = outcome["y"], outcome["scores"]
        metrics = binary_metrics(y, s, probabilities=False)
        print(f"  fit {outcome['n_fit']:,} rows ({outcome['fit_span'][0].date()} to {outcome['fit_span'][1].date()}), early stopping {outcome['n_stop']:,} "
              f"({outcome['stop_span'][0].date()} to {outcome['stop_span'][1].date()}), {outcome['fit'].best_iteration} trees ({outcome['fit'].seconds:,.0f} s)")
        print(f"  evaluation {len(y):,} rows, {int(y.sum()):,} events ({y.mean():.3%}); PR-AUC {metrics['pr_auc']:.5f} (lift {metrics['pr_auc_lift']:.2f}), ROC-AUC {metrics['roc_auc']:.4f}")
        model_rows.append({
            "target": target, "eval_year": year, "regime": REGIMES[year], "fit_rows": outcome["n_fit"],
            "fit_from": outcome["fit_span"][0], "fit_to": outcome["fit_span"][1], "early_stop_rows": outcome["n_stop"],
            "trees": outcome["fit"].best_iteration, "eval_rows": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
            "pr_auc": metrics["pr_auc"], "pr_auc_lift": metrics["pr_auc_lift"], "roc_auc": metrics["roc_auc"], "fit_seconds": round(outcome["fit"].seconds, 1),
        })

    reproduction = {"checked": False}
    if 2022 in results:
        locked = lb.assign_roles(frame.meta, lb.PROTOCOL_PHASE6).to_numpy()
        mine = np.where(results[2022]["roles"] == "eval", "validation", results[2022]["roles"])
        if not (mine == locked).all():
            fail("the 2022 expanding window differs from the locked Phase 6 roles")
        if target == "severe_delay_120" and not smoke and (root / REFERENCE_PATH).exists():
            stored = proto.align_scores(results[2022]["ids"], results[2022]["y"], pd.read_parquet(root / REFERENCE_PATH), REFERENCE_COLUMN)
            difference = abs(binary_metrics(results[2022]["y"], results[2022]["scores"], probabilities=False)["pr_auc"]
                             - binary_metrics(results[2022]["y"], stored, probabilities=False)["pr_auc"])
            reproduction = {"checked": True, "pr_auc_difference": float(difference), "largest_score_difference": float(np.max(np.abs(results[2022]["scores"] - stored)))}
            print(f"\nCheck against the stored 2022 scores: PR-AUC difference {difference:.6f}")
            if difference > REPRODUCTION_TOLERANCE:
                fail("the 2022 model does not reproduce the stored validation scores")

    # ------------------------------------------------------------------
    # Review capacity: whole year, by month, by day
    # ------------------------------------------------------------------
    capacity_frames, utility_frames = [], []
    for year, outcome in results.items():
        meta = outcome["meta"]
        day_codes = meta["day"].to_numpy()
        month_codes = meta["prediction_timestamp_utc"].dt.strftime("%Y-%m").to_numpy()
        table = review.top_k_review(outcome["y"], outcome["scores"], review.DEFAULT_CAPACITIES, month_codes, day_codes, n_boot=args.bootstrap, seed=args.seed)
        table.insert(0, "eval_year", year)
        table.insert(0, "target", target)
        capacity_frames.append(table)
        utility = review.utility_table(table)
        utility.insert(0, "eval_year", year)
        utility.insert(0, "target", target)
        utility_frames.append(utility)
    capacity = pd.concat(capacity_frames, ignore_index=True)
    utility = pd.concat(utility_frames, ignore_index=True)
    capacity.to_csv(tables / f"phase7b_{target}{infix}_review_capacity.csv", index=False)
    utility.to_csv(tables / f"phase7b_{target}{infix}_utility.csv", index=False)
    pd.DataFrame(model_rows).to_csv(tables / f"phase7b_{target}{infix}_backtest_models.csv", index=False)

    show = ["capacity", "reviewed", "events_captured", "false_alerts", "precision", "precision_low", "precision_high", "recall", "recall_low", "recall_high", "lift", "break_even_ratio"]
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    for year in results:
        print(f"\n--- {year}: review scenarios (ranking by the uncalibrated model score; intervals resample whole days) ---")
        for scheme in review.SCHEMES:
            part = capacity[(capacity["eval_year"] == year) & (capacity["scheme"] == scheme)]
            print(f"\n{scheme.replace('_', ' ')} capacity ({int(part['events'].iloc[0]):,} events in {int(part['flights'].iloc[0]):,} flights, prevalence {part['prevalence'].iloc[0]:.3%}):")
            print(part[show].to_string(index=False, float_format=fmt))

    focus = 2022 if 2022 in results else max(results)
    whole = capacity[(capacity["eval_year"] == focus) & (capacity["scheme"] == "whole_year")]
    print(f"\n--- Illustrative penalty reduction, {focus}, whole-year ranking ---")
    print("ASSUMPTION ONLY: a missed event costs `ratio` false alerts. Not an airline cost. Positive means reviewing pays off versus no review.")
    pivot = utility[(utility["eval_year"] == focus) & (utility["scheme"] == "whole_year")].pivot(index="capacity", columns="miss_to_false_alert_ratio", values="net_reduction_units")
    pivot = pivot.reindex([review.capacity_name(f) for f in review.DEFAULT_CAPACITIES])
    print(pivot.to_string(float_format=lambda v: f"{v:,.0f}"))
    print("Break-even ratio (measured, no assumption): a missed event must be worth this many false alerts for the review to pay off.")
    print(whole[["capacity", "break_even_ratio"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # ------------------------------------------------------------------
    # Where the focus year's alerts land: quarter, carrier, airport, hour, missingness (global cutoffs)
    # ------------------------------------------------------------------
    outcome = results[focus]
    y, s = outcome["y"], outcome["scores"]
    quarter = outcome["meta"]["prediction_timestamp_utc"].dt.quarter.to_numpy()
    cutoffs = errors.top_fraction_cutoffs(s, review.DEFAULT_CAPACITIES)
    frame_eval = pd.DataFrame({"label": y, "score": s, "probability": s})
    groups = [errors.group_metrics(frame_eval.assign(group=pd.Series(labels).to_numpy()), name, cutoffs, "group")
              for name, labels in build_dimensions(outcome["X_eval"], outcome["X_fit"], s, [f"{focus} Q{q}" for q in quarter], period_name=f"quarter_{focus}").items()
              if name != "score_decile"]
    by_group = pd.concat([errors.group_metrics(frame_eval.assign(group=f"all {focus} flights"), "overall", cutoffs, "group"), *groups], ignore_index=True)
    by_group = by_group.drop(columns=DROP_FROM_GROUP_TABLE)
    by_group.insert(0, "target", target)
    by_group.to_csv(tables / f"phase7b_{target}{infix}_review_by_group_{focus}.csv", index=False)
    wide = ["group", "rows", "events", "prevalence", "alert_rate_top1pct", "precision_top1pct", "recall_top1pct", "alert_rate_top10pct", "precision_top10pct", "recall_top10pct"]
    print(f"\n--- {focus}: where the alerts land (one global cutoff per capacity over the whole year) ---")
    print("Top-1% cutoff {:.4f}, top-10% cutoff {:.4f}. alert_rate is the share of the group's flights that would be reviewed.".format(cutoffs[0.01], cutoffs[0.10]))
    for dimension in (f"quarter_{focus}", "scheduled_departure_local_hours", "missing_or_unseen_values", "carrier"):
        part = by_group[by_group["dimension"] == dimension]
        print(f"\n{dimension}:")
        print(part[wide].to_string(index=False, float_format=fmt))
    airports = by_group[by_group["dimension"] == "origin_airport_top25"]
    print("\norigin airports: highest and lowest share of flights reviewed at the top 1% capacity")
    print(pd.concat([airports.nlargest(3, "alert_rate_top1pct"), airports.nsmallest(3, "alert_rate_top1pct")])[wide].to_string(index=False, float_format=fmt))

    plots.review_figure(figures / f"review_capacity_{target}{infix}.png", capacity,
                        utility[(utility["eval_year"] == focus) & (utility["scheme"] == "whole_year")], cfg["label"], focus_year=focus)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "smoke_run": smoke, "target": target,
        "feature_version": FEATURE_VERSION, "feature_set": args.feature_set, "seed": args.seed, "params_file": args.params_file,
        "params_tuned_for_this_target": target == "severe_delay_120",
        "ranking_score": "uncalibrated LightGBM probability (Platt is monotone, so the ranking is identical); no probability threshold used",
        "capacities": [review.capacity_name(f) for f in review.DEFAULT_CAPACITIES],
        "schemes": {"whole_year": "top K% of every flight in the year", "by_month": "top K% within each month", "by_day": "top K% within each UTC prediction day"},
        "rule": "flights reviewed in a group of n = ceil(K * n), at least 1; ties by row order",
        "windows": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in row.items()} for row in model_rows],
        "reproduction_check_2022": reproduction,
        "bootstrap": {"unit": "whole UTC prediction days", "resamples": args.bootstrap, "interval": "95% percentile"},
        "penalty_table": {"status": "ILLUSTRATIVE ASSUMPTIONS ONLY, not airline costs", "ratios_miss_to_false_alert": list(review.DEFAULT_RATIOS),
                          "break_even_ratio": "false alerts divided by events captured, computed from measured counts"},
        "not_evaluable": ["2019 (no earlier data for an expanding window)", "2023 (locked final test, never loaded)",
                          "airport recovery episodes and regression errors (not part of this script)"],
        "limits": ["each year is a different model trained on less data for earlier years; years are not one model tested three times",
                   "2022 was used to choose LightGBM settings and the calibration method, so it is validation, not a clean test",
                   "no cost of reviewing a flight, no cost of a missed event and no action taken after a review is modelled"],
        "final_test_used": False, "total_seconds": round(time.perf_counter() - started, 1), "hardware": hardware_context(), "packages": package_versions(),
    }
    (tables / f"phase7b_{target}{infix}_decision.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote tables in {tables} and the figure in {figures}. Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 7A, Step 9b: calibration of the tuned LightGBM (validation only).

Save as:   scripts/run_phase6_calibration.py
Run from the project root, after the Step 6 tuning run (it needs configs/lightgbm_benchmark_selected.json):

    # 1. Smoke run (a few minutes). Tagged "smoke"; numbers are not results and no selection file is written.
    python scripts/run_phase6_calibration.py --max-fit-rows 300000 --bootstrap 50

    # 2. Real run (about one minute of LightGBM fitting plus the bootstraps).
    python scripts/run_phase6_calibration.py

Question (DEC-023)
------------------
The model's probabilities were never checked against reality. This script measures that and tests
whether Platt (sigmoid) or isotonic calibration improves it. Nothing is tuned on 2023 and 2023 is never loaded.

Experiment A (decides the method)
    fit the calibrators on the early-stopping rows (2021-07-01 to 2021-12-31 UTC), which come after every
    fit row and before every 2022 row. Score 2022. Candidates are chosen on the first half of 2022 (Brier
    score); the choice is confirmed on the second half with a paired day-level bootstrap. Quarter-by-quarter
    results for 2022 are descriptive.
Experiment B (diagnostic only)
    fit the calibrators on the first half of 2022 and score the second half. If that is clearly better than
    Experiment A on the same rows, recency matters and the final-test calibrator will be fitted on the
    most recent labelled year before the test.

The rules are in src/airline_disruption/calibration/decision.py and configs/calibration.yaml.

What this can and cannot show
-----------------------------
* Regimes: 2019 and 2020 are training years for this model, so they cannot be scored on fresh data here.
  The regime evidence is 2022 by quarter. Two other regimes need refitted models and are not attempted.
* The early-stopping rows chose the number of trees, so the model's scores on them are slightly optimistic.
  That is a known small overlap between the calibration rows and model selection (not the tree weights).
* Ranking: Platt never changes the order of flights. Isotonic can. Rankings later on say which scores they use.

Writes (tables are small and safe to commit; predictions stay out of Git):
    reports/tables/calibration_summary.csv                      (plain run; a tagged run writes phase6_calibration_<tag>_summary.csv)
    reports/tables/phase6_calibration[_tag]_{reliability,comparisons}.csv
    reports/tables/phase6_calibration[_tag]_decision.json
    reports/figures/calibration_curve[_tag].png
    configs/calibration_selected.json                            (plain run only)
    data/processed/phase6/predictions/lightgbm_calibrated_validation[_tag].parquet
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
    from airline_disruption.calibration import decision as rules
    from airline_disruption.calibration import guards
    from airline_disruption.calibration.config import load_calibration_config
    from airline_disruption.calibration.experiment import fit_and_apply, metrics_table
    from airline_disruption.calibration.metrics import paired_day_bootstrap_brier, reliability_table
    from airline_disruption.deep import protocol as proto
    from airline_disruption.evaluation.classification import binary_metrics
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.calibration import decision as rules  # noqa: E402
    from airline_disruption.calibration import guards  # noqa: E402
    from airline_disruption.calibration.config import load_calibration_config  # noqa: E402
    from airline_disruption.calibration.experiment import fit_and_apply, metrics_table  # noqa: E402
    from airline_disruption.calibration.metrics import paired_day_bootstrap_brier, reliability_table  # noqa: E402
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

CONFIG_PATH = "configs/calibration.yaml"
SELECTION, CONFIRMATION, FULL_YEAR = "2022 first half (selection)", "2022 second half (confirmation)", "2022 full year"
CALIBRATED_METHODS = ("platt", "isotonic")


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def show_metrics(table: pd.DataFrame, title: str) -> None:
    columns = ["method", "period", "rows", "events", "prevalence", "mean_predicted", "predicted_to_observed", "brier",
               "brier_skill", "calibration_slope", "calibration_in_the_large", "ece", "pr_auc", "distinct_scores"]
    print(f"\n{title}")
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.5f}" if abs(v) < 100 else f"{v:,.0f}"))


def draw_figure(path: Path, y_h2, probs_a: dict, brier_h2: dict, quarter_tables: dict) -> bool:
    """Reliability figure. Returns False (and prints why) if matplotlib is missing; the run does not fail."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib is not installed, so the figure was skipped.")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    styles = {"uncalibrated": ("#444444", "o", "Uncalibrated"), "platt": ("#1f77b4", "s", "Platt"), "isotonic": ("#d62728", "^", "Isotonic")}

    ax = axes[0]
    top = 0.0
    for method, p in probs_a.items():
        color, marker, label = styles[method]
        table = reliability_table(y_h2, p, 10)
        top = max(top, float(table[["mean_predicted", "observed_rate"]].to_numpy().max()))
        ax.plot(table["mean_predicted"], table["observed_rate"], marker=marker, color=color, lw=1.4, ms=5,
                label=f"{label} (Brier {brier_h2[method]:.5f})")
    limit = top * 1.12
    ax.plot([0, limit], [0, limit], "--", color="#999999", lw=1, label="Perfect calibration")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean predicted probability (10 equal-count bins)",
           ylabel="Observed severe-delay rate", title="Second half of 2022: calibrators fitted on 2021 H2 rows")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    quarter_colors = ["#8c564b", "#2ca02c", "#ff7f0e", "#9467bd"]
    top = 0.0
    for (quarter, table), color in zip(quarter_tables.items(), quarter_colors):
        top = max(top, float(table[["mean_predicted", "observed_rate"]].to_numpy().max()))
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", ms=4, lw=1.3, color=color, label=quarter)
    limit = top * 1.12
    ax.plot([0, limit], [0, limit], "--", color="#999999", lw=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean predicted probability (10 equal-count bins)",
           ylabel="Observed severe-delay rate", title="Uncalibrated scores by quarter of 2022")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("Severe delay (arrival delay >= 120 min), tuned LightGBM. Validation year only; 2023 is not used.", fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7A calibration of the tuned LightGBM (validation only)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=None, help="override bootstrap.resamples, smoke runs only")
    parser.add_argument("--max-fit-rows", type=int, default=None, help="subsample fit rows, smoke runs only")
    parser.add_argument("--reference-predictions", default=None, help="override model.reference_predictions")
    parser.add_argument("--reference-column", default=None, help="override model.reference_column")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    try:
        config = load_calibration_config(root / args.config)
    except ValueError as problem:
        fail(str(problem))
    model_cfg, boot_cfg = config["model"], config["bootstrap"]
    methods = list(config["methods"])
    n_bins = int(config["metrics"]["reliability_bins"])
    feature_set = model_cfg["feature_set"]
    if feature_set not in FEATURE_SETS:
        fail(f"unknown feature set {feature_set!r}")
    seed = int(model_cfg["seed"])
    n_boot = int(args.bootstrap if args.bootstrap is not None else boot_cfg["resamples"])
    boot_seed, confidence = int(boot_cfg["seed"]), float(boot_cfg["confidence"])
    reference_path = args.reference_predictions or model_cfg["reference_predictions"]
    reference_column = args.reference_column or model_cfg["reference_column"]

    smoke = args.max_fit_rows is not None or args.bootstrap is not None
    tag = args.tag or ("smoke" if smoke else "")
    official = not smoke and not args.tag
    infix = f"_{tag}" if tag else ""
    tables = root / "reports/tables"
    figures = root / "reports/figures"
    predictions_dir = root / "data/processed/phase6/predictions"
    for folder in (tables, figures, predictions_dir):
        folder.mkdir(parents=True, exist_ok=True)

    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH, model_cfg["params_file"]):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2, 3 and 6 scripts first)")
    try:
        params = tune.load_params_file(root / model_cfg["params_file"])
    except (OSError, ValueError) as problem:
        fail(f"cannot use {model_cfg['params_file']}: {problem}")
    print(f"Model: tuned LightGBM, feature set {feature_set}, seed {seed}   methods: {', '.join(methods)}   tag: {tag or '-'}")
    if smoke:
        print("SMOKE RUN: reduced rows or resamples. Numbers are not results and no selection file is written.")

    # ------------------------------------------------------------------
    # Rows and roles, identical to the benchmark. The final test cannot load.
    # ------------------------------------------------------------------
    try:
        frame = lb.load_model_frame(root, feature_set)
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
    fit_pos = proto.subsample_rows(fit_pos, args.max_fit_rows, seed)
    y_fit, y_stop, y_val = frame.y[fit_pos], frame.y[stop_pos], frame.y[val_pos]
    sec_stop, sec_val = seconds[stop_pos], seconds[val_pos]
    ids_stop, ids_val = frame.meta[ID_COLUMN].to_numpy()[stop_pos], frame.meta[ID_COLUMN].to_numpy()[val_pos]
    meta_val = frame.meta.iloc[val_pos].reset_index(drop=True)

    try:
        guards.reject_final_test_rows(seconds[np.concatenate([fit_pos, stop_pos, val_pos])], "loaded")
        guards.require_calibration_before_evaluation(seconds[fit_pos], sec_stop, "fit rows before calibration rows")
        selection, confirmation = tune.validation_halves(meta_val["prediction_timestamp_utc"])
    except ValueError as problem:
        fail(str(problem))
    days = meta_val["day"].to_numpy()
    quarter = meta_val["prediction_timestamp_utc"].dt.quarter.to_numpy()
    everything = np.ones(len(y_val), dtype=bool)
    print(f"  fit {len(fit_pos):,} rows | calibration (early-stopping) {len(stop_pos):,} rows, {int(y_stop.sum()):,} events "
          f"({y_stop.mean():.4%}) | validation {len(val_pos):,} rows, {int(y_val.sum()):,} events ({y_val.mean():.4%})")
    print(f"  2022 first half {int(selection.sum()):,} rows ({y_val[selection].mean():.4%})   second half {int(confirmation.sum()):,} rows "
          f"({y_val[confirmation].mean():.4%})")

    # ------------------------------------------------------------------
    # Refit the selected model (fit rows only; the early-stopping rows choose the number of trees)
    # ------------------------------------------------------------------
    order = np.concatenate([fit_pos, stop_pos, val_pos])
    X = frame.X.iloc[order].reset_index(drop=True)
    fit_mask = np.zeros(len(X), dtype=bool)
    fit_mask[: len(fit_pos)] = True
    lb.apply_fit_categories(X, fit_mask)  # category levels from the fit rows only
    n_fit, n_stop = len(fit_pos), len(stop_pos)
    try:
        fit = lb.fit_lightgbm(X.iloc[:n_fit], y_fit, X.iloc[n_fit : n_fit + n_stop], y_stop, n_jobs=args.n_jobs, seed=seed, params=params)
    except ValueError as problem:
        fail(str(problem))
    s_stop = lb.predict_scores(fit, X.iloc[n_fit : n_fit + n_stop])
    s_val = lb.predict_scores(fit, X.iloc[n_fit + n_stop :])
    del X
    print(f"  LightGBM refitted: {fit.best_iteration} trees, {fit.seconds:,.0f} s")

    # The calibrated model must be the model already reported.
    reproduction = {"checked": False}
    if args.max_fit_rows is None and (root / reference_path).exists():
        try:
            stored = proto.align_scores(ids_val, y_val, pd.read_parquet(root / reference_path), reference_column)
        except ValueError as problem:
            fail(str(problem))
        difference = abs(binary_metrics(y_val, s_val)["pr_auc"] - binary_metrics(y_val, stored)["pr_auc"])
        largest = float(np.max(np.abs(s_val - stored)))
        reproduction = {"checked": True, "file": reference_path, "column": reference_column,
                        "pr_auc_difference": float(difference), "largest_score_difference": largest}
        print(f"  check against the stored scores ({reference_column}): PR-AUC difference {difference:.6f}, largest score difference {largest:.2e}")
        if difference > float(model_cfg["reproduction_pr_auc_tolerance"]):
            fail("the refitted model does not reproduce the stored validation scores, so a different model would be calibrated. "
                 "Check that the same feature set, settings file and seed were used.")
    elif args.max_fit_rows is None:
        print(f"  (no stored scores at {reference_path}: reproduction check skipped)")
    else:
        print("  (smoke run: reproduction check skipped)")

    periods = {SELECTION: selection, CONFIRMATION: confirmation, FULL_YEAR: everything}
    for number in (1, 2, 3, 4):
        periods[f"2022 Q{number}"] = quarter == number

    # ------------------------------------------------------------------
    # Experiment A: calibrators fitted on the early-stopping rows
    # ------------------------------------------------------------------
    try:
        calibrators_a, probs_a = fit_and_apply(methods, s_stop, y_stop, sec_stop, s_val, sec_val, "Experiment A",
                                               ids_fit=ids_stop, ids_apply=ids_val)
    except ValueError as problem:
        fail(str(problem))
    rate_a = float(y_stop.mean())
    table_a = metrics_table("A", "early-stopping rows (2021-07 to 2021-12)", y_val, probs_a, periods, rate_a, n_bins)
    in_sample = metrics_table("A", "early-stopping rows (2021-07 to 2021-12)", y_stop,
                              {m: c.transform(s_stop) for m, c in calibrators_a.items()},
                              {"calibration rows (in-sample, not evidence)": np.ones(len(y_stop), dtype=bool)}, rate_a, n_bins)
    print("\nCalibrators fitted on the early-stopping rows:")
    for method, calibrator in calibrators_a.items():
        print(f"  {calibrator.describe()}")
    show_metrics(table_a[table_a["period"].isin([SELECTION, CONFIRMATION, FULL_YEAR])], "Experiment A on 2022 (calibrators fitted on 2021 H2 rows):")

    # ------------------------------------------------------------------
    # Rule 1: choose the method
    # ------------------------------------------------------------------
    brier_selection = {m: float(table_a[(table_a["method"] == m) & (table_a["period"] == SELECTION)]["brier"].iloc[0]) for m in methods}
    comparisons: list[dict] = []
    ci_high = {}
    ci_high_iso_vs_platt = float("nan")
    if n_boot > 0:
        def compare(name: str, reference: np.ndarray, challenger: np.ndarray, reference_name: str, challenger_name: str, period: str) -> dict:
            mask = periods[period]
            outcome = paired_day_bootstrap_brier(y_val[mask], reference[mask], challenger[mask], days[mask], n_boot=n_boot,
                                                 seed=boot_seed, confidence=confidence)
            row = {"comparison": name, "challenger": challenger_name, "reference": reference_name, "period": period, **outcome}
            comparisons.append(row)
            return row

        for method in CALIBRATED_METHODS:
            if method not in probs_a:
                continue
            for period in (CONFIRMATION, SELECTION, FULL_YEAR):
                row = compare("A: calibrator vs uncalibrated", probs_a["uncalibrated"], probs_a[method], "uncalibrated", method, period)
                if period == CONFIRMATION:
                    ci_high[method] = row["ci_high"]
        if "platt" in probs_a and "isotonic" in probs_a:
            row = compare("A: isotonic vs Platt", probs_a["platt"], probs_a["isotonic"], "platt", "isotonic", CONFIRMATION)
            ci_high_iso_vs_platt = row["ci_high"]
    chosen, reason = rules.choose_method(brier_selection, ci_high, ci_high_iso_vs_platt) if n_boot > 0 else (
        "uncalibrated", "The bootstrap was skipped, so nothing could be confirmed.")

    print("\nBrier score, paired day-level bootstrap on the second half of 2022 (negative = the calibrator is better):")
    for row in comparisons:
        if row["period"] == CONFIRMATION:
            print(f"  {row['comparison']:<32} {row['challenger']:<9} difference {row['difference']:+.6f}  "
                  f"95% interval [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]")
    print(f"\nRule 1 (DEC-023): {chosen.upper()}. {reason}")

    # ------------------------------------------------------------------
    # Experiment B: calibrators fitted on the first half of 2022 (diagnostic)
    # ------------------------------------------------------------------
    probs_b: dict[str, np.ndarray] = {}
    table_b = pd.DataFrame()
    ci_high_recent = float("nan")
    try:
        _, probs_b_half = fit_and_apply(methods, s_val[selection], y_val[selection], sec_val[selection], s_val[confirmation],
                                        sec_val[confirmation], "Experiment B", ids_fit=ids_val[selection], ids_apply=ids_val[confirmation])
    except ValueError as problem:
        fail(str(problem))
    for method, p in probs_b_half.items():
        full = np.full(len(y_val), np.nan)
        full[confirmation] = p
        probs_b[method] = full
    table_b = metrics_table("B", "2022 first half", y_val[confirmation], probs_b_half, {CONFIRMATION: np.ones(int(confirmation.sum()), dtype=bool)},
                            float(y_val[selection].mean()), n_bins)
    show_metrics(table_b, "Experiment B on the second half of 2022 (calibrators fitted on 2022 H1 rows, diagnostic only):")
    if n_boot > 0:
        for method in CALIBRATED_METHODS:
            if method not in probs_a:
                continue
            outcome = paired_day_bootstrap_brier(y_val[confirmation], probs_a[method][confirmation], probs_b[method][confirmation],
                                                 days[confirmation], n_boot=n_boot, seed=boot_seed, confidence=confidence)
            comparisons.append({"comparison": "B vs A (recent rows vs older rows)", "challenger": f"{method}, fitted on 2022 H1",
                                "reference": f"{method}, fitted on 2021 H2", "period": CONFIRMATION, **outcome})
            print(f"  {method:<9} fitted on 2022 H1 minus fitted on 2021 H2: difference {outcome['difference']:+.6f}  "
                  f"95% interval [{outcome['ci_low']:+.6f}, {outcome['ci_high']:+.6f}]")
            if method == chosen:
                ci_high_recent = outcome["ci_high"]
    fit_source, fit_source_reason = rules.choose_final_fit_source(chosen, ci_high_recent)
    print(f"\nRule 2 (DEC-023): final-test calibrator fit source = {fit_source.upper()}. {fit_source_reason}")

    # ------------------------------------------------------------------
    # Reliability tables
    # ------------------------------------------------------------------
    reliability_rows = []
    for method, p in probs_a.items():
        for period in (CONFIRMATION, FULL_YEAR):
            table = reliability_table(y_val[periods[period]], p[periods[period]], n_bins)
            reliability_rows.append(table.assign(experiment="A", method=method, period=period))
    for number in (1, 2, 3, 4):
        table = reliability_table(y_val[periods[f"2022 Q{number}"]], probs_a["uncalibrated"][periods[f"2022 Q{number}"]], n_bins)
        reliability_rows.append(table.assign(experiment="A", method="uncalibrated", period=f"2022 Q{number}"))
    for method, p in probs_b_half.items():
        reliability_rows.append(reliability_table(y_val[confirmation], p, n_bins).assign(experiment="B", method=method, period=CONFIRMATION))
    reliability = pd.concat(reliability_rows, ignore_index=True)
    reliability = reliability[["experiment", "method", "period", "bin", "rows", "events", "mean_predicted", "observed_rate"]]

    summary = pd.concat([table_a, in_sample, table_b], ignore_index=True)
    summary_name = "calibration_summary.csv" if official else f"phase6_calibration{infix}_summary.csv"
    summary.to_csv(tables / summary_name, index=False)
    reliability.to_csv(tables / f"phase6_calibration{infix}_reliability.csv", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(tables / f"phase6_calibration{infix}_comparisons.csv", index=False)

    # ------------------------------------------------------------------
    # Figure and predictions
    # ------------------------------------------------------------------
    brier_h2 = {m: float(table_a[(table_a["method"] == m) & (table_a["period"] == CONFIRMATION)]["brier"].iloc[0]) for m in probs_a}
    quarter_tables = {f"2022 Q{n}": reliability_table(y_val[periods[f"2022 Q{n}"]], probs_a["uncalibrated"][periods[f"2022 Q{n}"]], n_bins)
                      for n in (1, 2, 3, 4)}
    figure_path = figures / f"calibration_curve{infix}.png"
    drew = draw_figure(figure_path, y_val[confirmation], {m: p[confirmation] for m, p in probs_a.items()}, brier_h2, quarter_tables)
    if drew:
        print(f"\nWrote {figure_path}")

    predictions = pd.DataFrame({
        ID_COLUMN: np.concatenate([ids_stop, ids_val]),
        "role": ["early_stop"] * len(ids_stop) + ["validation"] * len(ids_val),
        "prediction_time_utc": pd.to_datetime(np.concatenate([sec_stop, sec_val]), unit="s", utc=True),
        "label": np.concatenate([y_stop, y_val]),
        "uncalibrated": np.concatenate([s_stop, s_val]),
    })
    nothing_stop = np.full(len(ids_stop), np.nan)
    for method in CALIBRATED_METHODS:
        if method in probs_a:
            predictions[f"{method}_fitted_2021h2"] = np.concatenate([nothing_stop, probs_a[method]])
        if method in probs_b:
            predictions[f"{method}_fitted_2022h1"] = np.concatenate([nothing_stop, probs_b[method]])
    predictions.to_parquet(predictions_dir / f"lightgbm_calibrated_validation{infix}.parquet", index=False)

    # ------------------------------------------------------------------
    # Decision record
    # ------------------------------------------------------------------
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "smoke_run": smoke,
        "target": "severe_delay_120",
        "prediction_time": "scheduled departure UTC minus 2 hours",
        "feature_version": FEATURE_VERSION,
        "feature_set": feature_set,
        "model": "LightGBM, tuned settings from " + model_cfg["params_file"],
        "seed": seed,
        "best_iteration": int(fit.best_iteration),
        "reproduction_check": reproduction,
        "rows": {"fit": int(len(fit_pos)), "calibration_early_stop": int(len(stop_pos)), "validation": int(len(val_pos)),
                 "selection_half": int(selection.sum()), "confirmation_half": int(confirmation.sum())},
        "events": {"calibration_early_stop": int(y_stop.sum()), "validation": int(y_val.sum())},
        "prevalence": {"fit": float(y_fit.mean()), "calibration_early_stop": float(y_stop.mean()), "selection_half": float(y_val[selection].mean()),
                       "confirmation_half": float(y_val[confirmation].mean())},
        "calibrators_fitted_on_early_stop_rows": {m: c.describe() for m, c in calibrators_a.items()},
        "brier_selection_half": brier_selection,
        "rule_1": {"chosen_method": chosen, "reason": reason, "confirmation_ci_high_vs_uncalibrated": ci_high,
                   "confirmation_ci_high_isotonic_vs_platt": None if not np.isfinite(ci_high_iso_vs_platt) else ci_high_iso_vs_platt},
        "rule_2": {"final_test_fit_source": fit_source, "reason": fit_source_reason,
                   "ci_high_recent_vs_old": None if not np.isfinite(ci_high_recent) else ci_high_recent},
        "ranking_policy": {"score_used_for_ranking": "uncalibrated LightGBM score",
                           "note": "Platt is strictly increasing, so it gives the same ranking. Isotonic creates ties. "
                                   "Later ranking tables state which scores they use."},
        "bootstrap": {"resamples": n_boot, "seed": boot_seed, "confidence": confidence, "unit": "day"},
        "limits": [
            "2019 and 2020 are training years for this model, so regime evidence is 2022 by quarter only.",
            "The early-stopping rows that calibrate also chose the number of trees, so their scores are slightly optimistic.",
            "Calibration measured on 2022 does not guarantee calibration in 2023; the final test reports it once, with the locked method.",
        ],
        "final_test_used": False,
        "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / f"phase6_calibration{infix}_decision.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    if official:
        (root / "configs").mkdir(exist_ok=True)
        selected = {k: record[k] for k in ("created_utc", "target", "prediction_time", "feature_version", "feature_set", "model", "seed",
                                           "rule_1", "rule_2", "ranking_policy", "bootstrap", "final_test_used")}
        selected["methods_compared"] = methods
        (root / "configs/calibration_selected.json").write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")
        print("Wrote configs/calibration_selected.json")
    print(f"\nWrote tables in {tables} (summary: {summary_name}), predictions in {predictions_dir}.")
    print(f"Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

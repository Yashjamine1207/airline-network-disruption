#!/usr/bin/env python
"""Phase 7A, Step 10a: SHAP explanations, case studies and error analysis for the tuned LightGBM (validation only).

Save as:   scripts/run_phase7a_explain_and_errors.py
Run from the project root, after the calibration run (it reads configs/calibration_selected.json):

    # 1. Smoke run (a few minutes). Tagged "smoke"; numbers are not results.
    python scripts/run_phase7a_explain_and_errors.py --max-fit-rows 300000

    # 2. Real run (a few minutes).
    python scripts/run_phase7a_explain_and_errors.py

What it does
------------
* Refits the selected model exactly as the benchmark did and stops if it does not reproduce the stored scores.
* SHAP (LightGBM's TreeSHAP, in log-odds) on the SECOND half of 2022, where nothing was tuned. The values are
  checked to add up to the model's own output. Global importance by feature and family, direction tables for
  numeric features and levels, and a beeswarm figure.
* Case studies chosen by a fixed rule from score and label only: true positives, false positives, false
  negatives, low-risk correct cases and the most confident mistakes. The typical case of each category is
  drawn as a local explanation.
* Error breakdown on all of 2022 by quarter, carrier, origin airport, route volume, departure time, weekday,
  missing or unseen values, score decile, and (events only) the size of the delay. Alerts use ONE global
  cutoff for the top 1% and 10%, never a cutoff per group.
* Also draws pr_curve.png (all models on the second half of 2022) and training_curves.png from files that
  earlier steps wrote. A missing file is skipped.

What SHAP does and does not say
-------------------------------
SHAP values describe how the model reaches its score. A large value for an airport means the model has
learned that airport goes with higher risk in the 2019-2021 data. It is an association inside the model. It
is not a cause of delay, and it says nothing about weather, aircraft or crews, which this data does not hold.

2023 is never loaded. The arrival-delay minutes used in the severity table are read from the cohort file for
2022 rows only and are used for a retrospective description, never as an input to any model.

Writes (tables are small and safe to commit; no row-level file is written except the case table):
    reports/tables/phase7a_shap_{global,family,numeric_effects,categorical_effects}.csv
    reports/tables/phase7a_case_studies.csv
    reports/tables/error_breakdown.csv                       (plain run; a tagged run writes phase7a_<tag>_error_breakdown.csv)
    reports/tables/phase7a_explain_decision.json
    reports/figures/shap_summary.png, local_flight_explanation.png, pr_curve.png, training_curves.png
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
    from airline_disruption.calibration.config import load_calibration_config
    from airline_disruption.calibration.experiment import fit_and_apply
    from airline_disruption.deep import protocol as proto
    from airline_disruption.evaluation.classification import binary_metrics
    from airline_disruption.explain import cases, errors, plots, shap_tools
    from airline_disruption.explain.dimensions import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dimensions
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.models.refit import check_against_stored, refit_selected_lightgbm
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.calibration.config import load_calibration_config  # noqa: E402
    from airline_disruption.calibration.experiment import fit_and_apply  # noqa: E402
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics  # noqa: E402
    from airline_disruption.explain import cases, errors, plots, shap_tools  # noqa: E402
    from airline_disruption.explain.dimensions import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dimensions  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_SETS, FEATURE_VERSION, ID_COLUMN  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.models.refit import check_against_stored, refit_selected_lightgbm  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

CONFIG_PATH = "configs/calibration.yaml"
SELECTED_PATH = "configs/calibration_selected.json"
PREDICTIONS_DIR = "data/processed/phase6/predictions"
MODEL_FILES = {"MLP control": "mlp", "LSTM": "lstm", "GRU": "gru", "Compact Transformer": "transformer"}
CASE_TITLES = {
    "true_positive": "True positive (in the top 1%, was severe)",
    "false_positive": "False positive (in the top 1%, was not severe)",
    "false_negative": "False negative (severe, but outside the top 10%)",
    "low_risk_correct": "Low-risk correct (not severe, lower-half score)",
}


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7A SHAP, case studies and error analysis (validation only)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--n-jobs", type=int, default=8)
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
    if not (root / SELECTED_PATH).exists():
        fail(f"{SELECTED_PATH} not found. Run scripts/run_phase6_calibration.py first.")
    selected = json.loads((root / SELECTED_PATH).read_text(encoding="utf-8"))
    method = selected["rule_1"]["chosen_method"]
    model_cfg = config["model"]
    feature_set = model_cfg["feature_set"]
    if feature_set not in FEATURE_SETS:
        fail(f"unknown feature set {feature_set!r}")
    seed = int(model_cfg["seed"])
    reference_path = args.reference_predictions or model_cfg["reference_predictions"]
    reference_column = args.reference_column or model_cfg["reference_column"]

    smoke = args.max_fit_rows is not None
    tag = args.tag or ("smoke" if smoke else "")
    official = not smoke and not args.tag
    infix = f"_{tag}" if tag else ""
    tables, figures = root / "reports/tables", root / "reports/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    print(f"Model: tuned LightGBM, feature set {feature_set}, seed {seed}. Calibrated probabilities use: {method}. tag: {tag or '-'}")
    if smoke:
        print("SMOKE RUN: reduced fit rows. Numbers are not results.")

    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH, model_cfg["params_file"]):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}")
    try:
        params = tune.load_params_file(root / model_cfg["params_file"])
        refit = refit_selected_lightgbm(root, feature_set, params, seed, args.n_jobs, args.max_fit_rows)
    except (OSError, ValueError) as problem:
        fail(str(problem))
    print(f"  fit {refit.n_fit:,} | calibration rows {refit.n_stop:,} | validation {len(refit.y_val):,} rows, {int(refit.y_val.sum()):,} events; "
          f"{refit.fit.best_iteration} trees ({refit.fit.seconds:,.0f} s)")

    reproduction = {"checked": False}
    if args.max_fit_rows is None and (root / reference_path).exists():
        try:
            reproduction = check_against_stored(refit, root / reference_path, reference_column, float(model_cfg["reproduction_pr_auc_tolerance"]))
        except ValueError as problem:
            fail(str(problem))
        print(f"  check against stored scores: PR-AUC difference {reproduction['pr_auc_difference']:.6f}")

    y, score = refit.y_val, refit.s_val
    stamps = refit.meta_val["prediction_timestamp_utc"]
    try:
        selection, confirmation = tune.validation_halves(stamps)
        _, calibrated = fit_and_apply([method], refit.s_stop, refit.y_stop, refit.seconds_stop, score, refit.seconds_val, "explain",
                                      ids_fit=refit.ids_stop, ids_apply=refit.ids_val)
    except ValueError as problem:
        fail(str(problem))
    probability = calibrated[method]
    quarter = stamps.dt.quarter.to_numpy()

    # ------------------------------------------------------------------
    # SHAP on the second half of 2022
    # ------------------------------------------------------------------
    h2 = np.flatnonzero(confirmation)
    X_h2 = refit.X_val.iloc[h2]
    contrib, expected = shap_tools.contributions(refit.fit.model, X_h2, refit.fit.best_iteration)
    gap = shap_tools.check_additivity(refit.fit.model, X_h2.iloc[:20_000], contrib.iloc[:20_000], expected, refit.fit.best_iteration)
    print(f"\nSHAP on {len(h2):,} rows of the second half of 2022. Additivity check: largest gap {gap:.2e} (log-odds).")
    importance = shap_tools.global_importance(contrib)
    families = shap_tools.family_importance(importance)
    print("\nMean absolute SHAP value (log-odds), largest first:")
    print(importance.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBy feature family:")
    print(families.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    importance.to_csv(tables / f"phase7a{infix}_shap_global.csv", index=False)
    families.to_csv(tables / f"phase7a{infix}_shap_family.csv", index=False)
    numeric_tables = [shap_tools.numeric_effect(X_h2[c], contrib[c]) for c in X_h2.columns if c in NUMERIC_FEATURES]
    pd.concat(numeric_tables, ignore_index=True).to_csv(tables / f"phase7a{infix}_shap_numeric_effects.csv", index=False)
    cat_tables = [shap_tools.categorical_effect(X_h2[c], contrib[c]) for c in CATEGORICAL_FEATURES if c in X_h2.columns]
    pd.concat(cat_tables, ignore_index=True).to_csv(tables / f"phase7a{infix}_shap_categorical_effects.csv", index=False)
    subtitle = f"Second half of 2022 ({len(h2):,} flights, {min(4000, len(h2)):,} shown). Validation data only."
    plots.shap_beeswarm(figures / f"shap_summary{infix}.png", contrib, X_h2, NUMERIC_FEATURES, importance, subtitle=subtitle)

    # ------------------------------------------------------------------
    # Case studies (chosen by rule from score and label only)
    # ------------------------------------------------------------------
    picks = cases.select_cases(y[h2], score[h2])
    print("\nCase studies chosen by rule on the second half of 2022:")
    print(picks.groupby("category").size().to_string())
    cohort = pd.read_parquet(root / lb.COHORT_PATH, columns=[ID_COLUMN, "arrival_delay_minutes"])
    delay = cohort.set_index(ID_COLUMN)["arrival_delay_minutes"].reindex(refit.ids_val).to_numpy()
    del cohort
    # The realised delay must agree with the stored label. If it does not, the severity table would be wrong.
    threshold = float(model_cfg.get("severe_delay_minutes", 120))
    disagree = int((((delay >= threshold) & np.isfinite(delay)) != (y == 1)).sum())
    if disagree:
        message = f"{disagree:,} validation rows have a label that disagrees with arrival_delay_minutes >= {threshold:.0f}"
        if not smoke:
            fail(message)
        print(f"  WARNING (smoke run only): {message}")
    case_rows, panels = [], []
    for pick in picks.itertuples():
        pos = h2[pick.position]
        values = X_h2.iloc[pick.position]
        top = shap_tools.top_contributions(contrib.iloc[pick.position], values, k=8)
        case_rows.append({
            "category": pick.category, "pick": pick.pick, "source_row_number": int(refit.ids_val[pos]),
            "prediction_time_utc": stamps.iloc[pos], "origin": values["origin_airport"], "destination": values["destination_airport"],
            "carrier": values["carrier_identifier"], "local_departure_hour": values["scheduled_departure_hour_local"],
            "score": pick.score, "score_rank_in_h2": pick.rank, "calibrated_probability": float(probability[pos]),
            "event": pick.label, "arrival_delay_minutes": delay[pos],
            "top_features": "; ".join(f"{f}={v} ({s:+.3f})" for f, v, s in top[:6]),
        })
        if pick.pick == "typical_q50" and pick.category in CASE_TITLES:
            panels.append({"category": pick.category, "contributions": top, "title": (
                f"{CASE_TITLES[pick.category]}\n{values['origin_airport']} to {values['destination_airport']}, carrier {values['carrier_identifier']}; "
                f"score {pick.score:.3f} (rank {pick.rank:,}), calibrated {probability[pos]:.3f}, arrival delay {delay[pos]:.0f} min")})
    pd.DataFrame(case_rows).to_csv(tables / f"phase7a{infix}_case_studies.csv", index=False)
    panels.sort(key=lambda p: list(CASE_TITLES).index(p["category"]))
    plots.local_explanations(figures / f"local_flight_explanation{infix}.png", panels)

    # ------------------------------------------------------------------
    # Error breakdown on all of 2022 (one global cutoff for each review capacity)
    # ------------------------------------------------------------------
    frame = pd.DataFrame({"label": y, "score": score, "probability": probability, "arrival_delay_minutes": delay})
    cutoffs = errors.top_fraction_cutoffs(score, (0.01, 0.10))
    print(f"\nGlobal cutoffs on 2022: top 1% = score {cutoffs[0.01]:.4f}, top 10% = score {cutoffs[0.10]:.4f}")
    breakdown = [errors.group_metrics(frame.assign(group=pd.Series(labels).to_numpy()), name, cutoffs, "group")
                 for name, labels in build_dimensions(refit.X_val, refit.X_fit, score, [f"2022 Q{q}" for q in quarter]).items()]
    overall = errors.group_metrics(frame.assign(group="all 2022 flights"), "overall", cutoffs, "group")
    severity = errors.severity_table(frame, cutoffs)
    table = pd.concat([overall, *breakdown, severity], ignore_index=True)
    table["small_sample"] = table["small_sample"].fillna(False).astype(bool)  # one clean boolean column in the CSV
    for count in ("rows", "events"):
        table[count] = table[count].astype("Int64")  # whole numbers, not 156223.0
    name = "error_breakdown.csv" if official else f"phase7a{infix}_error_breakdown.csv"
    table.to_csv(tables / name, index=False)

    show = ["group", "rows", "events", "prevalence", "probability_to_observed", "recall_top1pct", "recall_top10pct", "pr_auc"]
    for dimension in ("quarter_2022", "score_decile", "route_volume_in_fit_period"):
        print(f"\n{dimension}:")
        print(table[table["dimension"] == dimension][show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nEvents by size of delay (share caught by a global top-K% review):")
    print(severity.to_string(index=False, float_format=lambda v: f"{v:.4f}") if len(severity) else "  (no events with a delay of 120 minutes or more)")
    wide = ["group", "rows", "events", "prevalence", "probability_to_observed", "precision_top1pct", "recall_top1pct", "recall_top10pct", "pr_auc", "lift"]
    for dimension, title in (("carrier", "Every carrier"), ("origin_airport_top25", "Origin airports (25 largest by 2022 volume, then the rest)"),
                             ("scheduled_departure_local_hours", "Scheduled local departure hour"), ("day_of_week", "Day of week (0 = Monday)"),
                             ("missing_or_unseen_values", "Missing or unseen values")):
        print(f"\n{title}:")
        print(table[table["dimension"] == dimension][wide].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPR-AUC is blank where a group has fewer than 30 events or non-events.")

    print("\nSHAP direction: mean SHAP value (log-odds) by fifth of the feature, for the calendar features:")
    for feature in ("scheduled_departure_month", "scheduled_departure_hour_local", "scheduled_arrival_hour_local"):
        effect = pd.concat(numeric_tables, ignore_index=True)
        effect = effect[effect["feature"] == feature]
        if len(effect):
            print(f"  {feature}: " + "  ".join(f"[{r.value_low:g} to {r.value_high:g}] {r.mean_shap:+.3f}" for r in effect.itertuples()))
    for feature in CATEGORICAL_FEATURES:
        effect = pd.concat(cat_tables, ignore_index=True)
        effect = effect[effect["feature"] == feature]
        for direction, words in (("highest", "raise the score most"), ("lowest", "lower the score most")):
            part = effect[effect["direction"] == direction].head(5)
            if len(part):
                print(f"  {feature}, levels that {words}: " + ", ".join(f"{r.level} {r.mean_shap:+.3f}" for r in part.itertuples()))
    cases_table = pd.read_csv(tables / f"phase7a{infix}_case_studies.csv")
    print("\nThe median case of each category (full table in the case-studies file):")
    typical = cases_table[cases_table["pick"] == "typical_q50"]
    print(typical[["category", "origin", "destination", "carrier", "local_departure_hour", "score_rank_in_h2", "calibrated_probability", "event", "arrival_delay_minutes"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------
    # Comparison figures from earlier steps' files
    # ------------------------------------------------------------------
    scores_h2 = {"LightGBM (tuned)": score[confirmation]}
    histories = {}
    for label, arch in MODEL_FILES.items():
        path = root / PREDICTIONS_DIR / f"{arch}_validation.parquet"
        if path.exists():
            try:
                scores_h2[label] = proto.align_scores(refit.ids_val, y, pd.read_parquet(path), arch)[confirmation]
            except ValueError as problem:
                print(f"  skipped {label} in the PR figure: {problem}")
        history_path = tables / f"phase6_deep_{arch}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            history = history[(history["seed"] == history["seed"].min()) & (history["config"] == history["config"].min())].reset_index(drop=True)
            histories[label] = history
    plots.pr_curves(figures / f"pr_curve{infix}.png", y[confirmation], scores_h2, "Second half of 2022 (validation data only)")
    plots.training_curves(figures / f"training_curves{infix}.png", histories)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "smoke_run": smoke,
        "feature_version": FEATURE_VERSION, "feature_set": feature_set, "seed": seed, "best_iteration": int(refit.fit.best_iteration),
        "calibration_method_used_for_probabilities": method, "reproduction_check": reproduction,
        "shap": {"rows": int(len(h2)), "period": "2022 second half", "unit": "log-odds", "additivity_gap": gap,
                 "expected_value_log_odds": expected, "top_features": importance.head(5)["feature"].tolist()},
        "cases": {"rule": "cases.select_cases: alert = top 1%, missed = outside top 10%, quantiles 0.25/0.5/0.75, 3 extremes",
                  "counts": picks.groupby("category").size().to_dict()},
        "error_breakdown": {"period": "2022 full year", "cutoffs": {str(k): v for k, v in cutoffs.items()},
                            "note": "the first half of 2022 chose LightGBM settings, so the full year favours the model slightly"},
        "not_evaluable": ["2019 and 2020 regimes (training years for this model)",
                          "sequence availability and length (the tabular model has no sequence input)",
                          "large regression errors and airport recovery episodes (no regression model or episode table is loaded here)"],
        "final_test_used": False, "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(), "packages": package_versions(),
    }
    (tables / f"phase7a{infix}_explain_decision.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote tables in {tables} and figures in {figures}. Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

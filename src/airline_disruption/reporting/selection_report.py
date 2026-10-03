"""reports/evaluation/final_model_selection_report.md (Phase 7B, Step 14).

The record of what was selected, on what evidence, and what was rejected. Every number is read from a table or from the
lock. The sentences that judge a result (supported or not, drift or no drift) are chosen by the numbers, never fixed.
"""

from __future__ import annotations

import json

import pandas as pd

from airline_disruption.final import protocol as proto
from airline_disruption.reporting import facts
from airline_disruption.reporting.calibration_report import CAPACITY_ORDER, _capacity_table, _season_text, _with_interval_text
from airline_disruption.reporting.common import Evidence, header, inputs_section, not_supported_section
from airline_disruption.reporting.error_report import _regime_table
from airline_disruption.reporting.inputs import ARCHITECTURES, Inputs
from airline_disruption.reporting.text import bullets, interval, join_sections, md_table, num, pct, signed, whole

SEVERE, CANCEL = "severe_delay_120", "cancelled"
ABLATION_FILE = "reports/tables/phase6_ablation2020_lightgbm_comparisons.csv"


# ---------------------------------------------------------------------------------------------- shared helpers
def claim_outcomes(claims: dict) -> dict[str, bool]:
    return {name: bool(claim["supported"]) for name, claim in claims.items()}


def failed_claims(claims: dict) -> list[str]:
    return [name for name, ok in claim_outcomes(claims).items() if not ok]


def locked_scorer_row(summary: pd.DataFrame, period: str = "final_2023") -> pd.Series:
    return facts.one(summary, period=period, scorer="lightgbm_locked")


def deep_results(inp: Inputs) -> dict[str, dict]:
    """The latest non-smoke run of each sequence architecture, found by its decision file.

    A run counts only if its decision file names the architecture, says it was not a smoke run, and has a validation
    table next to it. Anything else is ignored, so a test build can never be reported as a result.
    """
    found: dict[str, dict] = {}
    folder = inp.root / "reports/tables"
    for arch in ARCHITECTURES:
        best = None
        for path in sorted(folder.glob(f"phase6_deep_{arch}_*_decision.json")):
            relative = path.relative_to(inp.root).as_posix()
            validation_name = relative.replace("_decision.json", "_validation.csv")
            if not (inp.root / validation_name).exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("architecture") != arch or data.get("smoke_run") is not False:
                continue
            if best is None or data.get("created_utc", "") > best[0]:
                best = (data.get("created_utc", ""), relative, validation_name, data)
        if best is not None:
            _, relative, validation_name, data = best
            inp.json(relative)
            found[arch] = {"decision": data, "validation": inp.csv(validation_name),
                           "comparisons": inp.csv(validation_name.replace("_validation.csv", "_comparisons.csv"), quiet=True)}
    return found


def sequence_text(deep: dict[str, dict]) -> str:
    for arch in ARCHITECTURES:
        sequence = deep.get(arch, {}).get("decision", {}).get("sequence")
        try:
            if sequence:
                return (f"{sequence['entity']}: {sequence['lookback_bins']} bins of {sequence['bin_hours']} hours ({sequence['lookback_hours']} hours of lookback), "
                        f"channels {', '.join(sequence['channels'])}; padding is masked and no outcome column is read")
        except KeyError:
            continue
    return "an origin-airport schedule sequence (DEC-017); the run files that define it were not found by this builder"


# ---------------------------------------------------------------------------------------------- sections
def _decision(inp: Inputs, ev: Evidence) -> str:
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    capacity = inp.csv(inp.final_name(SEVERE, "capacity"), required=True)
    d = ev.target(SEVERE)
    claims = d["claims"]
    failed = failed_claims(claims)
    fin = locked_scorer_row(summary)
    c2 = claims["C2_beats_history"]
    day10 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="10%")
    day1 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="1%")
    lines = [
        "## Decision", "",
        "**Selected for severe arrival delay (120 minutes or more, a project definition):** the tuned LightGBM classifier on schedule, carrier, origin, destination "
        "and route features, locked before the 2023 rows were read. Its settings, its tree count and its calibrator were all chosen on 2019 to 2022 data.",
    ]
    if not failed:
        verdict = (f"All three pre-registered claims were supported on the 2023 final test. PR-AUC was {num(fin['pr_auc'])} against an event rate of {pct(fin['prevalence'], 2)} "
                   f"(lift {num(fin['lift'], 2)}, lower end of its interval {num(fin['lift_low'], 2)}), and it beat the best history baseline ({c2['baseline']}) by "
                   f"{signed(c2['difference'])} {interval(c2['difference_low'], c2['difference_high'])}.")
    else:
        verdict = (f"{3 - len(failed)} of 3 pre-registered claims were supported on the 2023 final test. Not supported: {', '.join(failed)}. PR-AUC was {num(fin['pr_auc'])} against an "
                   f"event rate of {pct(fin['prevalence'], 2)} (lift {num(fin['lift'], 2)}). Against the best history baseline ({c2['baseline']}) the difference was "
                   f"{signed(c2['difference'])} {interval(c2['difference_low'], c2['difference_high'])}.")
    size = (f"ROC-AUC was {num(fin['roc_auc'], 3)}. Reviewing the top 1% of flights inside each day caught {pct(day1['recall'])} of the severe delays "
            f"at {pct(day1['precision'])} precision, and the top 10% caught {pct(day10['recall'])} at {pct(day10['precision'])}.")
    if day10["recall"] < 0.5:
        size += " Most severe delays are not in a top-10% list."
    lines += ["", verdict, size]
    cancel = ev.target(CANCEL)["claims"]
    cancel_failed = failed_claims(cancel)
    if not cancel_outcome_positive(cancel):
        lines += ["", "**Cancellation: no model is selected.** The untuned baseline did not show skill over chance and did not beat the best history baseline on 2023. "
                      "It is kept in the repository as a labelled negative result with ranking only, no calibration and no thresholds."]
    else:
        lines += ["", f"**Cancellation:** the untuned baseline passed {3 - len(cancel_failed)} of 3 claims (not supported: {', '.join(cancel_failed) or 'none'}). It stays a baseline with ranking only; "
                      "no calibration or threshold was built for it."]
    rejected = ev.lock.get("rejected_challengers", {})
    lines += ["", f"**No sequence or deep model is retained.** Rejected on 2022 validation evidence before the final test, and not scored on 2023: {', '.join(rejected)}."]
    return "\n".join(lines)


def cancel_outcome_positive(claims: dict) -> bool:
    """False when cancellation shows no skill over chance (C1 fails) and does not beat history (C2 fails): a negative result."""
    outcomes = claim_outcomes(claims)
    return outcomes["C1_skill_over_chance"] or outcomes["C2_beats_history"]


def _record(inp: Inputs, ev: Evidence, deep: dict[str, dict]) -> str:
    lock, spec, d = ev.lock, ev.locked(SEVERE), ev.target(SEVERE)
    cohort_file = next((f for f in list(lock.get("files", [])) if "cohort" in f), "the Phase 6 cohort table")
    params = lock["params"]
    rows = [
        ("Target", f"{spec['label']}. The 120-minute threshold is a project definition, not an official classification."),
        ("Eligible cohort", f"rows with `{proto.TARGETS[SEVERE]['eligibility']}` set in `{cohort_file}`. The flag is defined in the Phase 6 cohort build."),
        ("Prediction timestamp", lock["prediction_time"]),
        ("Feature version", f"{lock['feature_version']}, set `{lock['feature_set']}`, {spec['n_features']} features: schedule, carrier, origin, destination and route. "
                            "No history, network, sequence, weather or outcome features."),
        ("Sequence definition", f"none in the selected model. The rejected sequence challengers used {sequence_text(deep)}."),
        ("Training period", f"{lock['periods']['fit']} (UTC prediction time), {whole(spec['fit_rows'])} rows"),
        ("Early-stopping rows", f"{lock['periods']['early_stop']}, {whole(spec['early_stop_rows'])} rows. Never 2023."),
        ("Calibration period", f"{spec['calibrator_primary']['fit_period']}, {whole(spec['calibrator_primary']['fit_rows'])} rows, method {spec['calibrator_primary']['method']}"),
        ("Validation period", f"{lock['periods']['validation']}, {whole(spec['validation_rows'])} rows, {whole(spec['validation_events'])} events"),
        ("Final-test period", f"{lock['periods']['final_test']}, {whole(d['rows'])} rows, {whole(d['events'])} events, scored once"),
        ("Selected model", f"LightGBM, {whole(d['trees'])} trees (early stopping on the early-stopping rows), num_leaves {params['num_leaves']}, learning rate {params['learning_rate']}, "
                           f"min_child_samples {params['min_child_samples']}, reg_lambda {params['reg_lambda']}, seed {lock['seed']}"),
        ("Refit before the final test", f"no. {lock['model_policy']['reason']}"),
        ("Rejected challengers", ", ".join(lock.get("rejected_challengers", {}))),
        ("Ranking score", "the uncalibrated model score. Platt calibration is strictly increasing, so it gives the same ranking."),
        ("Lock", f"`{ev.lock_sha[:16]}`, first run {ev.marker.get('first_run_utc', 'unknown')}"),
    ]
    table = "| Field | Value |\n| --- | --- |\n" + "\n".join(f"| {k} | {v.replace('|', '/')} |" for k, v in rows)
    return "\n".join(["## Record", "", table])


def _evidence(inp: Inputs, ev: Evidence) -> str:
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    parts = ["## Evidence: severe delay", "", "PR-AUC is the primary metric. Lift is PR-AUC divided by the period's event rate, so years with different event rates can be compared."]
    for period, label in (("validation_2022", "2022 validation (chose the settings and the calibrator)"), ("final_2023", "2023 final test (scored once)")):
        part = summary[summary["period"] == period].copy()
        part["interval_text"] = [interval(l, h) for l, h in zip(part["pr_auc_low"], part["pr_auc_high"])]
        parts += ["", f"**{label}**", "", md_table(part, [("Scorer", "scorer", None), ("PR-AUC", "pr_auc", num), ("95% interval", "interval_text", None),
                                                          ("Lift", "lift", lambda v: num(v, 2)), ("ROC-AUC", "roc_auc", lambda v: num(v, 3))])]
    parts += ["", "Pre-registered claims (fixed in the lock before 2023 was read; the only pass or fail statements):", "", bullets(facts.claim_lines(ev.target(SEVERE)["claims"]))]
    parts += ["", "MAE: not applicable to this classifier. No regression model was locked or scored on 2023. Validation results for the Phase 4 linear, ridge and elastic net "
                  "regressions are in `reports/tables/phase4_*_regression_validation.csv` and are not part of this decision."]
    return "\n".join(parts)


def _calibration(inp: Inputs, ev: Evidence) -> str:
    calibration = inp.csv(inp.final_name(SEVERE, "calibration"), required=True)
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    locked = calibration[calibration["variant"].str.contains("locked primary")]
    r22, r23 = facts.one(locked, period="validation_2022"), facts.one(locked, period="final_2023")
    prev22 = locked_scorer_row(summary, "validation_2022")["prevalence"]
    prev23 = locked_scorer_row(summary, "final_2023")["prevalence"]
    rule1, rule2 = ev.selected["rule_1"], ev.selected["rule_2"]
    table = md_table(locked, [("Period", "period", None), ("Mean predicted", "mean_predicted", lambda v: pct(v, 2)), ("Observed", "prevalence", lambda v: pct(v, 2)),
                              ("Predicted / observed", "predicted_to_observed", lambda v: num(v, 3)), ("Brier", "brier", lambda v: num(v, 5)),
                              ("Slope", "calibration_slope", lambda v: num(v, 3)), ("Intercept", "calibration_in_the_large", lambda v: num(v, 3))])
    fit_period = ev.locked(SEVERE)["calibrator_primary"]["fit_period"]
    lines = [f"Method: {rule1['chosen_method']}. {rule1['reason']} The final calibrator was fitted on {fit_period}. {rule2['reason']}"]
    if r23["predicted_to_observed"] < r22["predicted_to_observed"] and prev23 > prev22:
        lines.append(f"On 2023 the probabilities were too low by more than on 2022 (ratio {num(r23['predicted_to_observed'], 3)} against {num(r22['predicted_to_observed'], 3)}), because the event rate "
                     f"rose from {pct(prev22, 2)} to {pct(prev23, 2)} and the calibrator was fitted on earlier rows. Read the probabilities as too low when the event rate is higher than in the calibration period.")
    else:
        lines.append(f"The predicted-to-observed ratio was {num(r22['predicted_to_observed'], 3)} on 2022 and {num(r23['predicted_to_observed'], 3)} on 2023.")
    return "\n".join(["## Calibration evidence", "", table, "", bullets(lines)])


def _ranking(inp: Inputs) -> str:
    capacity = inp.csv(inp.final_name(SEVERE, "capacity"), required=True)
    by_group = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    shaped = _with_interval_text(capacity)
    parts = ["## Ranking evidence (2023, uncalibrated scores)", "",
             "Inside each day (one fixed share of that day's flights reviewed). This is the figure to quote for ranking skill:", "", _capacity_table(shaped, "final_2023", "by_day"),
             "", "Over the whole period (one ranking over every flight, which also rewards knowing which months are busy):", "", _capacity_table(shaped, "final_2023", "whole_year")]
    season = _season_text(by_group, capacity)
    if season:
        parts += ["", bullets(season)]
    return "\n".join(parts)


def _robustness(inp: Inputs, ev: Evidence) -> str:
    table, lifts = _regime_table(inp, ev, SEVERE)
    parts = ["## Robustness across regimes", "",
             "Each row is a separate model fitted on the years before the year it is scored on, except the last row (the locked model). 2020 is kept as a distribution-shift regime.",
             "", table]
    if lifts:
        parts += ["", lifts]
    ablation = inp.csv(ABLATION_FILE, quiet=True)
    if ablation is not None and len(ablation):
        rows = ablation.assign(interval_text=[interval(l, h, 5) for l, h in zip(ablation["ci_low"], ablation["ci_high"])])
        parts += ["", "Training with the 2020 shock year against training without it, scored on later 2022 rows:", "",
                  md_table(rows, [("Challenger", "challenger", None), ("Reference", "reference", None), ("Period", "period", None), ("Difference", "difference", lambda v: signed(v, 5)),
                                  ("95% interval", "interval_text", None), ("Verdict", "verdict", None)])]
    return "\n".join(parts)


def _challengers(inp: Inputs, ev: Evidence, deep: dict[str, dict]) -> str:
    rejected = ev.lock.get("rejected_challengers", {})
    parts = ["## Rejected challengers", "", "No challenger was scored on 2023. Scoring them there would have turned the final test into a second selection round.", "",
             bullets([f"**{name}**: {why}" for name, why in rejected.items()])]
    rows = []
    for arch, result in deep.items():
        validation = result["validation"]
        period = "2022 full year" if (validation["period"] == "2022 full year").any() else validation["period"].iloc[0]
        for model in (arch, "lightgbm_base_no_year"):
            part = validation[(validation["model"] == model) & (validation["period"] == period)]
            if len(part):
                r = part.iloc[0]
                rows.append({"Challenger": arch, "Model": "LightGBM in the same file" if model != arch else arch, "Period": period,
                             "PR-AUC": num(r["pr_auc"]), "Lift": num(r["pr_auc_lift"], 2), "ROC-AUC": num(r["roc_auc"], 3)})
    if rows:
        parts += ["", "Validation results from the Phase 6 runs that were found (2022 data, never 2023):", "", md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]])]
        verdicts = []
        for arch, result in deep.items():
            comp = result["comparisons"]
            if comp is not None and len(comp):
                verdicts += [f"{arch} against {r.reference}, {r.period}: {r.verdict}" for r in comp.itertuples()]
        if verdicts:
            parts += ["", "Paired day-bootstrap verdicts from those runs:", "", bullets(verdicts)]
    else:
        parts += ["", "The Phase 6 validation tables for the sequence models were not found by this builder, so no PR-AUC is restated here. "
                      "The reasons above come from the lock and cite the decision log."]
    return "\n".join(parts)


def _limits(inp: Inputs, ev: Evidence) -> str:
    by_group = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    missing = by_group[by_group["dimension"] == "missing_or_unseen_values"]
    parts = ["## Limitations that apply to this decision", ""]
    if len(missing):
        parts += ["Flights with an unseen category (a route, airport or carrier the model never saw in training) against complete rows, 2023:", "",
                  md_table(missing, [("Group", "group", None), ("Flights", "rows", whole), ("Events", "events", whole), ("Event rate", "prevalence", lambda v: pct(v, 2)),
                                     ("Top 10% alert rate", "alert_rate_top10pct", pct), ("Recall at top 10%", "recall_top10pct", pct)]), ""]
    limits = list(ev.decision.get("limits", []))
    limits += ["No held-out carrier or held-out airport experiment was run, so generalisation to new entities is not tested. The test is chronological only.",
               "There is no sequence input in the selected model, so missing-sequence subgroups do not apply to it.",
               "All results are associations in retrospective data. Nothing here says why a flight was late, or what an airline could have done about it."]
    parts += [bullets(limits)]
    return "\n".join(parts)


def _complexity(ev: Evidence) -> str:
    c2 = ev.target(SEVERE)["claims"]["C2_beats_history"]
    rejected = ev.lock.get("rejected_challengers", {})
    if c2["supported"]:
        first = (f"The tuned LightGBM was kept because it beat the best history baseline on 2023: difference {signed(c2['difference'])} "
                 f"{interval(c2['difference_low'], c2['difference_high'])}.")
    else:
        first = (f"The tuned LightGBM did not clearly beat the best history baseline on 2023: difference {signed(c2['difference'])} "
                 f"{interval(c2['difference_low'], c2['difference_high'])}. A history rate would be the simpler and equally defensible choice.")
    return "\n".join(["## Why complexity was rejected", "", first, "",
                      f"{len(rejected)} challengers were rejected before the final test. The reasons recorded in the lock:", "",
                      bullets([f"{name}: {why}" for name, why in rejected.items()]),
                      "", "A more complex model has to earn its place on validation evidence. No challenger did."])


def build(inp: Inputs, ev: Evidence) -> str:
    deep = deep_results(inp)
    return join_sections([
        header("Final model-selection report", inp, ev), _decision(inp, ev), _record(inp, ev, deep), _evidence(inp, ev), _calibration(inp, ev), _ranking(inp),
        _robustness(inp, ev), _challengers(inp, ev, deep), _complexity(ev), _limits(inp, ev), not_supported_section(), inputs_section(inp),
    ])

"""reports/evaluation/final_technical_report.md (Phase 7B, Step 14).

One readable account of the whole study. It states the results that the Phase 6 and Phase 7 tables support, says plainly what was
not done, and points to the earlier phase reports for Phase 3 to 5 results instead of restating numbers it does not read.
"""

from __future__ import annotations

import pandas as pd

from airline_disruption.reporting import facts
from airline_disruption.reporting.calibration_report import _season_text, _with_interval_text
from airline_disruption.reporting.common import Evidence, header, inputs_section, missing_case_studies, not_supported_section
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.selection_report import cancel_outcome_positive, claim_outcomes, failed_claims, locked_scorer_row
from airline_disruption.reporting.text import bullets, interval, join_sections, md_table, num, pct, signed, whole

SEVERE, CANCEL = "severe_delay_120", "cancelled"
EVAL = "reports/evaluation"
TABLES = "reports/tables"

# Reports written by the same run. They do not exist yet when this report is built, so they are not checked on disk.
BUILT_HERE = frozenset(f"{EVAL}/{name}.md" for name in ("calibration_and_alert_policy_report", "explainability_report", "error_analysis_report",
                                                       "final_model_selection_report", "final_technical_report"))

QUESTIONS = (
    ("Which flights are likely to see severe arrival delay or cancellation?", "answer_1", (f"{EVAL}/final_model_selection_report.md", f"{EVAL}/calibration_and_alert_policy_report.md")),
    ("Which operational variables are associated with disruption risk?", "answer_2", (f"{EVAL}/explainability_report.md",)),
    ("How does disruption pressure vary across airports and routes over time?", None, (f"{EVAL}/network_resilience_report.md", f"{TABLES}/airport_route_disruption_summary.csv")),
    ("How quickly do airport-level disruption episodes recover?", None, (f"{EVAL}/survival_recovery_report.md", f"{TABLES}/recovery_summary.csv")),
    ("Which airports and routes show more exposure or downstream scheduled pressure?", None, (f"{EVAL}/network_resilience_report.md", f"{EVAL}/propagation_report.md", f"{TABLES}/network_exposure_summary.csv")),
    ("Do schedule, airport, route and network features add predictive value?", None, (f"{TABLES}/classification_model_comparison.csv", f"{EVAL}/final_model_selection_report.md")),
    ("Do LSTM, GRU or a compact Transformer improve on the strong tabular model?", "answer_7", (f"{EVAL}/final_model_selection_report.md",)),
    ("Does sequence information add value beyond the tabular baseline?", "answer_7", (f"{EVAL}/final_model_selection_report.md", f"{EVAL}/explainability_report.md")),
    ("How reliable are the probabilities across operating regimes?", "answer_9", (f"{EVAL}/calibration_and_alert_policy_report.md",)),
    ("Which flights could be prioritised under transparent retrospective review scenarios?", "answer_10", (f"{EVAL}/calibration_and_alert_policy_report.md",)),
)


def _summary(inp: Inputs, ev: Evidence) -> str:
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    d = ev.target(SEVERE)
    failed = failed_claims(d["claims"])
    fin = locked_scorer_row(summary)
    c2 = d["claims"]["C2_beats_history"]
    lines = [
        "## Summary", "",
        f"This study asks which flights will arrive 120 or more minutes late (a project-defined threshold) or be cancelled, using only the schedule and information known two hours before "
        f"scheduled departure. A tuned LightGBM model was locked on 2019 to 2022 data and scored once on 2023-01-01 to 2023-08-31.",
        "",
        (f"Severe delay: all three pre-registered claims were supported. PR-AUC {num(fin['pr_auc'])} against an event rate of {pct(fin['prevalence'], 2)}, lift {num(fin['lift'], 2)}, "
         f"ahead of the best history baseline by {signed(c2['difference'])} {interval(c2['difference_low'], c2['difference_high'])}." if not failed else
         f"Severe delay: {3 - len(failed)} of 3 pre-registered claims were supported (not supported: {', '.join(failed)}). PR-AUC {num(fin['pr_auc'])} against an event rate of "
         f"{pct(fin['prevalence'], 2)}, lift {num(fin['lift'], 2)}."),
        "",
        ("Cancellation: no model was selected. The baseline showed no skill over chance and did not beat the best history baseline on 2023." if not cancel_outcome_positive(ev.target(CANCEL)["claims"])
         else "Cancellation: the untuned baseline passed at least one of the first two claims. It stays a ranking-only baseline."),
        "",
        f"{len(ev.lock.get('rejected_challengers', {}))} sequence and neural challengers were rejected on 2022 evidence and none was scored on 2023. "
        "The simpler tabular model is the result, and that is a valid outcome.",
    ]
    return "\n".join(lines)


def _answers(inp: Inputs, ev: Evidence) -> dict[str, str]:
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    capacity = inp.csv(inp.final_name(SEVERE, "capacity"), required=True)
    calibration = inp.csv(inp.final_name(SEVERE, "calibration"), required=True)
    fin = locked_scorer_row(summary)
    locked = calibration[calibration["variant"].str.contains("locked primary")]
    r22, r23 = facts.one(locked, period="validation_2022"), facts.one(locked, period="final_2023")
    day1 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="1%")
    day10 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="10%")
    cancel_text = ("cancellation: no demonstrated skill" if not cancel_outcome_positive(ev.target(CANCEL)["claims"])
                   else "cancellation: the baseline passed at least one of the first two claims, ranking only")
    answers = {
        "answer_1": f"Severe delay: PR-AUC {num(fin['pr_auc'])} on 2023 (event rate {pct(fin['prevalence'], 2)}); {cancel_text}. See the model-selection report.",
        "answer_7": f"{len(ev.lock.get('rejected_challengers', {}))} challengers were rejected on 2022 evidence before the final test. No challenger is retained.",
        "answer_9": f"Predicted-to-observed ratio {num(r22['predicted_to_observed'], 3)} on 2022 and {num(r23['predicted_to_observed'], 3)} on 2023.",
        "answer_10": f"Inside each 2023 day, the top 1% caught {pct(day1['recall'])} of severe delays at {pct(day1['precision'])} precision; the top 10% caught {pct(day10['recall'])} at {pct(day10['precision'])}.",
    }
    families = inp.csv(inp.shap_name("family"))
    if families is not None and len(families):
        top = families.sort_values("share_of_total", ascending=False).iloc[0]
        answers["answer_2"] = f"The largest feature family by mean absolute SHAP value on 2022 data was {top['family']} ({pct(top['share_of_total'], 0)}). Associations only."
    else:
        answers["answer_2"] = "See the explainability report."
    return answers


def _questions(inp: Inputs, ev: Evidence) -> str:
    answers = _answers(inp, ev)
    rows = []
    missing = []
    for question, key, paths in QUESTIONS:
        where = []
        for path in paths:
            found = path in BUILT_HERE or inp.path(path).exists()
            where.append(f"`{path}`" + ("" if found else " (not found)"))
            if not found:
                missing.append(path)
        rows.append({"Question": question, "Answer from the tables": answers.get(key, "Answered in the earlier phase reports. Not restated here.") if key else
                     "Answered in the earlier phase reports. Not restated here.", "Where": "; ".join(where)})
    parts = ["## The ten questions and where each is answered", "", md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]])]
    if missing:
        parts += ["", "Files named above that were not found when this report was built. Their results are not covered by this report:", "", bullets(sorted(set(missing)))]
    return "\n".join(parts)


def _design(ev: Evidence) -> str:
    lock = ev.lock
    lines = [
        "Data: the audited Kaggle flight dataset only, `data/raw/flights_kaggle/flights_sample_3m.csv`, 2019-01-01 to 2023-08-31, about three million flights. No weather, FAA or other source is in the pipeline from Phase 5 on.",
        f"Prediction timestamp: {lock['prediction_time']}. Only information available by then enters a feature. Outcomes of the predicted flight, delay causes and the cancellation outcome never do.",
        "Time: local times are kept as given. UTC timestamps built from an audited airport time-zone table are used for ordering and comparison. Local times at different airports are never subtracted.",
        f"Periods: fit {lock['periods']['fit']}, early stopping {lock['periods']['early_stop']}, validation {lock['periods']['validation']}, final test {lock['periods']['final_test']}.",
        "Splits are chronological. 2020 stays in training as a distribution-shift regime. The final period was untouched until the model, settings, calibrator and ranking policy were locked.",
        f"Final-test control: the lock `{ev.lock_sha[:16]}` was sealed before 2023 was read. A consumed marker records the first run. Challengers were not scored on 2023.",
        "There is no usable aircraft identifier, so no aircraft rotation or tail-level propagation is built or claimed. Network and recovery results are airport and route associations.",
    ]
    return "\n".join(["## Data, timing and leakage control", "", bullets(lines)])


def _results(inp: Inputs, ev: Evidence) -> str:
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    capacity = inp.csv(inp.final_name(SEVERE, "capacity"), required=True)
    by_group = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    fin = summary[summary["period"] == "final_2023"].copy()
    fin["interval_text"] = [interval(l, h) for l, h in zip(fin["pr_auc_low"], fin["pr_auc_high"])]
    table = md_table(fin, [("Scorer", "scorer", None), ("PR-AUC", "pr_auc", num), ("95% interval", "interval_text", None), ("Lift", "lift", lambda v: num(v, 2)), ("ROC-AUC", "roc_auc", lambda v: num(v, 3))])
    shaped = _with_interval_text(capacity)
    day = shaped[(shaped["period"] == "final_2023") & (shaped["scheme"] == "by_day")]
    rank = md_table(day, [("Capacity", "capacity", None), ("Reviewed", "reviewed", whole), ("Events caught", "events_captured", whole), ("Precision", "precision", pct),
                          ("Precision 95% interval", "precision_low", None), ("Recall", "recall", pct), ("Lift", "lift", lambda v: num(v, 2))])
    parts = ["## Results on the final test: severe delay", "",
             "All scorers on the 2023 rows. Intervals are 95% day-level bootstrap intervals. Rankings use the uncalibrated model score.", "", table, "",
             "Claims:", "", bullets(facts.claim_lines(ev.target(SEVERE)["claims"])), "",
             "Review capacity inside each 2023 day (a fixed share of that day's flights):", "", rank]
    season = _season_text(by_group, capacity)
    if season:
        parts += ["", bullets(season)]
    return "\n".join(parts)


def _cancellation(inp: Inputs, ev: Evidence) -> str:
    claims = ev.target(CANCEL)["claims"]
    summary = inp.csv(inp.final_name(CANCEL, "summary"), required=True)
    fin = locked_scorer_row(summary)
    lines = [f"2023 event rate {pct(fin['prevalence'], 2)}; PR-AUC {num(fin['pr_auc'])}, lift {num(fin['lift'], 2)}, ROC-AUC {num(fin['roc_auc'], 3)}.",
             "Claims: " + "; ".join(f"{name} {'supported' if ok else 'not supported'}" for name, ok in claim_outcomes(claims).items()) + "."]
    return "\n".join(["## Results on the final test: cancellation", "", "Cancellation used the severe-delay settings unchanged, with no tuning, calibration or thresholds.", "", bullets(lines)])


def _not_done(inp: Inputs, ev: Evidence) -> str:
    not_scored = list(ev.lock.get("not_in_this_test", []))
    not_done = missing_case_studies(inp) + [
        "held-out carrier and held-out airport experiments (the test is chronological only)",
        "any regime comparison inside the final test (one partial year)",
        "weather, delay-cause, aircraft or passenger analysis (outside the audited data)",
    ]
    return "\n".join(["## What was not done", "",
                      "Not scored on the 2023 final test. Some of these were studied in other phases on validation data; the lock only says they are not part of the final test:", "",
                      bullets(not_scored), "",
                      "Not done at all. These are gaps against the project plan or the data, not results:", "", bullets(not_done)])


def _reproduce(ev: Evidence) -> str:
    env = ev.decision.get("packages", {})
    hardware = ev.decision.get("hardware", {})
    steps = ["before the lock: `python scripts/run_phase7a_explain_and_errors.py`",
             "lock: `python scripts/run_phase7b_final_lock.py`, committed to Git before the test",
             "final test, once: `python scripts/run_phase7b_final_test.py --confirm USE-2023-ONCE`",
             "written reports: `python scripts/build_phase7_reports.py`"]
    parts = ["## Reproduction", "",
             "The Phase 7 scripts, in the order they were run. Earlier phases are described in the README. Data and model files are not in Git; the README says how to download and rebuild them.", "",
             bullets(steps), "",
             "Running the final-test script a second time is a reporting rerun: it must reproduce the stored 2023 scores or it stops."]
    if env:
        parts += ["", "Packages used for the final test: " + ", ".join(f"{k} {v}" for k, v in env.items()) + "."]
    if hardware:
        parts += ["", "Hardware: " + ", ".join(f"{k} {v}" for k, v in hardware.items()) + "."]
    return "\n".join(parts)


def build(inp: Inputs, ev: Evidence) -> str:
    return join_sections([
        header("Final technical report", inp, ev), _summary(inp, ev), _questions(inp, ev), _design(ev), _results(inp, ev), _cancellation(inp, ev),
        _not_done(inp, ev), _reproduce(ev), not_supported_section(), inputs_section(inp),
    ])

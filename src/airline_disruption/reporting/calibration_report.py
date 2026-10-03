"""reports/evaluation/calibration_and_alert_policy_report.md (Phase 7A calibration and Phase 7B review scenarios)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from airline_disruption.reporting import facts
from airline_disruption.reporting.common import Evidence, header, inputs_section, not_supported_section
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.text import (
    bullets, interval, join_sections, md_table, num, pct, ratio_text, signed, whole,
)

SEVERE, CANCEL = "severe_delay_120", "cancelled"
VALIDATION_PERIODS = ("2022 first half (selection)", "2022 second half (confirmation)", "2022 full year")
METHOD_ORDER = ("uncalibrated", "platt", "isotonic")
CAPACITY_ORDER = ("0.5%", "1%", "5%", "10%")
SCHEME_WORDS = {"whole_year": "whole period (one ranking over every flight)",
                "by_month": "within each month", "by_day": "within each day"}


def sci(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2e}"


def _order(frame: pd.DataFrame, column: str, order) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(order)}
    return frame.assign(_o=frame[column].map(rank).fillna(len(rank))).sort_values("_o", kind="stable").drop(columns="_o")


# ---------------------------------------------------------------------------------------------- scope and protocol
def _scope() -> str:
    return "\n".join([
        "## What this report covers", "",
        "Two questions, answered for the selected severe-delay model only (arrival delay of 120 minutes or more, a project definition):",
        "",
        "1. Are the model's probabilities reliable, and was the calibration method chosen fairly?",
        "2. If an analyst could review only the top K% of flights ranked by the model, how many severe delays would be in that list?",
        "",
        "The prediction time is the scheduled departure time in UTC minus two hours. The model uses only the flight's own schedule "
        "(month, weekday, scheduled local departure and arrival time, scheduled elapsed time, distance) and its carrier, origin, destination and route. "
        "It uses no outcome of the flight it scores and nothing that happened after the prediction time.",
        "",
        "The review scenarios are retrospective. They show how well a ranking separates severe delays after the fact. They are not "
        "a procedure for any airline.",
    ])


def _protocol(inp: Inputs, ev: Evidence) -> str:
    d, locked = ev.target(SEVERE), ev.locked(SEVERE)
    primary, diagnostic = locked["calibrator_primary"], locked["calibrator_diagnostic"]
    rule1, rule2 = ev.selected["rule_1"], ev.selected["rule_2"]
    periods = ev.lock["periods"]
    rule2_words = {"same_as_here": "the same rows as in the validation experiment (the early-stopping rows)", "validation_2022": "all of 2022"}
    lines = [
        f"Model: tuned LightGBM, feature set `{ev.lock['feature_set']}`, seed {ev.lock['seed']}, {d['trees']} trees.",
        f"Fit rows: {periods['fit']}. Early-stopping rows: {periods['early_stop']}. Validation: {periods['validation']}. "
        f"Final test: {periods['final_test']}.",
        f"The model was not refitted on 2022 before the final test. {ev.lock['model_policy']['reason']}",
        f"Methods compared on validation data: {', '.join(ev.selected['methods_compared'])}.",
        f"Rule 1 chose **{rule1['chosen_method']}**. {rule1['reason']}",
        f"Rule 2 (which rows to fit the final calibrator on) chose {rule2_words.get(rule2['final_test_fit_source'], rule2['final_test_fit_source'])}. {rule2['reason']}",
        f"Final calibrator: Platt (sigmoid), p = expit(a + b * logit(score)), fitted on {whole(primary['fit_rows'])} early-stopping rows "
        f"with {whole(primary['fit_events'])} events (event rate {pct(primary['fit_event_rate'], 2)}). a = {num(primary['intercept'], 3)}, "
        f"b = {num(primary['slope'], 3)}.",
        f"A second Platt calibrator, fitted on all of 2022, is shown as a diagnostic only (a = {num(diagnostic['intercept'], 3)}, "
        f"b = {num(diagnostic['slope'], 3)}). It was not the locked choice and is never used for a probability statement.",
        "No calibrator was fitted or chosen with 2023 labels.",
    ]
    return "\n".join(["## Calibration protocol", "", bullets(lines)])


# ---------------------------------------------------------------------------------------------- calibration tables
def _validation_calibration(inp: Inputs) -> str:
    summary = inp.csv("reports/tables/calibration_summary.csv")
    comparisons = inp.csv("reports/tables/phase6_calibration_comparisons.csv")
    if summary is None:
        return "## Calibration on 2022 validation data\n\n_The validation calibration table was not found._"
    a = summary[summary["experiment"] == "A"]
    rows = pd.concat([_order(a[(a["period"] == p)], "method", METHOD_ORDER) for p in VALIDATION_PERIODS if (a["period"] == p).any()])
    table = md_table(rows, [
        ("Period", "period", None), ("Method", "method", None), ("Mean predicted", "mean_predicted", lambda v: pct(v, 2)),
        ("Observed", "prevalence", lambda v: pct(v, 2)), ("Predicted / observed", "predicted_to_observed", lambda v: num(v, 3)),
        ("Brier", "brier", lambda v: num(v, 5)), ("Brier skill", "brier_skill", lambda v: num(v, 4)),
        ("Slope", "calibration_slope", lambda v: num(v, 3)), ("Intercept", "calibration_in_the_large", lambda v: num(v, 3)),
        ("ECE", "ece", lambda v: num(v, 4)),
    ])
    quarters = []
    for method in METHOD_ORDER:
        cells = {"Method": method}
        for q in (1, 2, 3, 4):
            part = a[(a["method"] == method) & (a["period"] == f"2022 Q{q}")]
            cells[f"2022 Q{q}"] = num(part["predicted_to_observed"].iloc[0], 3) if len(part) else "n/a"
        quarters.append(cells)
    quarter_table = md_table(pd.DataFrame(quarters), [(c, c, None) for c in quarters[0]])
    parts = [
        "## Calibration on 2022 validation data", "",
        "Calibrators were fitted on the early-stopping rows (2021-07 to 2021-12) and judged on 2022. The first half of 2022 chose settings and "
        "the second half confirmed them. Intercept is calibration-in-the-large (0 is ideal; positive means probabilities are too low). "
        "Slope is 1 when the spread of probabilities is right.", "", table, "",
        "Predicted-to-observed ratio by quarter of 2022 (1 is ideal):", "", quarter_table,
    ]
    if comparisons is not None:
        c = comparisons[comparisons["period"] == "2022 second half (confirmation)"]
        parts += ["", "Paired Brier-score differences on the confirmation half (negative favours the challenger; 95% day-bootstrap interval):", "",
                  md_table(c, [("Comparison", "comparison", None), ("Challenger", "challenger", None), ("Reference", "reference", None),
                               ("Difference", "difference", sci), ("Low", "ci_low", sci), ("High", "ci_high", sci)])]
    return "\n".join(parts)


def _same_pr_auc(calibration: pd.DataFrame) -> bool:
    spread = calibration.groupby("period")["pr_auc"].agg(lambda s: float(s.max() - s.min()))
    return bool((spread < 1e-9).all())


def _final_calibration(inp: Inputs, ev: Evidence) -> str:
    calibration = inp.csv(inp.final_name(SEVERE, "calibration"), required=True)
    summary = inp.csv(inp.final_name(SEVERE, "summary"), required=True)
    reliability = inp.csv(inp.final_name(SEVERE, "reliability"))
    if not _same_pr_auc(calibration):
        raise ValueError("PR-AUC differs between calibration variants inside one period. Platt cannot change a ranking, so a table is wrong.")
    table = md_table(calibration, [
        ("Period", "period", None), ("Variant", "variant", None), ("Mean predicted", "mean_predicted", lambda v: pct(v, 2)),
        ("Observed", "prevalence", lambda v: pct(v, 2)), ("Predicted / observed", "predicted_to_observed", lambda v: num(v, 3)),
        ("Brier", "brier", lambda v: num(v, 5)), ("Brier skill", "brier_skill", lambda v: num(v, 4)), ("Slope", "calibration_slope", lambda v: num(v, 3)),
        ("Intercept", "calibration_in_the_large", lambda v: num(v, 3)), ("ECE", "ece", lambda v: num(v, 4)), ("PR-AUC", "pr_auc", num),
    ])
    boot = ev.target(SEVERE)["calibration_bootstrap"]
    primary_vs = boot["primary_vs_uncalibrated"]
    diag_vs = boot["diagnostic_vs_primary"]
    locked_rows = calibration[calibration["variant"].str.contains("locked primary")]
    r22 = facts.one(locked_rows, period="validation_2022")
    r23 = facts.one(locked_rows, period="final_2023")
    lock_primary = ev.locked(SEVERE)["calibrator_primary"]
    prev22 = facts.one(summary, period="validation_2022", scorer="lightgbm_locked")["prevalence"]
    prev23 = facts.one(summary, period="final_2023", scorer="lightgbm_locked")["prevalence"]
    drift = [
        f"The event rate was {pct(lock_primary['fit_event_rate'], 2)} in the rows the calibrator was fitted on, {pct(prev22, 2)} in 2022 and {pct(prev23, 2)} in 2023.",
        f"The locked calibrator's predicted-to-observed ratio was {num(r22['predicted_to_observed'], 3)} on 2022 and {num(r23['predicted_to_observed'], 3)} on 2023. "
        f"Its slope was {num(r22['calibration_slope'], 3)} and {num(r23['calibration_slope'], 3)}.",
    ]
    if r23["predicted_to_observed"] < r22["predicted_to_observed"] and prev23 > prev22:
        drift.append("A calibrator fitted on earlier rows cannot know that the event rate has risen, so it under-predicts by more. A slope near 1 "
                     "means the ordering and spread of the probabilities held. The level is what moved. Probabilities from this model should be read "
                     "as too low when the event rate is above the calibration period, and a probability threshold fixed in advance is not stable.")
    parts = [
        "## Calibration on the 2023 final test", "",
        "The 2023 rows were never used to fit or choose anything. Rows for 2022 are repeated for comparison.", "", table, "",
        f"Locked Platt against uncalibrated, Brier difference {sci(primary_vs['difference'])} with 95% day-bootstrap interval "
        f"({sci(primary_vs['ci_low'])} to {sci(primary_vs['ci_high'])}). Negative favours the calibrated probabilities.",
        f"The diagnostic calibrator (fitted on 2022) against the locked one: {sci(diag_vs['difference'])} "
        f"({sci(diag_vs['ci_low'])} to {sci(diag_vs['ci_high'])}).",
        "", bullets(drift),
        "", "PR-AUC is identical across the variants in every period. Platt is strictly increasing, so it cannot change a ranking, "
        "and every ranking in this report uses the uncalibrated model score.",
    ]
    if reliability is not None:
        rel = reliability[reliability["variant"].str.contains("locked primary")]
        parts += ["", "Reliability of the locked probabilities on 2023 (ten equal-count bins, lowest scores first):", "",
                  md_table(rel, [("Bin", "bin", None), ("Flights", "rows", whole), ("Events", "events", whole),
                                 ("Mean predicted", "mean_predicted", lambda v: pct(v, 2)), ("Observed rate", "observed_rate", lambda v: pct(v, 2))])]
    for figure, alt in ((inp.figure("final_calibration_curve_severe_delay_120.png"), "Reliability on the 2023 final test"),
                        (inp.figure("calibration_curve.png"), "Reliability on 2022 validation data")):
        if figure:
            parts += ["", f"![{alt}]({figure})"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------- review scenarios
def _capacity_table(capacity: pd.DataFrame, period: str, scheme: str) -> str:
    part = _order(capacity[(capacity["period"] == period) & (capacity["scheme"] == scheme)], "capacity", CAPACITY_ORDER)
    return md_table(part, [
        ("Capacity", "capacity", None), ("Reviewed", "reviewed", whole), ("Events caught", "events_captured", whole),
        ("False alerts", "false_alerts", whole), ("Precision", "precision", pct), ("Precision 95% interval", "precision_low", None),
        ("Recall", "recall", pct), ("Lift", "lift", lambda v: num(v, 2)), ("False alerts per event caught", "break_even_ratio", ratio_text),
    ])


def _with_interval_text(capacity: pd.DataFrame) -> pd.DataFrame:
    out = capacity.copy()
    out["precision_low"] = [interval(l, h, 1, as_percent=True) for l, h in zip(capacity["precision_low"], capacity["precision_high"])]
    return out


def _season_text(by_group: pd.DataFrame, capacity: pd.DataFrame) -> list[str]:
    months = by_group[by_group["dimension"] == "month_2023"]
    if months.empty:
        return []
    top = facts.concentration(months, 3)
    whole1 = facts.one(capacity, period="final_2023", scheme="whole_year", capacity="1%")
    daily1 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="1%")
    lines = [
        f"Season: the three months with the most alerts at the whole-period top 1% ({', '.join(top['groups'])}) hold {pct(top['share_of_alerts'], 0)} of those alerts "
        f"while holding {pct(top['share_of_rows'], 0)} of the flights and {pct(top['share_of_events'], 0)} of the severe events.",
    ]
    same = f"Precision at the top 1% is {pct(whole1['precision'])} for the whole period and {pct(daily1['precision'])} within each day (lift {num(whole1['lift'], 2)} and {num(daily1['lift'], 2)})."
    if whole1["precision"] > daily1["precision"]:
        lines.append(same + " The gap is the part of the whole-period result that comes from knowing which months and days are busy. Quote the within-day figure for ranking skill.")
    else:
        lines.append(same + " The whole-period figure is no higher than the within-day one, so season and weekday add nothing to the ranking here. Quote the within-day figure for ranking skill.")
    return lines


def _review(inp: Inputs, ev: Evidence) -> str:
    capacity = inp.csv(inp.final_name(SEVERE, "capacity"), required=True)
    by_group = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    capacity = _with_interval_text(capacity)
    parts = [
        "## Retrospective review scenarios, severe delay, 2023", "",
        "A capacity K means the K% of flights with the highest model score are reviewed. Three ways to apply it: one ranking over the whole "
        "period, a fixed share inside each month, or a fixed share inside each day. The last is closest to how a review desk would work, because "
        "season and weekday cannot help. Rankings use the uncalibrated score. No probability threshold is used in this section.", "",
        "False alerts per event caught is the break-even ratio: reviewing pays off only if one missed severe delay is worth more than that many false alerts.",
    ]
    for scheme in ("whole_year", "by_month", "by_day"):
        parts += ["", f"**{SCHEME_WORDS[scheme].capitalize()}, 2023** ({whole(facts.one(capacity, period='final_2023', scheme=scheme, capacity='1%')['flights'])} flights, "
                      f"{whole(facts.one(capacity, period='final_2023', scheme=scheme, capacity='1%')['events'])} severe delays):", "", _capacity_table(capacity, "final_2023", scheme)]
    both = []
    for cap in CAPACITY_ORDER:
        row = {"Capacity": cap}
        for period, label in (("validation_2022", "2022"), ("final_2023", "2023")):
            for scheme, word in (("whole_year", "whole"), ("by_day", "within day")):
                r = facts.one(capacity, period=period, scheme=scheme, capacity=cap)
                row[f"Lift {label} {word}"] = num(r["lift"], 2)
        both.append(row)
    parts += ["", "Lift (precision divided by the event rate) in 2022 and 2023, side by side:", "",
              md_table(pd.DataFrame(both), [(c, c, None) for c in both[0]])]
    parts += ["", bullets(_season_text(by_group, capacity))]
    figure = inp.figure(f"final_review_capacity_{SEVERE}.png")
    if figure:
        parts += ["", f"![review capacity]({figure})"]
    return "\n".join(parts)


def _operating_points(inp: Inputs) -> str:
    points = inp.csv(inp.final_name(SEVERE, "operating_points"), required=True)
    val = points[points["period"] == "validation_2022"].set_index("operating_point")
    fin = points[points["period"] == "final_2023"].set_index("operating_point")
    rows = []
    for name in val.index:
        rows.append({"Operating point": name, "Score cutoff": num(val.loc[name, "cutoff_score"], 4),
                     "Probability at cutoff": pct(val.loc[name, "cutoff_probability"], 2),
                     "Flagged 2022": pct(val.loc[name, "alert_rate"], 2), "Flagged 2023": pct(fin.loc[name, "alert_rate"], 2),
                     "Precision 2022": pct(val.loc[name, "precision"]), "Precision 2023": pct(fin.loc[name, "precision"]),
                     "Recall 2022": pct(val.loc[name, "recall"]), "Recall 2023": pct(fin.loc[name, "recall"]),
                     "F1 2023": num(fin.loc[name, "f1"], 4)})
    ten = "capacity_10pct"
    note = ""
    if ten in val.index:
        factor = fin.loc[ten, "alert_rate"] / val.loc[ten, "alert_rate"]
        note = (f"Cutoffs were learned on 2022 and applied unchanged to 2023. The cutoff that flagged {pct(val.loc[ten, 'alert_rate'], 1)} of 2022 flights flagged "
                f"{pct(fin.loc[ten, 'alert_rate'], 1)} of 2023 flights ({num(factor, 2)} times as many). ")
        if factor > 1.05:
            note += "Scores rose between the two years, so a fixed cutoff flags more flights than planned. A capacity rule (review the top K% of each day) holds the workload fixed."
        elif factor < 0.95:
            note += "Scores fell between the two years, so a fixed cutoff flags fewer flights than planned. A capacity rule (review the top K% of each day) holds the workload fixed."
        else:
            note += "The workload held within 5%, so a fixed cutoff did not drift in this test."
    return "\n".join(["## Fixed cutoffs learned on 2022, applied to 2023", "", md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]]), "", note]).strip()


def _by_group(inp: Inputs) -> str:
    groups = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    parts = ["## Where the alerts land in 2023", "",
             "One global cutoff per capacity over the whole final period. Share of alerts is the share of the flights flagged at the top 1%."]
    hours = groups[groups["dimension"] == "scheduled_departure_local_hours"]
    if len(hours):
        shaped = facts.with_alert_shares(hours)
        parts += ["", "By scheduled local departure hour:", "",
                  md_table(shaped, [("Hours", "group", None), ("Share of flights", "share_of_rows", pct), ("Share of severe events", "share_of_events", pct),
                                    ("Share of top 1% alerts", "share_of_alerts", pct), ("Recall at top 10%", "recall_top10pct", pct), ("PR-AUC", "pr_auc", num)])]
        late = shaped[shaped["group"].isin(["14-17", "18-23"])]
        if len(late):
            parts += ["", f"Departures from 14:00 hold {pct(late['share_of_events'].sum(), 0)} of the severe events and {pct(late['share_of_alerts'].sum(), 0)} of the top 1% alerts."]
    carriers = groups[groups["dimension"] == "carrier"]
    if len(carriers):
        none = facts.groups_without_alerts(carriers)
        shaped = facts.with_alert_shares(carriers)
        flagged_none = shaped[shaped["group"].isin(none["group"])]
        eligible = int((carriers["rows"] >= 1000).sum())
        parts += ["", f"Carriers: {len(none)} of {eligible} carriers with at least 1,000 flights received no top 1% alert. They hold "
                      f"{pct(flagged_none['share_of_rows'].sum(), 0)} of the flights and {pct(flagged_none['share_of_events'].sum(), 0)} of the severe events."]
    airports = groups[groups["dimension"] == "origin_airport_top25"]
    if len(airports):
        a = facts.with_alert_shares(airports[airports["rows"] >= 1000]).sort_values("alert_rate_top1pct", ascending=False)
        pick = pd.concat([a.head(5), a.tail(5)]).drop_duplicates("group")
        parts += ["", "Origin airports with the highest and lowest share of their flights reviewed at the top 1%:", "",
                  md_table(pick, [("Airport", "group", None), ("Flights", "rows", whole), ("Event rate", "prevalence", pct),
                                  ("Flights reviewed at top 1%", "alert_rate_top1pct", lambda v: pct(v, 2)), ("Precision at top 1%", "precision_top1pct", pct),
                                  ("Recall at top 10%", "recall_top10pct", pct)])]
    unseen = groups[(groups["dimension"] == "missing_or_unseen_values") & (groups["group"] != "complete")]
    if len(unseen):
        parts += ["", "Flights with a missing number or a category not seen in the fit rows:", "",
                  md_table(unseen, [("Group", "group", None), ("Flights", "rows", whole), ("Events", "events", whole), ("Event rate", "prevalence", pct),
                                    ("Recall at top 10%", "recall_top10pct", pct), ("PR-AUC", "pr_auc", num)])]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------- utility
def _utility_reading(rows: list[dict]) -> str:
    key = "Break-even (false alerts per event caught)"
    first, last = rows[0], rows[-1]
    text = ("Read the break-even column first: it is the smallest R at which that capacity pays, and it needs no assumption. "
            f"At {first['Capacity']} it is {first[key]} false alerts per event caught, at {last['Capacity']} it is {last[key]}.")
    if float(first[key]) < float(last[key]):
        text += " The smaller review pays at a lower R, because the top of the ranking is more precise."
    else:
        text += " The break-even does not fall as the review gets smaller, so a smaller review is not more efficient here."
    return text + " Whether R is anywhere near these values is not something this data can say."


def _utility(inp: Inputs) -> str:
    utility = inp.csv(inp.final_name(SEVERE, "utility"), required=True)
    u = utility[(utility["period"] == "final_2023") & (utility["scheme"] == "by_day")]
    ratios = sorted(u["miss_to_false_alert_ratio"].unique())
    rows = []
    for cap in CAPACITY_ORDER:
        part = u[u["capacity"] == cap]
        if part.empty:
            continue
        row = {"Capacity": cap, "Break-even (false alerts per event caught)": ratio_text(part["break_even_ratio"].iloc[0])}
        for r in ratios:
            cell = part[part["miss_to_false_alert_ratio"] == r].iloc[0]
            row[f"ratio {int(r)}"] = "pays" if cell["pays_off_at_this_ratio"] else "no"
        rows.append(row)
    return "\n".join([
        "## Illustrative penalty analysis", "",
        "These weights are assumptions made for analysis. They are not airline costs and nothing here estimates a saving. A caught severe delay earns R "
        "units, a false alert costs 1 unit, and R is the assumed ratio of the cost of a missed severe delay to the cost of a false alert. The measured "
        "quantities (events caught, false alerts) are in the tables above and do not depend on R.", "",
        "Does reviewing the top K% inside each day beat doing nothing, at each assumed ratio R? 2023, fixed daily capacity:", "",
        md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]]), "",
        _utility_reading(rows),
    ])


# ---------------------------------------------------------------------------------------------- cancellation
def _cancellation(inp: Inputs, ev: Evidence) -> str:
    summary = inp.csv(inp.final_name(CANCEL, "summary"), required=True)
    capacity = inp.csv(inp.final_name(CANCEL, "capacity"), required=True)
    groups = inp.csv(inp.final_name(CANCEL, "by_group"))
    d = ev.target(CANCEL)
    fin = summary[summary["period"] == "final_2023"]
    fin = fin.assign(interval_text=[interval(l, h, 4) for l, h in zip(fin["pr_auc_low"], fin["pr_auc_high"])])
    table = md_table(fin, [("Scorer", "scorer", None), ("PR-AUC", "pr_auc", num), ("95% interval", "interval_text", None),
                           ("Lift", "lift", lambda v: num(v, 2)), ("ROC-AUC", "roc_auc", lambda v: num(v, 3))])
    lifts = []
    for cap in CAPACITY_ORDER:
        row = {"Capacity": cap}
        for scheme, word in (("whole_year", "whole period"), ("by_day", "within day")):
            r = facts.one(capacity, period="final_2023", scheme=scheme, capacity=cap)
            row[f"Lift, {word}"] = num(r["lift"], 2)
        lifts.append(row)
    claims = d["claims"]
    c1, c2, c3 = (claims[k]["supported"] for k in ("C1_skill_over_chance", "C2_beats_history", "C3_ranking_not_only_seasonal"))
    parts = [
        "## Cancellation", "",
        f"Cancellation was run as an untuned baseline: the severe-delay settings were reused, with no calibration and no thresholds. "
        f"The 2023 event rate is {pct(d['event_rate'], 2)} ({whole(d['events'])} cancelled of {whole(d['rows'])} flights).", "", table, "",
        "Review lift, whole period and within each day:", "", md_table(pd.DataFrame(lifts), [(c, c, None) for c in lifts[0]]),
    ]
    if groups is not None:
        months = groups[(groups["dimension"] == "month_2023") & (groups["rows"] >= 1000)]
        if len(months):
            shaped = facts.with_alert_shares(months)
            silent = shaped[shaped["alerts"] < 0.5]["group"].tolist()
            top = shaped.sort_values("alerts", ascending=False).iloc[0]
            none_text = "No month went without a top 1% alert." if not silent else f"{len(silent)} of {len(months)} months received no top 1% alert ({', '.join(silent)})."
            parts += ["", f"Month pattern: {pct(top['share_of_alerts'], 0)} of the top 1% alerts fall in {top['group']}, which holds {pct(top['share_of_rows'], 0)} of the flights; "
                          f"precision there was {pct(top['precision_top1pct'])} against a cancellation rate of {pct(top['prevalence'], 2)}. {none_text}"]
    if not c1 and not c2:
        verdict = "Result: cancellation is reported as a negative result. The model shows no demonstrated skill over chance on 2023 and does not beat the best history baseline."
        if claims["C2_beats_history"].get("difference_high", 0) < 0:
            verdict += " The baseline is better, with the whole interval below zero."
        verdict += (" Ranking inside a single day does beat the event rate, which is a weaker statement than a useful alert list." if c3
                    else " Ranking inside a single day does not beat the event rate either.")
        verdict += " No calibration, threshold or alert policy was built for it."
        parts += ["", verdict]
    parts += ["", "Pre-registered claims for cancellation:", "", bullets(facts.claim_lines(claims))]
    return "\n".join(parts)


def _claims(ev: Evidence) -> str:
    claims = ev.target(SEVERE)["claims"]
    return "\n".join(["## Pre-registered claims, severe delay", "",
                      "Fixed in the lock before 2023 was read. These are the only pass or fail statements in the final test; everything else is description.", "",
                      bullets(facts.claim_lines(claims))])


def _limits(ev: Evidence) -> str:
    stray = sum(ev.target(t).get("stray_rows_on_2023_09_01_utc", 0) for t in (SEVERE, CANCEL))
    lines = [
        "One period only (January to August 2023). There is no regime comparison inside the final test, and no claim about a full year: the missing autumn and winter months are not scored.",
        "The model has seen nothing after 2021-06 and the calibrator nothing after 2021-12. The test measures how a model frozen in mid-2021 does on 2023.",
        "The model uses the schedule only. It has no state of the airport on the day, so it cannot see disruption building up before the prediction time.",
        f"{whole(stray)} rows across both targets have a UTC prediction time on 2023-09-01 (late flights of 31 August in western time zones). They belong to the final period by flight date.",
        "Review precision and lift are measured against every flight, so they include all the ways a flight can be late. They do not say which delays an analyst could have acted on.",
        "The 2023 event rate was seen in earlier descriptive work before the test and is disclosed in DEC-027. No model, calibration or policy choice used it.",
    ]
    return "\n".join(["## Limits", "", bullets(lines)])


def build(inp: Inputs, ev: Evidence) -> str:
    return join_sections([
        header("Calibration and alert-policy report", inp, ev),
        _scope(), _protocol(inp, ev), _validation_calibration(inp), _final_calibration(inp, ev), _review(inp, ev),
        _operating_points(inp), _by_group(inp), _utility(inp), _claims(ev), _cancellation(inp, ev), _limits(ev),
        not_supported_section(), inputs_section(inp),
    ])

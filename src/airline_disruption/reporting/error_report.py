"""reports/evaluation/error_analysis_report.md (where the locked model fails, on 2022 validation data and the 2023 final test)."""

from __future__ import annotations

import pandas as pd

from airline_disruption.reporting import facts
from airline_disruption.reporting.common import Evidence, header, inputs_section, missing_case_studies, not_supported_section
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.text import bullets, join_sections, md_table, num, pct, ratio_text, whole

SEVERE, CANCEL = "severe_delay_120", "cancelled"
TARGET_WORDS = {SEVERE: "severe delay (arrival delay of 120 minutes or more)", CANCEL: "cancellation"}
HOUR_ORDER = ("00-05", "06-09", "10-13", "14-17", "18-23")
MIN_ROWS = 1000


def _purpose() -> str:
    return "\n".join([
        "## Purpose and rules", "",
        "This report looks for where the locked model fails. It is written to expose limits, not to show the cases that worked.", "",
        bullets([
            "Alerts use one global cutoff for the top 1% and top 10% of all flights in the period, never a cutoff per group. A group whose flights all score low is shown with its low alert rate.",
            "PR-AUC is left blank for a group with fewer than 30 events or 30 non-events. Groups under 1,000 flights are not named as patterns.",
            "Realised delay minutes and cancellation outcomes are used to describe errors after the fact. They are never model inputs.",
            "2022 is validation data (the first half chose settings). 2023 is the final test, scored once.",
        ]),
    ])


# ---------------------------------------------------------------------------------------------- regimes
def _regime_table(inp: Inputs, ev: Evidence, target: str) -> tuple[str, str]:
    backtest = inp.csv(inp.step11_name(target, "backtest_models"))
    summary = inp.csv(inp.final_name(target, "summary"), required=True)
    capacity = inp.csv(inp.step11_name(target, "review_capacity"))
    rows = []
    if backtest is not None:
        for r in backtest.sort_values("eval_year").itertuples():
            rows.append({"Evaluated on": r.regime, "Trees": whole(r.trees), "Fit rows": whole(r.fit_rows), "Flights": whole(r.eval_rows), "Events": whole(r.events),
                         "Event rate": pct(r.prevalence, 2), "PR-AUC": num(r.pr_auc), "Lift": num(r.pr_auc_lift, 2)})
    fin = facts.one(summary, period="final_2023", scorer="lightgbm_locked")
    d = ev.target(target)
    rows.append({"Evaluated on": "2023 final test (locked model)", "Trees": whole(d.get("trees")), "Fit rows": whole(ev.locked(target).get("fit_rows")), "Flights": whole(fin["rows"]), "Events": whole(fin["events"]),
                 "Event rate": pct(fin["prevalence"], 2), "PR-AUC": num(fin["pr_auc"]), "Lift": num(fin["lift"], 2)})
    table = md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]])
    text = ""
    if capacity is not None:
        lifts = []
        for year in sorted(capacity["eval_year"].unique()):
            row = {"Year": str(year)}
            for scheme, word in (("whole_year", "whole"), ("by_day", "within day")):
                for cap in ("1%", "10%"):
                    part = capacity[(capacity["eval_year"] == year) & (capacity["scheme"] == scheme) & (capacity["capacity"] == cap)]
                    row[f"Lift top {cap}, {word}"] = num(part["lift"].iloc[0], 2) if len(part) else "n/a"
            lifts.append(row)
        text = "Review lift by evaluation year (precision divided by that year's event rate):\n\n" + md_table(pd.DataFrame(lifts), [(c, c, None) for c in lifts[0]])
    return table, text


def _regimes(inp: Inputs, ev: Evidence) -> str:
    parts = ["## Regimes: do the results hold across periods?", "",
             "Each row is a different model, fitted on the years before the year it is scored on (expanding window), except the last row, which is the locked model. "
             "2020 is kept as a distribution-shift regime, not removed. The years are not comparable on PR-AUC alone because the event rate changes; use lift."]
    for target in (SEVERE, CANCEL):
        table, text = _regime_table(inp, ev, target)
        parts += ["", f"**{TARGET_WORDS[target].capitalize()}**", "", table]
        if text:
            parts += ["", text]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------- time and hour
def _alerts_by_period(inp: Inputs) -> str:
    q22 = inp.csv(inp.step11_name(SEVERE, "review_by_group_2022"))
    g23 = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    parts = ["## Time of year", "", "Quarter of 2022 and month of 2023, severe delay. Alert rate is the share of that period's flights inside the global top 1%."]
    cols = [("Period", "group", None), ("Flights", "rows", whole), ("Event rate", "prevalence", lambda v: pct(v, 2)),
            ("Flights in top 1%", "alert_rate_top1pct", lambda v: pct(v, 2)), ("Precision at top 1%", "precision_top1pct", pct),
            ("Recall at top 10%", "recall_top10pct", pct), ("PR-AUC", "pr_auc", num)]
    notes = []
    if q22 is not None:
        quarters = q22[q22["dimension"] == "quarter_2022"]
        if len(quarters):
            parts += ["", "2022, by quarter:", "", md_table(quarters, cols)]
            top = facts.concentration(quarters, 2)
            notes.append(f"2022: two quarters ({', '.join(top['groups'])}) hold {pct(top['share_of_alerts'], 0)} of the top 1% alerts and {pct(top['share_of_rows'], 0)} of the flights.")
    months = g23[g23["dimension"] == "month_2023"]
    parts += ["", "2023, by month:", "", md_table(months, cols)]
    top = facts.concentration(months, 3)
    if top["groups"]:
        notes.append(f"2023: three months ({', '.join(top['groups'])}) hold {pct(top['share_of_alerts'], 0)} of the top 1% alerts and {pct(top['share_of_rows'], 0)} of the flights.")
    parts += ["", bullets(notes), "",
              "A global cutoff sends alerts to the months where the model has learned that scores run high. Month-level differences in the event rate that the model "
              "did not learn show up as months with a low alert rate and low recall."]
    return "\n".join(parts)


def _hours(inp: Inputs) -> str:
    q22 = inp.csv(inp.step11_name(SEVERE, "review_by_group_2022"))
    g23 = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    h23 = facts.with_alert_shares(g23[g23["dimension"] == "scheduled_departure_local_hours"])
    h23_10 = facts.with_alert_shares(g23[g23["dimension"] == "scheduled_departure_local_hours"], "alert_rate_top10pct")
    h23["share_of_alerts_top10"] = h23_10["share_of_alerts"]
    rows = []
    h22 = None
    if q22 is not None:
        h22 = facts.with_alert_shares(q22[q22["dimension"] == "scheduled_departure_local_hours"])
    for hour in HOUR_ORDER:
        r23 = h23[h23["group"] == hour]
        if r23.empty:
            continue
        row = {"Local departure hour": hour, "Share of flights 2023": pct(r23["share_of_rows"].iloc[0], 0), "Share of severe events 2023": pct(r23["share_of_events"].iloc[0], 0),
               "Share of top 10% alerts 2023": pct(r23["share_of_alerts_top10"].iloc[0], 0), "Recall at top 10% 2023": pct(r23["recall_top10pct"].iloc[0])}
        if h22 is not None and (h22["group"] == hour).any():
            row["Recall at top 10% 2022"] = pct(h22[h22["group"] == hour]["recall_top10pct"].iloc[0])
        rows.append(row)
    morning = h23[h23["group"].isin(["06-09"])]
    text = ""
    if len(morning):
        m = morning.iloc[0]
        text = (f"Departures from 06:00 to 09:59 are {pct(m['share_of_rows'], 0)} of the 2023 flights and {pct(m['share_of_events'], 0)} of the severe events. "
                f"A top 10% review catches {pct(m['recall_top10pct'])} of them.")
    return "\n".join(["## Time of day", "", md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]]), "", text]).strip()


# ---------------------------------------------------------------------------------------------- carriers, airports, other groups
def _entities(inp: Inputs) -> str:
    g23 = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    parts = ["## Carriers and airports (2023)", ""]
    carriers = g23[(g23["dimension"] == "carrier") & (g23["rows"] >= MIN_ROWS)].sort_values("recall_top10pct")
    cols = [("Group", "group", None), ("Flights", "rows", whole), ("Event rate", "prevalence", lambda v: pct(v, 2)), ("Flights in top 1%", "alert_rate_top1pct", lambda v: pct(v, 2)),
            ("Recall at top 10%", "recall_top10pct", pct), ("PR-AUC", "pr_auc", num)]
    parts += ["Carriers, lowest recall at the top 10% first:", "", md_table(carriers, cols)]
    none = facts.groups_without_alerts(carriers)
    if len(none):
        biggest = none.sort_values("rows", ascending=False).iloc[0]
        noun = "carrier received" if len(none) == 1 else "carriers received"
        parts += ["", f"{len(none)} {noun} no top 1% alert. The largest of them, {biggest['group']}, has {pct(biggest['rows'] / carriers['rows'].sum(), 0)} of the flights and an event rate of {pct(biggest['prevalence'], 2)}."]
    airports = g23[(g23["dimension"] == "origin_airport_top25") & (g23["rows"] >= MIN_ROWS)].sort_values("recall_top10pct")
    if len(airports):
        pick = pd.concat([airports.head(5), airports.tail(5)]).drop_duplicates("group")
        parts += ["", "Origin airports, five lowest and five highest recall at the top 10%:", "", md_table(pick, cols)]
    return "\n".join(parts)


def _other_groups(inp: Inputs) -> str:
    g23 = inp.csv(inp.final_name(SEVERE, "by_group"), required=True)
    parts = ["## Route volume, weekday, missing values and score deciles (2023)", ""]
    cols = [("Group", "group", None), ("Flights", "rows", whole), ("Events", "events", whole), ("Event rate", "prevalence", lambda v: pct(v, 2)),
            ("Recall at top 10%", "recall_top10pct", pct), ("PR-AUC", "pr_auc", num)]
    for dimension, title in (("route_volume_in_fit_period", "Route volume in the fit period"), ("day_of_week", "Day of week (0 is Monday)"),
                             ("missing_or_unseen_values", "Missing or unseen values")):
        part = g23[g23["dimension"] == dimension]
        if len(part):
            parts += [f"{title}:", "", md_table(part, cols), ""]
    deciles = g23[g23["dimension"] == "score_decile"].sort_values("group")
    if len(deciles):
        rising = facts.is_increasing(deciles["prevalence"])
        low, high = deciles.iloc[0], deciles.iloc[-1]
        parts += ["Score deciles:", "", md_table(deciles, cols), "",
                  f"The event rate {'rises with every decile' if rising else 'does not rise with every decile'}, from {pct(low['prevalence'], 2)} in the lowest to {pct(high['prevalence'], 2)} in the highest "
                  f"({num(high['prevalence'] / low['prevalence'], 1)} times)."]
    return "\n".join(parts).strip()


def _severity(inp: Inputs) -> str:
    eb = inp.csv(inp.error_breakdown_name())
    sev23 = inp.csv(inp.final_name(SEVERE, "severity"), required=True)
    s23 = sev23[sev23["dimension"] == "event_delay_band_minutes"]
    rows = []
    s22 = eb[eb["dimension"] == "event_delay_band_minutes"] if eb is not None else None
    for r in s23.itertuples():
        row = {"Delay band (minutes)": r.group, "Events 2023": whole(r.events), "Share of events 2023": pct(r.share_of_events, 1),
               "Recall at top 1% 2023": pct(r.recall_top1pct), "Recall at top 10% 2023": pct(r.recall_top10pct)}
        if s22 is not None and (s22["group"] == r.group).any():
            row["Recall at top 10% 2022"] = pct(s22[s22["group"] == r.group]["recall_top10pct"].iloc[0])
        rows.append(row)
    rising = facts.is_increasing(s23["recall_top10pct"])
    text = (f"Recall at the top 10% runs from {pct(s23['recall_top10pct'].min())} to {pct(s23['recall_top10pct'].max())} across delay bands. "
            + ("It rises with every band, so longer delays are found more often." if rising else "It does not rise steadily with the size of the delay, so the model is not finding the worst delays more often than the mild ones."))
    return "\n".join(["## Size of the delay", "", "Among severe events only. The realised delay is used to describe who the model misses, after the fact.", "",
                      md_table(pd.DataFrame(rows), [(c, c, None) for c in rows[0]]), "", text])


# ---------------------------------------------------------------------------------------------- cases
def _cases(inp: Inputs) -> str:
    cases = inp.csv(f"reports/tables/phase7a{inp.infix}_case_studies.csv")
    if cases is None:
        return "## Case studies\n\n_The case-study table was not found._"
    cases = cases.copy()
    cases["trip"] = cases["origin"].astype(str) + " to " + cases["destination"].astype(str)
    summary = cases.groupby("category").agg(cases=("pick", "size"), median_rank=("score_rank_in_h2", "median"), median_delay=("arrival_delay_minutes", "median")).reset_index()
    return "\n".join([
        "## Case studies", "",
        "The full table (typical cases at the 25th, 50th and 75th percentile of each category, plus the most confident mistakes) is "
        "`reports/tables/phase7a_case_studies.csv`. The explainability report shows the typical case of each category with the features behind its score. "
        "All cases are validation flights from the second half of 2022.", "",
        md_table(summary, [("Category", "category", None), ("Cases", "cases", whole), ("Median score rank", "median_rank", whole), ("Median arrival delay (min)", "median_delay", lambda v: num(v, 0))]),
    ])


# ---------------------------------------------------------------------------------------------- cancellation
def _cancellation(inp: Inputs, ev: Evidence) -> str:
    q22 = inp.csv(inp.step11_name(CANCEL, "review_by_group_2022"))
    g23 = inp.csv(inp.final_name(CANCEL, "by_group"), required=True)
    capacity = inp.csv(inp.final_name(CANCEL, "capacity"), required=True)
    claims = ev.target(CANCEL)["claims"]
    c1, c2, c3 = (claims[k]["supported"] for k in ("C1_skill_over_chance", "C2_beats_history", "C3_ranking_not_only_seasonal"))
    parts = ["## Cancellation: where the ranking goes", ""]
    cols_q = [("Period", "group", None), ("Flights", "rows", whole), ("Cancellation rate", "prevalence", lambda v: pct(v, 2)),
              ("Flights in top 1%", "alert_rate_top1pct", lambda v: pct(v, 2)), ("Precision at top 1%", "precision_top1pct", pct), ("Recall at top 10%", "recall_top10pct", pct)]
    notes = []
    if q22 is not None:
        quarters = q22[q22["dimension"] == "quarter_2022"]
        if len(quarters):
            shaped = facts.with_alert_shares(quarters)
            busiest = shaped.sort_values("prevalence", ascending=False).iloc[0]
            most = shaped.sort_values("alerts", ascending=False).iloc[0]
            parts += ["2022, by quarter:", "", md_table(quarters, cols_q), ""]
            notes.append(f"2022: the quarter with the highest cancellation rate is {busiest['group']} ({pct(busiest['prevalence'], 2)}) and it holds {pct(busiest['share_of_alerts'], 0)} of the top 1% alerts. "
                         f"The quarter with the most alerts is {most['group']} ({pct(most['share_of_alerts'], 0)} of them, {pct(most['share_of_rows'], 0)} of the flights).")
    months = g23[(g23["dimension"] == "month_2023") & (g23["rows"] >= MIN_ROWS)]
    if len(months):
        shaped = facts.with_alert_shares(months)
        silent = shaped[shaped["alerts"] < 0.5]["group"].tolist()
        most = shaped.sort_values("alerts", ascending=False).iloc[0]
        parts += ["2023, by month:", "", md_table(months, cols_q), ""]
        notes.append(f"2023: {pct(most['share_of_alerts'], 0)} of the top 1% alerts fall in {most['group']}, which holds {pct(most['share_of_rows'], 0)} of the flights. "
                     + ("No month went without an alert." if not silent else f"{len(silent)} of {len(months)} months got none ({', '.join(silent)})."))
    whole1 = facts.one(capacity, period="final_2023", scheme="whole_year", capacity="10%")
    day1 = facts.one(capacity, period="final_2023", scheme="by_day", capacity="10%")
    notes.append(f"Lift at the top 10%: {num(whole1['lift'], 2)} over the whole period and {num(day1['lift'], 2)} inside each day.")
    parts += [bullets(notes), ""]
    if not c1 and not c2:
        parts += ["Both pre-registered tests of skill failed on 2023. A model that sends most of its alerts to a few periods cannot rank across the year, "
                  "and the periods it prefers came from the training years."]
        parts[-1] += (" Ranking inside a single day does beat the event rate (claim C3 holds), so some within-day signal exists, but it does not make a useful alert list."
                      if c3 else " Ranking inside a single day does not beat the event rate either.")
    else:
        parts += ["See the claims in the calibration and alert-policy report."]
    return "\n".join(parts).strip()


def _open_items(inp: Inputs) -> str:
    d = inp.json(f"reports/tables/phase7a{inp.infix}_explain_decision.json")
    items = list(d.get("not_evaluable", [])) if d else []
    items += ["2023 results by regime (the final period is one partial year, so there is no regime comparison inside it)",
              "held-out carrier or airport experiments (not run; the main test is chronological)"]
    parts = ["## Not evaluated", "", "These were not analysed in this report, and the reason is the data or the design, not a result:", "", bullets(items)]
    skipped = missing_case_studies(inp)
    if skipped:
        parts += ["", "Case studies the project plan requires and that were NOT built. The error analysis above is therefore incomplete against that plan:", "", bullets(skipped)]
    return "\n".join(parts)


def build(inp: Inputs, ev: Evidence) -> str:
    return join_sections([header("Error analysis report", inp, ev), _purpose(), _regimes(inp, ev), _alerts_by_period(inp), _hours(inp), _entities(inp),
                          _other_groups(inp), _severity(inp), _cases(inp), _cancellation(inp, ev), _open_items(inp), not_supported_section(), inputs_section(inp)])

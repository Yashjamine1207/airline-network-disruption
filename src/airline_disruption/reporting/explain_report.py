"""reports/evaluation/explainability_report.md (Phase 7A explanations of the locked LightGBM and the sequence checks)."""

from __future__ import annotations

import math

import pandas as pd

from airline_disruption.reporting import facts
from airline_disruption.reporting.common import Evidence, header, inputs_section, not_supported_section
from airline_disruption.reporting.inputs import ARCHITECTURES, Inputs
from airline_disruption.reporting.text import bullets, interval, join_sections, md_table, num, pct, signed, whole

CASE_ORDER = ("true_positive", "false_positive", "false_negative", "low_risk_correct",
              "high_confidence_false_positive", "high_confidence_false_negative")
CASE_WORDS = {
    "true_positive": "Correct high-risk alert (top 1%, severe)",
    "false_positive": "False alarm (top 1%, not severe)",
    "false_negative": "Missed severe delay (outside the top 10%)",
    "low_risk_correct": "Low-risk and correct (not severe, lower-half score)",
    "high_confidence_false_positive": "Most confident false alarm (highest-scoring non-event)",
    "high_confidence_false_negative": "Most confident miss (lowest-scoring event)",
}


def _expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _meaning() -> str:
    return "\n".join([
        "## What an explanation says here", "",
        "SHAP values describe how the model reaches its score. A large positive value for an airport means the model has learned, from the 2019 to mid-2021 "
        "flights it was fitted on, that flights from that airport go with more severe delays. It is an association inside the model. It is not a cause of delay, "
        "and it says nothing about weather, aircraft or crews, which the data does not hold.", "",
        "Values are in log-odds. A positive value moves a flight's score above the average flight, a negative value below. Where features overlap (an airport, a "
        "route and a carrier often say the same thing) the credit is shared between them, so read a family total, not a single bar, when two features are close.",
    ])


def _model_and_data(inp: Inputs, ev: Evidence) -> str:
    d = inp.json(f"reports/tables/phase7a{inp.infix}_explain_decision.json")
    if d is None:
        return "## Model and data explained\n\n_The explanation decision file was not found._"
    shap = d["shap"]
    base = _expit(shap["expected_value_log_odds"])
    lines = [
        f"Model: tuned LightGBM, feature set `{d['feature_set']}`, seed {d['seed']}, {d['best_iteration']} trees. This is the model that was locked and later scored on 2023.",
        f"SHAP values were computed on {whole(shap['rows'])} flights from the {shap['period']} (validation data). They were checked to add up to the model's own output "
        f"(largest gap {shap['additivity_gap']:.1e} log-odds).",
        f"The average flight sits at {num(shap['expected_value_log_odds'], 3)} log-odds, a score of {pct(base, 2)}.",
        "The 2023 final test was used for scoring only. No explanation, feature choice or model choice was drawn from it.",
    ]
    return "\n".join(["## Model and data explained", "", bullets(lines)])


def _global(inp: Inputs) -> str:
    importance = inp.csv(inp.shap_name("global"))
    families = inp.csv(inp.shap_name("family"))
    if importance is None or families is None:
        return "## Global importance\n\n_The SHAP importance tables were not found._"
    top = importance.head(10)
    parts = [
        "## Global importance", "",
        md_table(top, [("Feature", "feature", None), ("Family", "family", None), ("Mean absolute SHAP", "mean_abs_shap", lambda v: num(v, 4)),
                       ("Share of total", "share_of_total", pct)]), "",
        "By feature family (shares add to 100%):", "",
        md_table(families, [("Family", "family", None), ("Mean absolute SHAP", "mean_abs_shap", lambda v: num(v, 4)), ("Share of total", "share_of_total", pct)]),
    ]
    lead = families.sort_values("share_of_total", ascending=False).iloc[0]
    parts += ["", f"The largest family is {lead['family']} at {pct(lead['share_of_total'])}. Carrier, airport and route together hold "
                  f"{pct(families[families['family'].isin(['carrier', 'airport', 'route'])]['share_of_total'].sum())}. "
                  "The model's features are the flight's schedule and its carrier, airports and route, so what it can express is mainly when, where and with whom a flight runs."]
    figure = inp.figure("shap_summary.png")
    if figure:
        parts += ["", f"![SHAP summary]({figure})"]
    return "\n".join(parts)


def _direction(inp: Inputs) -> str:
    numeric = inp.csv(inp.shap_name("numeric_effects"))
    categorical = inp.csv(inp.shap_name("categorical_effects"))
    parts = ["## Direction of the effects", "", "Mean SHAP value by fifth of the feature (numeric) and by level (categories). Positive means a higher score."]
    if numeric is not None:
        for feature in ("scheduled_departure_month", "scheduled_departure_hour_local", "scheduled_arrival_hour_local", "distance_miles"):
            part = numeric[numeric["feature"] == feature]
            if len(part):
                part = part.assign(range=[f"{lo:g} to {hi:g}" for lo, hi in zip(part["value_low"], part["value_high"])])
                parts += ["", f"`{feature}`:", "", md_table(part, [("Range", "range", None), ("Flights", "rows", whole), ("Mean SHAP", "mean_shap", lambda v: signed(v, 3))])]
    if categorical is not None:
        for feature in ("carrier_identifier", "origin_airport"):
            part = categorical[categorical["feature"] == feature]
            if len(part):
                part = pd.concat([part[part["direction"] == "highest"].head(5), part[part["direction"] == "lowest"].head(5)])
                parts += ["", f"`{feature}`, five levels with the highest and five with the lowest mean SHAP value:", "",
                          md_table(part, [("Direction", "direction", None), ("Level", "level", None), ("Flights", "rows", whole), ("Mean SHAP", "mean_shap", lambda v: signed(v, 3))])]
    return "\n".join(parts)


def _cases(inp: Inputs) -> str:
    cases = inp.csv(f"reports/tables/phase7a{inp.infix}_case_studies.csv")
    if cases is None:
        return "## Local explanations and case studies\n\n_The case-study table was not found._"
    counts = cases.groupby("category").size()
    typical = cases[cases["pick"] == "typical_q50"].copy()
    extremes = cases[cases["category"].str.startswith("high_confidence")].copy()
    for frame in (typical, extremes):
        frame["case"] = frame["category"].map(CASE_WORDS)
        frame["trip"] = frame["origin"].astype(str) + " to " + frame["destination"].astype(str)
    cols = [("Case", "case", None), ("Trip", "trip", None), ("Carrier", "carrier", None), ("Local departure hour", "local_departure_hour", lambda v: f"{float(v):g}"),
            ("Score rank in 2022 H2", "score_rank_in_h2", whole), ("Calibrated probability", "calibrated_probability", lambda v: pct(v, 2)),
            ("Severe", "event", lambda v: "yes" if int(v) == 1 else "no"), ("Arrival delay (min)", "arrival_delay_minutes", lambda v: num(v, 0))]
    typical = typical.assign(_o=typical["category"].map({c: i for i, c in enumerate(CASE_ORDER)})).sort_values("_o")
    parts = [
        "## Local explanations and case studies", "",
        f"Cases were chosen by a fixed rule from score and label only, on the second half of 2022 (validation data): {', '.join(f'{k} {v}' for k, v in counts.items())} rows. "
        "Each category shows the typical case (the median by score) here; the quartile cases and extremes are in `reports/tables/phase7a_case_studies.csv`. "
        "The arrival delay is the realised outcome and is used to describe a case after the fact, never as a model input.", "",
        "Typical case of each category:", "", md_table(typical, cols), "", "The most confident mistakes:", "", md_table(extremes, cols), "",
        "The features that moved each typical case most, in log-odds:", "",
        bullets([f"{CASE_WORDS[r.category]}: {r.top_features}" for r in typical.itertuples()]),
    ]
    figure = inp.figure("local_flight_explanation.png")
    if figure:
        parts += ["", f"![local explanations]({figure})"]
    return "\n".join(parts)


def _sequence(inp: Inputs, ev: Evidence) -> str:
    parts = ["## Sequence-model checks", ""]
    found = False
    for arch in ARCHITECTURES:
        table = inp.csv(f"reports/tables/phase7a_sequence_perturbation_{arch}{inp.infix}.csv", quiet=True)
        if table is None:
            continue
        found = True
        decision = inp.json(f"reports/tables/phase7a_sequence_perturbation_{arch}{inp.infix}_decision.json", quiet=True)
        availability = inp.csv(f"reports/tables/phase7a_sequence_availability_{arch}{inp.infix}.csv", quiet=True)
        shown = table[table["perturbation"] != "baseline"].copy()
        shown["interval"] = [interval(l, h, 5) for l, h in zip(shown["ci_low"], shown["ci_high"])]
        clear = shown[~shown["verdict"].astype(str).str.startswith("no clear change")]
        title = f"**{arch.upper()}**"
        if decision:
            title += (f" ({whole(decision['rows_scored'])} flights of the {decision['period']}, lookback of {decision['lookback_bins']} bins; "
                      f"reloaded model reproduces the stored PR-AUC to {decision['reproduction_check']['pr_auc_difference']:.1e})")
        parts += [title, "", md_table(shown, [("Input change", "perturbation", None), ("What was done", "description", None),
                                              ("Change in PR-AUC", "delta_pr_auc", lambda v: signed(v, 5)), ("95% interval", "interval", None), ("Verdict", "verdict", None)]), ""]
        parts += [f"{len(clear)} of {len(shown)} input changes moved PR-AUC clearly (interval excludes zero)"
                  + (": " + ", ".join(clear["perturbation"]) + "." if len(clear) else "."), ""]
        if availability is not None:
            parts += [md_table(availability, [("Sequence availability", "group", None), ("Flights", "rows", whole), ("Event rate", "prevalence", lambda v: pct(v, 2)),
                                              ("PR-AUC", "pr_auc", num), ("Lift", "lift", lambda v: num(v, 2))]), ""]
    if not found:
        parts += ["_No sequence perturbation tables were found._", ""]
    rejected = ev.lock.get("rejected_challengers", {})
    parts += ["Perturbation means changing what the trained network sees (removing the lookback, shuffling it between flights, keeping only the newest bins, hiding one quarter) "
              "and measuring how PR-AUC moves. It shows what the network uses, not why a flight is delayed. Attention weights are not reported as explanations."]
    if rejected:
        parts += ["", "Challengers rejected on validation evidence before the final test (none was scored on 2023):", "", bullets([f"{name}: {why}" for name, why in rejected.items()])]
    return "\n".join(parts)


def _link(inp: Inputs) -> str:
    families = inp.csv(inp.shap_name("family"))
    importance = inp.csv(inp.shap_name("global"))
    if families is None or importance is None:
        return ""
    cal = families[families["family"] == "calendar and clock"]["share_of_total"]
    ident = families[families["family"].isin(["carrier", "airport", "route"])]["share_of_total"].sum()

    def share(feature: str) -> str:
        part = importance[importance["feature"] == feature]["share_of_total"]
        return pct(part.iloc[0]) if len(part) else "n/a"

    text = (f"The calendar and clock family holds {pct(cal.iloc[0]) if len(cal) else 'n/a'} of the explanation mass (departure hour {share('scheduled_departure_hour_local')}, "
            f"month {share('scheduled_departure_month')}) and carrier, airport and route hold {pct(ident)}. A model built from these ranks flights by time of day, time of year and "
            "who flies where. `reports/evaluation/error_analysis_report.md` shows where that pattern fails on 2022 and 2023.")
    return "\n".join(["## How this connects to the 2023 errors", "", text])


def _limits(inp: Inputs) -> str:
    d = inp.json(f"reports/tables/phase7a{inp.infix}_explain_decision.json")
    open_items = d.get("not_evaluable", []) if d else []
    lines = ["SHAP was computed on validation data and describes the locked model, not the 2023 flights.",
             "Features that overlap share credit, so single-feature rankings can move if a related feature is added.",
             "No explanation here is a statement about why a flight was late."]
    lines += [f"Not evaluated here: {item}." for item in open_items]
    return "\n".join(["## Limits", "", bullets(lines)])


def build(inp: Inputs, ev: Evidence) -> str:
    return join_sections([header("Explainability report", inp, ev), _meaning(), _model_and_data(inp, ev), _global(inp), _direction(inp), _cases(inp),
                          _sequence(inp, ev), _link(inp), _limits(inp), not_supported_section(), inputs_section(inp)])

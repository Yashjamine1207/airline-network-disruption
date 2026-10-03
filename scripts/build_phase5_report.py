"""Generate the retrospective Phase 5 network resilience report."""

from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
OUTPUT = ROOT / "reports/phase5_network_resilience_report.md"


def load(name):
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def percent(value):
    return f"{100 * value:.2f}%"


def main():
    network = load("yearly_network_summary.csv")
    regime = load("network_regime_summary.csv")
    recovery = load("airport_recovery_sensitivity.csv")
    lagged = load("lagged_airport_disruption_association.csv")
    airport = load("airport_route_disruption_summary.csv")

    years = [2019, 2020, 2021, 2022, 2023]
    if (
        network["utc_year"].tolist() != years
        or regime["utc_year"].tolist() != years
    ):
        raise ValueError("Expected ordered UTC network years 2019–2023")

    if not network["scheduled_flight_count"].equals(
        regime["scheduled_flights"]
    ):
        raise ValueError("Network and regime flight counts disagree")

    primary_rows = recovery.loc[
        recovery["scenario"].eq("primary_candidate")
    ]
    if len(primary_rows) != 1:
        raise ValueError("Expected one primary recovery scenario")
    primary = primary_rows.iloc[0]

    if (
        primary["episodes"]
        != primary["observed_recoveries"]
        + primary["censored_episodes"]
    ):
        raise ValueError("Recovery episode counts disagree")

    if set(lagged["outcome"]) != {"severe", "cancellation"}:
        raise ValueError("Missing a lagged association outcome")

    chart_rows = int(airport["meets_plot_minimum"].sum())
    reference = regime.set_index("utc_year").loc[2019]
    shock = regime.set_index("utc_year").loc[2020]
    partial = network.set_index("utc_year").loc[2023]

    lines = [
        "# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience",
        "",
        "## Purpose and scope",
        "",
        "This is an advanced retrospective portfolio study for Data Science, "
        "applied science, forecasting, operations research, decision science, "
        "transportation, and aviation analytics roles. It is not an airline "
        "operating platform, dispatch system, real-time disruption-control "
        "tool, or production service.",
        "",
        "The official external-source scope is the selected Kaggle Flight Delay "
        "and Cancellation Dataset (approximately 2019–2023) and NOAA Global "
        "Historical Climatology Network hourly weather data. The Phase 5 "
        "network analyses below use the sampled flight records; they do not "
        "establish a weather effect. Verify actual downloaded coverage, "
        "licensing/terms, schemas, counts, and transformations in the "
        "project's source documentation.",
        "",
        "## UTC-year network",
        "",
        "The yearly directed graphs count scheduled flights on origin-to-"
        "destination routes using validated scheduled-departure UTC years. "
        "The tables describe the analysed sample, not the entire airline "
        "industry. The 2023 observation window is partial.",
        "",
        "| UTC year | Scheduled flights | Airports | Directed routes | Density | Strong components |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in network.itertuples(index=False):
        lines.append(
            f"| {row.utc_year}"
            f"{' (partial)' if row.partial_year_coverage else ''} "
            f"| {row.scheduled_flight_count:,} "
            f"| {row.node_count:,} "
            f"| {row.edge_count:,} "
            f"| {row.density:.4f} "
            f"| {row.strongly_connected_components:,} |"
        )

    lines += [
        "",
        "The 2019 and 2020 full-year network counts are "
        f"{int(reference['scheduled_flights']):,} and "
        f"{int(shock['scheduled_flights']):,}: a "
        f"{abs(shock['flight_volume_change_vs_2019_pct']):.2f}% "
        "decline in this sample. The cancellation rate rose from "
        f"{percent(reference['cancellation_rate'])} to "
        f"{percent(shock['cancellation_rate'])}; the severe arrival-delay "
        "rate **among completed flights** changed from "
        f"{percent(reference['severe_delay_rate_completed'])} to "
        f"{percent(shock['severe_delay_rate_completed'])}. "
        "The lower completed-flight severe-delay rate in 2020 does not "
        "demonstrate better resilience: cancellations and flight composition "
        "also changed. These comparisons are descriptive, not causal.",
        "",
        f"The last observed 2023 scheduled departure is "
        f"`{partial['last_observed_departure_utc']}`. "
        "Do not compare its raw flight or route total with a full year.",
        "",
        "![Fixed-airport 2019 and 2020 network illustration](figures/airport_network.png)",
        "",
        "The illustration fixes 20 airports using pooled 2019–2020 traffic "
        "and displays up to 60 busy directed routes per year. It is not "
        "the complete graph.",
        "",
        "## Airport association and recovery",
        "",
        f"The centrality-versus-disruption figure includes {chart_rows:,} "
        "origin airport-years meeting its minimum completed-flight count. "
        "Same-year centrality and disruption rates are associations only; "
        "they do not identify disruption transmission or causal effects.",
        "",
        "![Airport centrality and disruption](figures/centrality_vs_disruption.png)",
        "",
        "Under the primary airport recovery definition—at least "
        f"{int(primary['minimum_completed_per_airport_day'])} completed "
        "flights on a qualifying airport-day, disruption rate at least "
        f"{percent(primary['disruption_rate_threshold'])}, normal rate at "
        f"most {percent(primary['normal_rate_threshold'])}, and "
        f"{int(primary['consecutive_normal_days_required'])} consecutive "
        "normal days—the analysis found "
        f"{int(primary['episodes']):,} episodes across "
        f"{int(primary['airports_with_qualifying_days']):,} airports: "
        f"{int(primary['observed_recoveries']):,} observed recoveries and "
        f"{int(primary['censored_episodes']):,} censored episodes. "
        "The reported "
        f"{primary['median_days_among_observed_recoveries_only']:g}-day "
        "median is conditional on observed recovery; it is **not** a "
        "Kaplan–Meier median. Definitions and sensitivity scenarios are "
        "recorded in `tables/airport_recovery_sensitivity.csv`.",
        "",
        "## Month-to-month association",
        "",
        "Flight outcomes were grouped by the manifest's prediction-timestamp "
        "UTC month, then paired with the preceding UTC month at the same "
        "origin airport. Each month in a retained pair has at least 50 "
        "scheduled and 50 completed flights. This is retrospective: the "
        "flight-level outcome timestamps needed to establish pre-flight "
        "availability were not checked.",
        "",
        "| Outcome | Airport-month pairs | Airports | Pooled Pearson r | Within-airport Pearson r |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in lagged.itertuples(index=False):
        lines.append(
            f"| {row.outcome} "
            f"| {row.airport_month_pairs:,} "
            f"| {row.airports:,} "
            f"| {row.pooled_pearson_correlation:.3f} "
            f"| {row.within_airport_pearson_correlation:.3f} |"
        )

    lines += [
        "",
        "Within-airport centring removes each airport's average level, "
        "but does not remove seasonality, network-wide shocks, changing "
        "flight mix, or other confounding. Consecutive pairs can share a "
        "month. These correlations neither establish propagation nor "
        "constitute validated predictive performance.",
        "",
        "## Interpretation boundaries",
        "",
        "- Never use the same-year disruption table or lagged *outcome* "
        "rates as pre-flight features without separately proving that "
        "every contributing outcome was available before the precise "
        "prediction timestamp.",
        "- Prior-calendar-month schedule and graph features in the "
        "Phase 4 schedule-network table are distinct from these "
        "retrospective outcome associations. Their production timing "
        "and feature provenance still require an explicit audit.",
        "- The route-regime summaries and UTC-year graph summaries use "
        "different year groupings. Do not combine their flight "
        "denominators or infer a discrepancy in source completeness "
        "from that difference alone.",
        "- Recovery results depend on the qualifying-flight minimum, "
        "disruption and normal thresholds, consecutive-day rule, "
        "and censoring. The observed-only median must not be reported "
        "as the population recovery median.",
        "",
        "Reconstruct from the documented immutable raw flight source, "
        "timestamp mappings, transformation scripts, tests, and "
        "configuration. Keep processed flight data, feature matrices, "
        "and model artifacts out of Git.",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    print("Phase 5 report: PASS")
    print(f"UTC years: {len(network)}")
    print(f"Centrality plot airport-years: {chart_rows}")
    print(f"Primary recovery episodes: {int(primary['episodes']):,}")
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
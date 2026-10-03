"""Place the network report and write the bounded association report."""

from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EVALUATION = REPORTS / "evaluation"
TABLES = REPORTS / "tables"

COMBINED = REPORTS / "phase5_network_resilience_report.md"
NETWORK = EVALUATION / "network_resilience_report.md"
ASSOCIATION = EVALUATION / "propagation_report.md"


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    if not COMBINED.is_file():
        raise FileNotFoundError(COMBINED)

    network_text = COMBINED.read_text(encoding="utf-8")
    network_text = network_text.replace(
        "(figures/", "(../figures/"
    )
    network_text += (
        "\nThe dedicated survival and airport/route association "
        "reports are `survival_recovery_report.md` and "
        "`propagation_report.md` in this directory.\n"
    )

    lagged = pd.read_csv(
        TABLES / "lagged_airport_disruption_association.csv"
    )
    airport = pd.read_csv(
        TABLES / "airport_route_disruption_summary.csv"
    )

    if set(lagged["outcome"]) != {
        "severe", "cancellation"
    }:
        raise ValueError("Expected severe and cancellation rows")

    eligible = airport.loc[
        airport["meets_plot_minimum"].eq(True)
    ]
    if eligible.empty:
        raise ValueError("No eligible airport-year plot rows")

    table_rows = []
    for row in lagged.itertuples(index=False):
        table_rows.append(
            f"| {row.outcome} "
            f"| {int(row.airport_month_pairs):,} "
            f"| {int(row.airports):,} "
            f"| {row.pooled_pearson_correlation:.3f} "
            f"| {row.within_airport_pearson_correlation:.3f} |"
        )

    association_text = "\n".join([
        "# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience",
        "",
        "## Airport and route association",
        "",
        "This is an advanced retrospective portfolio study for Data "
        "Science, applied science, forecasting, operations research, "
        "decision science, transportation, and aviation analytics roles. "
        "This report documents airport-level associations, **not** "
        "aircraft-tail propagation or causal transmission.",
        "",
        "## Same-year airport association",
        "",
        f"The centrality-versus-disruption analysis retains "
        f"{len(eligible):,} origin airport-years meeting its minimum "
        "completed-flight count. Graph measures and disruption outcomes "
        "in the same UTC year describe association; their ordering "
        "does not establish that centrality preceded an event.",
        "",
        "![Centrality versus disruption](../figures/centrality_vs_disruption.png)",
        "",
        "## Consecutive-month association",
        "",
        "The flight sample and temporal-split manifest were joined by "
        "`source_row_number`. Outcomes were grouped by the manifest's "
        "prediction-timestamp UTC month at each origin airport. "
        "Each retained preceding and current month has at least 50 "
        "scheduled and 50 completed flights. The preceding month "
        "is non-overlapping with the current month.",
        "",
        "| Outcome | Airport-month pairs | Airports | Pooled Pearson r | Within-airport Pearson r |",
        "|---|---:|---:|---:|---:|",
        *table_rows,
        "",
        f"The retained current-month range runs from "
        f"{lagged['first_current_month_utc'].min()} through "
        f"{lagged['last_current_month_utc'].max()}. "
        "Within-airport correlations centre each airport's rates "
        "on its own mean. They do not control for seasonal or "
        "network-wide shocks, changes in flight mix, or dependence "
        "between consecutive pairs.",
        "",
        "## What this does not establish",
        "",
        "- These are retrospective outcome associations, not "
        "pre-flight predictive features. Actual outcome availability "
        "before each prediction timestamp has not been demonstrated.",
        "- The analysis has not linked individual aircraft: the "
        "audited source lacks a usable tail identifier. It cannot "
        "estimate aircraft-leg cascades or tail-based propagation.",
        "- A next-window **route** disruption association and "
        "downstream-flight exposure have not been estimated here. "
        "Annual route counts alone cannot establish either result.",
        "- Partial 2023 coverage and changes across operating regimes "
        "limit comparisons. These results must not be used to tune "
        "models against the reserved 2023 final-test period.",
        "",
        "The title `propagation_report.md` is the planned output name; "
        "its contents deliberately limit claims to the airport "
        "association that was actually measured. This is not a "
        "dispatch, optimisation, or real-time control system.",
        "",
    ]) + "\n"

    write_atomic(NETWORK, network_text)
    write_atomic(ASSOCIATION, association_text)

    for path in (NETWORK, ASSOCIATION):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Empty output: {path}")

    print("Phase 5 evaluation reports: PASS")
    print(f"Network: {NETWORK}")
    print(f"Airport association: {ASSOCIATION}")
    print(
        "Route next-window association remains unmeasured; "
        "the report states this explicitly."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
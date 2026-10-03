"""Summarise aligned UTC-year network and retrospective disruption data.

Graph structure and flight outcomes use the same scheduled-departure
UTC-year cohort. This is descriptive analysis, not a pre-flight feature
table or a causal/aircraft-propagation analysis.
"""

from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "reports/tables/yearly_network_summary.csv"
METRICS = ROOT / "reports/tables/yearly_airport_network_metrics.csv"
ROUTES = ROOT / "reports/tables/route_disruption_by_utc_year.csv"
OUTPUT = ROOT / "reports/tables/network_exposure_summary.csv"
YEARS = {2019, 2020, 2021, 2022, 2023}


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Replace the draft output only after validation and a full write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    for path in (GRAPH, METRICS, ROUTES):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    graph = pd.read_csv(GRAPH)
    metrics = pd.read_csv(METRICS)
    routes = pd.read_csv(ROUTES)

    for name, frame in (
        ("graph", graph),
        ("metrics", metrics),
        ("routes", routes),
    ):
        if set(frame["utc_year"]) != YEARS:
            raise ValueError(f"{name}: unexpected UTC-year coverage")

    if graph.duplicated("utc_year").any():
        raise ValueError("Duplicate graph year")
    if metrics.duplicated(["utc_year", "airport"]).any():
        raise ValueError("Duplicate airport-year metric")
    if routes.duplicated([
        "utc_year", "origin_airport", "destination_airport"
    ]).any():
        raise ValueError("Duplicate UTC-year route")

    counts = [
        "scheduled_flights",
        "cancelled_flights",
        "diverted_flights",
        "completed_arrivals",
        "severe_delay_flights",
    ]
    yearly = routes.groupby("utc_year")[counts].sum()
    yearly["route_count"] = routes.groupby("utc_year").size()
    yearly = yearly.reset_index()

    # Each route flight appears once as an origin appearance and once
    # as a destination appearance. These are NOT unique-flight totals.
    origin = routes[[
        "utc_year", "origin_airport", "scheduled_flights"
    ]].rename(columns={"origin_airport": "airport"})
    destination = routes[[
        "utc_year", "destination_airport", "scheduled_flights"
    ]].rename(columns={"destination_airport": "airport"})
    appearances = pd.concat(
        [origin, destination], ignore_index=True
    ).groupby(
        ["utc_year", "airport"], as_index=False
    )["scheduled_flights"].sum()

    airport_stats = appearances.groupby("utc_year").agg(
        airport_count=("airport", "nunique"),
        airport_appearances=("scheduled_flights", "sum"),
    ).reset_index()

    top_ten = appearances.sort_values(
        ["utc_year", "scheduled_flights"],
        ascending=[True, False],
    ).groupby("utc_year").head(10)
    top_ten = top_ten.groupby("utc_year")[
        "scheduled_flights"
    ].sum().rename("top_10_airport_appearances").reset_index()
    airport_stats = airport_stats.merge(
        top_ten, on="utc_year", validate="one_to_one"
    )
    airport_stats["top_10_airport_exposure_share"] = (
        airport_stats["top_10_airport_appearances"]
        / airport_stats["airport_appearances"]
    )

    metric_stats = metrics.groupby("utc_year").agg(
        metric_airports=("airport", "size"),
        mean_betweenness=("betweenness_centrality", "mean"),
        max_betweenness=("betweenness_centrality", "max"),
        mean_outgoing_route_hhi=(
            "outgoing_route_concentration_hhi", "mean"
        ),
        max_pagerank=("pagerank", "max"),
    ).reset_index()

    result = graph.merge(
        yearly, on="utc_year", validate="one_to_one"
    ).merge(
        airport_stats, on="utc_year", validate="one_to_one"
    ).merge(
        metric_stats, on="utc_year", validate="one_to_one"
    ).sort_values("utc_year").reset_index(drop=True)

    if not result["scheduled_flight_count"].eq(
        result["scheduled_flights"]
    ).all():
        raise ValueError("Graph and outcome flight totals differ")
    if not result["edge_count"].eq(
        result["route_count"]
    ).all():
        raise ValueError("Graph and outcome route counts differ")
    if not result["node_count"].eq(
        result["airport_count"]
    ).all():
        raise ValueError("Graph and airport counts differ")
    if not result["node_count"].eq(
        result["metric_airports"]
    ).all():
        raise ValueError("Graph and centrality airport counts differ")
    if not result["airport_appearances"].eq(
        2 * result["scheduled_flights"]
    ).all():
        raise ValueError("Origin/destination appearances do not reconcile")

    result["cancellation_rate"] = (
        result["cancelled_flights"] / result["scheduled_flights"]
    )
    result["diversion_rate"] = (
        result["diverted_flights"] / result["scheduled_flights"]
    )
    result["severe_delay_rate_completed"] = (
        result["severe_delay_flights"]
        / result["completed_arrivals"]
    )
    result["analysis_scope"] = (
        "retrospective_utc_year_network_association"
    )

    if len(result) != 5 or (
        result["scheduled_flights"].sum() != 2_999_998
    ):
        raise ValueError("Yearly cohort reconciliation failed")
    if result[[
        "cancellation_rate",
        "diversion_rate",
        "severe_delay_rate_completed",
        "top_10_airport_exposure_share",
    ]].isna().any().any():
        raise ValueError("Undefined yearly rate or exposure share")

    write_atomically(result, OUTPUT)
    print("Aligned UTC-year network exposure: PASS")
    print(f"Rows: {len(result)}; columns: {len(result.columns)}")
    print(result[[
        "utc_year", "scheduled_flights", "edge_count",
        "airport_count", "cancellation_rate",
        "severe_delay_rate_completed", "partial_year_coverage",
    ]].to_string(index=False))
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
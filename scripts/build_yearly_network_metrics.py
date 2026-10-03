"""Build centrality and route-concentration metrics for yearly graphs.

This script reads the validated yearly route-count table and reconstructs
one directed graph per UTC year. It produces descriptive airport-level
metrics only. It does not use labels, outcomes, weather, or future graph
states as prediction features.
"""

from collections import defaultdict
from pathlib import Path
import os
import sys

import pandas as pd

from airline_disruption.network.centrality import (
    calculate_node_centrality,
    calculate_route_concentration,
)
from airline_disruption.network.graph_builder import (
    build_directed_route_graph,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data/processed/network_windows/yearly_route_counts.csv"
)
OUTPUT = (
    ROOT
    / "reports/tables/yearly_airport_network_metrics.csv"
)

YEARS = (2019, 2020, 2021, 2022, 2023)

REQUIRED_COLUMNS = {
    "utc_year",
    "window_start_utc",
    "window_end_utc_exclusive",
    "origin_airport",
    "destination_airport",
    "scheduled_flight_count",
}


def read_route_table() -> pd.DataFrame:
    """Read and validate the yearly route-count table."""
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing route table: {INPUT}")

    routes = pd.read_csv(INPUT)

    missing = REQUIRED_COLUMNS - set(routes.columns)
    if missing:
        raise ValueError(f"Missing route columns: {sorted(missing)}")

    if routes.empty:
        raise ValueError("The yearly route table is empty.")

    if not routes["utc_year"].isin(YEARS).all():
        raise ValueError("Unexpected UTC year in route table.")

    counts = pd.to_numeric(
        routes["scheduled_flight_count"],
        errors="coerce",
    )
    if counts.isna().any() or (counts <= 0).any():
        raise ValueError(
            "Route counts must be positive numeric values."
        )

    if routes[
        "origin_airport"
    ].eq(routes["destination_airport"]).any():
        raise ValueError("Self-routes are not allowed.")

    duplicate_keys = routes.duplicated(
        [
            "utc_year",
            "origin_airport",
            "destination_airport",
        ]
    )
    if duplicate_keys.any():
        raise ValueError("Duplicate year-route keys found.")

    routes["scheduled_flight_count"] = counts.astype(int)
    return routes


def build_yearly_metrics(
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate airport metrics for each yearly graph."""
    rows = []

    for year in YEARS:
        subset = routes.loc[routes["utc_year"].eq(year)]

        route_counts = {
            (str(row.origin_airport), str(row.destination_airport)):
            int(row.scheduled_flight_count)
            for row in subset.itertuples(index=False)
        }

        graph = build_directed_route_graph(route_counts)
        centrality = calculate_node_centrality(graph)
        concentration = calculate_route_concentration(graph)

        metrics = centrality.merge(
            concentration,
            on="airport",
            how="outer",
            validate="one_to_one",
        )

        window_start = subset[
            "window_start_utc"
        ].drop_duplicates().tolist()
        window_end = subset[
            "window_end_utc_exclusive"
        ].drop_duplicates().tolist()

        if len(window_start) != 1 or len(window_end) != 1:
            raise ValueError(
                f"Inconsistent window metadata for {year}."
            )

        metrics.insert(0, "utc_year", year)
        metrics.insert(1, "window_start_utc", window_start[0])
        metrics.insert(2, "window_end_utc_exclusive", window_end[0])

        rows.append(metrics)

        print(
            f"Year {year}: "
            f"{len(graph.nodes):,} airports, "
            f"{len(graph.edges):,} routes"
        )

    result = pd.concat(rows, ignore_index=True)

    if result.duplicated(["utc_year", "airport"]).any():
        raise ValueError("Duplicate year-airport metrics found.")

    return result


def write_atomically(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Write a CSV through a temporary file and replace atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    routes = read_route_table()
    metrics = build_yearly_metrics(routes)
    write_atomically(metrics, OUTPUT)

    print("\nYearly network metrics created.")
    print(f"Rows: {len(metrics):,}")
    print(f"Columns: {len(metrics.columns):,}")
    print(f"Output: {OUTPUT}")
    print(
        metrics.groupby("utc_year", as_index=False)
        .size()
        .to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
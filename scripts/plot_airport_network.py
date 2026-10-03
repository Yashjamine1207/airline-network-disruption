"""Plot a comparable subset of the 2019 and 2020 route networks.

Select airports using combined 2019-2020 scheduled traffic, then show
the 60 busiest directed routes between them in each year. This is a
descriptive illustration, not the complete graph or a causal analysis.
"""

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "data/processed/network_windows/"
    "yearly_route_counts.csv"
)
OUTPUT = ROOT / "reports/figures/airport_network.png"

AIRPORT_LIMIT = 20
ROUTE_LIMIT_PER_YEAR = 60
YEARS = (2019, 2020)


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing route table: {INPUT}")

    routes = pd.read_csv(INPUT)
    selected_years = routes.loc[
        routes["utc_year"].isin(YEARS)
    ].copy()

    if set(selected_years["utc_year"]) != set(YEARS):
        raise ValueError("Both 2019 and 2020 are required")

    # Count each scheduled flight at both route endpoints solely to
    # choose one fixed airport set for the two visual panels.
    origins = selected_years[[
        "origin_airport", "scheduled_flight_count"
    ]].rename(columns={"origin_airport": "airport"})
    destinations = selected_years[[
        "destination_airport", "scheduled_flight_count"
    ]].rename(columns={"destination_airport": "airport"})
    traffic = (
        pd.concat([origins, destinations], ignore_index=True)
        .groupby("airport")["scheduled_flight_count"]
        .sum()
        .sort_values(ascending=False)
    )

    airports = list(traffic.head(AIRPORT_LIMIT).index)
    airport_set = set(airports)
    if len(airports) != AIRPORT_LIMIT:
        raise ValueError("Insufficient airports for the figure")

    panel_edges = {}
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(airports)

    for year in YEARS:
        eligible = selected_years.loc[
            selected_years["utc_year"].eq(year)
            & selected_years["origin_airport"].isin(airport_set)
            & selected_years["destination_airport"].isin(airport_set)
        ]
        busiest = eligible.nlargest(
            ROUTE_LIMIT_PER_YEAR,
            "scheduled_flight_count",
        )
        panel_edges[year] = busiest

        for row in busiest.itertuples(index=False):
            layout_graph.add_edge(
                row.origin_airport,
                row.destination_airport,
            )

    positions = nx.spring_layout(
        layout_graph, seed=42, iterations=100
    )
    largest_traffic = float(traffic.loc[airports].max())
    node_sizes = [
        250 + 950 * float(traffic.loc[airport]) / largest_traffic
        for airport in airports
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, year in zip(axes, YEARS):
        graph = nx.DiGraph()
        graph.add_nodes_from(airports)

        for row in panel_edges[year].itertuples(index=False):
            graph.add_edge(
                row.origin_airport,
                row.destination_airport,
                weight=int(row.scheduled_flight_count),
            )

        edge_list = list(graph.edges())
        maximum_weight = max(
            (graph[u][v]["weight"] for u, v in edge_list),
            default=1,
        )
        widths = [
            0.3 + 2.4 * graph[u][v]["weight"] / maximum_weight
            for u, v in edge_list
        ]

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=airports,
            node_size=node_sizes,
            node_color="#2563eb" if year == 2019 else "#dc2626",
            alpha=0.85,
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph, positions, font_size=8, ax=ax
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edge_list,
            width=widths,
            edge_color="#64748b",
            alpha=0.35,
            arrows=True,
            arrowsize=9,
            node_size=node_sizes,
            ax=ax,
        )

        ax.set_title(
            f"{year}: {graph.number_of_edges()} selected routes"
        )
        ax.axis("off")

    fig.suptitle(
        "Scheduled airport-route networks: fixed 20-airport subset\n"
        "Up to 60 busiest directed routes per year; "
        "node size uses pooled 2019-2020 traffic",
        fontsize=12,
    )
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    try:
        fig.savefig(temporary, format="png", dpi=160)
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)

    print("Airport network figure: PASS")
    print(f"Fixed airports: {len(airports)}")
    for year in YEARS:
        print(
            f"{year} routes displayed: "
            f"{len(panel_edges[year])}"
        )
    print(f"Output: {OUTPUT}")
    print(
        "Figure shows a selected subset, not the full "
        "airport network."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
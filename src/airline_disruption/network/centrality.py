"""Centrality and route-concentration measures for directed route graphs.

These functions are intended for retrospective descriptive network analysis.
They do not create point-in-time predictive features and do not establish
causal relationships between network structure and disruption.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import pandas as pd


def calculate_node_centrality(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """Calculate centrality measures for every airport in a graph.

    Args:
        graph: Directed airport-route graph. Each edge must contain a
            positive ``scheduled_flight_count`` attribute.

    Returns:
        One row per airport with degree, weighted degree, betweenness,
        closeness, and PageRank measures.
    """
    if type(graph) is not nx.DiGraph:
        raise TypeError("graph must be an exact networkx.DiGraph")

    for origin, destination, data in graph.edges(data=True):
        count = data.get("scheduled_flight_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(
                f"Invalid scheduled flight count on "
                f"{origin}->{destination}"
            )
        if count <= 0:
            raise ValueError(
                f"Scheduled flight count must be positive on "
                f"{origin}->{destination}"
            )

    nodes = sorted(graph.nodes)
    if not nodes:
        return pd.DataFrame(
            columns=[
                "airport",
                "in_degree",
                "out_degree",
                "weighted_in_degree",
                "weighted_out_degree",
                "betweenness_centrality",
                "closeness_centrality",
                "pagerank",
            ]
        )

    weighted_in = dict(
        graph.in_degree(weight="scheduled_flight_count")
    )
    weighted_out = dict(
        graph.out_degree(weight="scheduled_flight_count")
    )
    betweenness = nx.betweenness_centrality(
        graph,
        weight=None,
        normalized=True,
    )
    closeness = nx.closeness_centrality(graph)
    pagerank = nx.pagerank(
        graph,
        weight="scheduled_flight_count",
    )

    rows = []
    for airport in nodes:
        rows.append({
            "airport": airport,
            "in_degree": int(graph.in_degree(airport)),
            "out_degree": int(graph.out_degree(airport)),
            "weighted_in_degree": int(weighted_in[airport]),
            "weighted_out_degree": int(weighted_out[airport]),
            "betweenness_centrality": float(
                betweenness[airport]
            ),
            "closeness_centrality": float(
                closeness[airport]
            ),
            "pagerank": float(pagerank[airport]),
        })

    return pd.DataFrame(rows)


def calculate_route_concentration(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """Calculate airport-level route concentration.

    Concentration is measured with a Herfindahl-style index:

        HHI = sum(route_share ** 2)

    It is calculated separately for outgoing and incoming scheduled
    flight counts. A value near 1 means traffic is concentrated on a
    small number of routes; lower values indicate greater route
    diversity.

    Returns:
        One row per airport with incoming and outgoing route counts,
        weighted traffic totals, diversity counts, and concentration.
    """
    if type(graph) is not nx.DiGraph:
        raise TypeError("graph must be an exact networkx.DiGraph")

    for origin, destination, data in graph.edges(data=True):
        count = data.get("scheduled_flight_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(
                f"Invalid scheduled flight count on "
                f"{origin}->{destination}"
            )
        if count <= 0:
            raise ValueError(
                f"Scheduled flight count must be positive on "
                f"{origin}->{destination}"
            )

    rows = []

    for airport in sorted(graph.nodes):
        outgoing = [
            int(data["scheduled_flight_count"])
            for _, _, data in graph.out_edges(
                airport,
                data=True,
            )
        ]
        incoming = [
            int(data["scheduled_flight_count"])
            for _, _, data in graph.in_edges(
                airport,
                data=True,
            )
        ]

        def hhi(values: Iterable[int]) -> float:
            values = list(values)
            total = sum(values)
            if total == 0:
                return 0.0
            return float(
                sum((value / total) ** 2 for value in values)
            )

        outgoing_total = sum(outgoing)
        incoming_total = sum(incoming)

        rows.append({
            "airport": airport,
            "outgoing_route_count": len(outgoing),
            "incoming_route_count": len(incoming),
            "outgoing_scheduled_flight_count": outgoing_total,
            "incoming_scheduled_flight_count": incoming_total,
            "outgoing_route_diversity": len(outgoing),
            "incoming_route_diversity": len(incoming),
            "outgoing_route_concentration_hhi": hhi(outgoing),
            "incoming_route_concentration_hhi": hhi(incoming),
        })

    return pd.DataFrame(rows)
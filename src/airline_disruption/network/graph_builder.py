"""Build directed airport-route graphs from aggregated scheduled-flight counts.

This module builds descriptive graphs. It does not construct pre-flight
predictors or decide which flights belong in a time window. Callers must
filter flights to a documented UTC window before counting routes.
"""

from collections.abc import Mapping
from numbers import Integral

import networkx as nx


def build_directed_route_graph(
    route_counts: Mapping[tuple[str, str], int],
) -> nx.DiGraph:
    """Create a directed graph weighted by scheduled-flight count.

    Args:
        route_counts: Mapping from (origin airport, destination airport)
            to the number of eligible scheduled flights in one window.

    Returns:
        A directed graph. Each edge's ``weight`` and
        ``scheduled_flight_count`` equal its supplied flight count.

    Raises:
        ValueError: If an airport code, route, or count is invalid.
    """
    graph = nx.DiGraph()

    for route, count in route_counts.items():
        if not isinstance(route, tuple) or len(route) != 2:
            raise ValueError(
                f"Route must be an (origin, destination) tuple: {route!r}"
            )

        origin, destination = route
        if (
            not isinstance(origin, str)
            or not isinstance(destination, str)
            or not origin.strip()
            or not destination.strip()
        ):
            raise ValueError(f"Invalid airport identifier: {route!r}")

        if origin != origin.strip() or destination != destination.strip():
            raise ValueError(f"Airport identifier has whitespace: {route!r}")

        if origin == destination:
            raise ValueError(f"Origin and destination are equal: {route!r}")

        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError(f"Flight count must be an integer: {route!r}")

        if count <= 0:
            raise ValueError(f"Flight count must be positive: {route!r}")

        graph.add_edge(
            origin,
            destination,
            weight=int(count),
            scheduled_flight_count=int(count),
        )

    return graph


def summarise_route_graph(graph: nx.DiGraph) -> dict[str, int | float]:
    """Calculate basic structural measures for one descriptive graph.

    Connected-component counts are zero for an empty graph. Density is
    also set to zero when fewer than two airports are present.
    """
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Expected a networkx.DiGraph.")

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "scheduled_flight_count": sum(
            int(data["scheduled_flight_count"])
            for _, _, data in graph.edges(data=True)
        ),
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "weakly_connected_components": (
            nx.number_weakly_connected_components(graph)
            if node_count
            else 0
        ),
        "strongly_connected_components": (
            nx.number_strongly_connected_components(graph)
            if node_count
            else 0
        ),
    }
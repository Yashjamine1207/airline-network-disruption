"""Tests for descriptive, schedule-count airport-route graphs."""

import networkx as nx
import pytest

from airline_disruption.network.graph_builder import (
    build_directed_route_graph,
    summarise_route_graph,
)


def test_builds_directed_routes_with_scheduled_flight_weights():
    """Opposite directions remain separate edges with separate counts."""
    graph = build_directed_route_graph({
        ("JFK", "LAX"): 4,
        ("LAX", "JFK"): 2,
        ("JFK", "SFO"): 1,
    })

    assert isinstance(graph, nx.DiGraph)
    assert set(graph.nodes) == {"JFK", "LAX", "SFO"}
    assert graph.number_of_edges() == 3
    assert graph["JFK"]["LAX"]["weight"] == 4
    assert graph["LAX"]["JFK"]["weight"] == 2
    assert graph["JFK"]["SFO"]["scheduled_flight_count"] == 1
    assert not graph.has_edge("SFO", "JFK")


def test_summary_reports_structure_and_total_flights():
    """The edge count is routes; the weight total is scheduled flights."""
    graph = build_directed_route_graph({
        ("JFK", "LAX"): 4,
        ("LAX", "JFK"): 2,
        ("JFK", "SFO"): 1,
    })

    summary = summarise_route_graph(graph)

    assert summary["node_count"] == 3
    assert summary["edge_count"] == 3
    assert summary["scheduled_flight_count"] == 7
    assert summary["density"] == pytest.approx(0.5)
    assert summary["weakly_connected_components"] == 1
    assert summary["strongly_connected_components"] == 2


def test_empty_graph_has_zero_summary_values():
    """An empty window must not cause a density or component error."""
    summary = summarise_route_graph(
        build_directed_route_graph({})
    )

    assert summary == {
        "node_count": 0,
        "edge_count": 0,
        "scheduled_flight_count": 0,
        "density": 0.0,
        "weakly_connected_components": 0,
        "strongly_connected_components": 0,
    }


@pytest.mark.parametrize(
    ("route", "count"),
    [
        (("JFK", "JFK"), 1),
        (("", "LAX"), 1),
        (("JFK ", "LAX"), 1),
        (("JFK", "LAX"), 0),
        (("JFK", "LAX"), -2),
        (("JFK", "LAX"), 1.5),
        (("JFK", "LAX"), True),
        (("JFK",), 1),
    ],
)
def test_rejects_invalid_routes_and_counts(route, count):
    """Bad records must be identified, not silently added to the graph."""
    with pytest.raises(ValueError):
        build_directed_route_graph({route: count})


def test_rejects_graph_of_wrong_type():
    """The summary requires a directed graph."""
    with pytest.raises(TypeError):
        summarise_route_graph(nx.Graph())
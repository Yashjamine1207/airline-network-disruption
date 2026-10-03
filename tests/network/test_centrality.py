"""Tests for descriptive network centrality and concentration measures."""

import networkx as nx
import pandas as pd
import pytest

from airline_disruption.network.centrality import (
    calculate_node_centrality,
    calculate_route_concentration,
)


def build_test_graph() -> nx.DiGraph:
    """Create a small directed weighted airport graph."""
    graph = nx.DiGraph()
    graph.add_edge(
        "JFK",
        "LAX",
        scheduled_flight_count=4,
        weight=4,
    )
    graph.add_edge(
        "JFK",
        "SFO",
        scheduled_flight_count=1,
        weight=1,
    )
    graph.add_edge(
        "LAX",
        "JFK",
        scheduled_flight_count=2,
        weight=2,
    )
    return graph


def test_node_centrality_returns_one_row_per_airport():
    """Centrality output includes all graph airports."""
    result = calculate_node_centrality(build_test_graph())

    assert isinstance(result, pd.DataFrame)
    assert set(result["airport"]) == {"JFK", "LAX", "SFO"}
    assert len(result) == 3


def test_degree_and_weighted_degree_are_correct():
    """Unweighted degree counts routes; weighted degree counts flights."""
    result = calculate_node_centrality(build_test_graph())
    result = result.set_index("airport")

    assert result.loc["JFK", "out_degree"] == 2
    assert result.loc["JFK", "weighted_out_degree"] == 5
    assert result.loc["JFK", "in_degree"] == 1
    assert result.loc["JFK", "weighted_in_degree"] == 2

    assert result.loc["LAX", "out_degree"] == 1
    assert result.loc["LAX", "weighted_out_degree"] == 2
    assert result.loc["LAX", "in_degree"] == 1
    assert result.loc["LAX", "weighted_in_degree"] == 4


def test_pagerank_and_centrality_values_are_finite():
    """Calculated centrality values must be finite numbers."""
    result = calculate_node_centrality(build_test_graph())

    metric_columns = [
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
    ]

    assert result[metric_columns].notna().all().all()
    assert result[metric_columns].applymap(
        lambda value: value >= 0
    ).all().all()
    assert result["pagerank"].sum() == pytest.approx(1.0)


def test_route_concentration_uses_weighted_outgoing_shares():
    """JFK's HHI is (4/5)^2 + (1/5)^2."""
    result = calculate_route_concentration(build_test_graph())
    result = result.set_index("airport")

    assert result.loc["JFK", "outgoing_route_count"] == 2
    assert result.loc[
        "JFK",
        "outgoing_scheduled_flight_count",
    ] == 5
    assert result.loc[
        "JFK",
        "outgoing_route_diversity",
    ] == 2
    assert result.loc[
        "JFK",
        "outgoing_route_concentration_hhi",
    ] == pytest.approx(0.68)


def test_route_concentration_handles_zero_degree_airport():
    """An airport with no incoming or outgoing routes has zero HHI."""
    graph = nx.DiGraph()
    graph.add_node("ISOLATED")

    result = calculate_route_concentration(graph)
    row = result.iloc[0]

    assert row["airport"] == "ISOLATED"
    assert row["outgoing_route_count"] == 0
    assert row["incoming_route_count"] == 0
    assert row["outgoing_route_concentration_hhi"] == 0.0
    assert row["incoming_route_concentration_hhi"] == 0.0


@pytest.mark.parametrize(
    "graph",
    [
        nx.Graph(),
        nx.MultiDiGraph(),
    ],
)
def test_rejects_non_directed_simple_graphs(graph):
    """The functions require a simple directed graph."""
    with pytest.raises(TypeError):
        calculate_node_centrality(graph)

    with pytest.raises(TypeError):
        calculate_route_concentration(graph)


def test_rejects_invalid_edge_weight():
    """Every edge must have a positive integer flight count."""
    graph = nx.DiGraph()
    graph.add_edge("JFK", "LAX", scheduled_flight_count=0)

    with pytest.raises(ValueError):
        calculate_node_centrality(graph)

    with pytest.raises(ValueError):
        calculate_route_concentration(graph)


def test_empty_directed_graph_returns_expected_schema():
    """An empty directed graph returns an empty centrality table."""
    result = calculate_node_centrality(nx.DiGraph())

    assert result.empty
    assert list(result.columns) == [
        "airport",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
    ]
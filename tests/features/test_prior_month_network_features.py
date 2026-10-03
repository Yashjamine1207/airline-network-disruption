"""Test previous-month network features against future-flight leakage."""

import pandas as pd

import scripts.build_prior_month_network_features as builder


def sample_flights(include_future=False):
    flights = [
        ("2019-01-05T12:00:00Z", "JFK", "LAX"),
        ("2019-01-10T12:00:00Z", "JFK", "LAX"),
        ("2019-01-12T12:00:00Z", "JFK", "SFO"),
        ("2019-01-20T12:00:00Z", "LAX", "JFK"),
        ("2019-02-02T12:00:00Z", "JFK", "LAX"),
        ("2019-02-15T12:00:00Z", "JFK", "BOS"),
        ("2019-03-10T12:00:00Z", "JFK", "BOS"),
    ]
    if include_future:
        flights.append(
            ("2019-04-10T12:00:00Z", "JFK", "SFO")
        )

    departures = pd.to_datetime(
        [flight[0] for flight in flights],
        utc=True,
    )
    predictions = departures - pd.Timedelta(hours=2)

    return pd.DataFrame({
        "source_row_number": range(len(flights)),
        "scheduled_departure_utc": departures.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "prediction_timestamp_utc": predictions.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "origin_airport": [flight[1] for flight in flights],
        "destination_airport": [flight[2] for flight in flights],
    })


def build_test_output(monkeypatch, tmp_path, data):
    output = tmp_path / "network.csv"

    monkeypatch.setattr(
        builder, "chunks", lambda: iter([data.copy()])
    )
    monkeypatch.setattr(builder, "EXPECTED_ROWS", len(data))
    monkeypatch.setattr(builder, "OUTPUT", output)
    monkeypatch.setattr(
        builder, "TEMP", tmp_path / "network.csv.tmp"
    )

    edges = builder.count_prior_route_activity()
    metrics = builder.build_monthly_metrics(edges)
    builder.write_features(metrics)

    return pd.read_csv(output)


def test_february_uses_january_graph_only(
    monkeypatch,
    tmp_path,
):
    result = build_test_output(
        monkeypatch, tmp_path, sample_flights()
    )

    # Row 4 is JFK -> LAX on February 2. January's graph has:
    # JFK -> LAX twice, JFK -> SFO once, LAX -> JFK once.
    february_flight = result.loc[4]

    assert february_flight[
        "prior_month_origin_out_degree"
    ] == 2
    assert february_flight[
        "prior_month_origin_weighted_out_degree"
    ] == 3
    assert february_flight[
        "prior_month_destination_in_degree"
    ] == 1
    assert february_flight[
        "prior_month_destination_weighted_in_degree"
    ] == 2

    # January predictions have no December graph in this fixture.
    assert result.loc[0, "prior_month_origin_out_degree"] == 0


def test_future_month_does_not_change_earlier_graph_features(
    monkeypatch,
    tmp_path,
):
    original = build_test_output(
        monkeypatch, tmp_path, sample_flights()
    )
    with_future = build_test_output(
        monkeypatch,
        tmp_path,
        sample_flights(include_future=True),
    )

    columns = [
        "prior_month_origin_out_degree",
        "prior_month_origin_weighted_out_degree",
        "prior_month_destination_in_degree",
        "prior_month_destination_weighted_in_degree",
    ]

    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        with_future.loc[:6, columns].reset_index(drop=True),
    )
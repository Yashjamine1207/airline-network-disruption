"""Test that prior-month schedule counts exclude current/future months."""

import pandas as pd

import scripts.build_prior_month_schedule_counts as builder


def sample_flights(include_future: bool = False) -> pd.DataFrame:
    departures = [
        "2019-01-15T12:00:00Z",
        "2019-02-02T12:00:00Z",
        "2019-02-15T12:00:00Z",
        "2019-03-02T12:00:00Z",
        "2019-03-20T12:00:00Z",
        "2019-04-10T12:00:00Z",
    ]
    if include_future:
        departures.append("2019-12-10T12:00:00Z")

    departure_times = pd.to_datetime(departures, utc=True)
    prediction_times = departure_times - pd.Timedelta(hours=2)

    return pd.DataFrame({
        "source_row_number": range(len(departures)),
        "scheduled_departure_utc": departure_times.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "prediction_timestamp_utc": prediction_times.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "carrier_identifier": ["DOT_TEST"] * len(departures),
        "origin_airport": ["JFK"] * len(departures),
        "route": ["JFK-LAX"] * len(departures),
    })


def run_with_synthetic_flights(
    monkeypatch,
    tmp_path,
    flights: pd.DataFrame,
) -> pd.DataFrame:
    output = tmp_path / "prior_month_counts.csv"

    monkeypatch.setattr(
        builder,
        "read_chunks",
        lambda: iter([flights.copy()]),
    )
    monkeypatch.setattr(builder, "EXPECTED_ROWS", len(flights))
    monkeypatch.setattr(builder, "OUTPUT", output)
    monkeypatch.setattr(
        builder,
        "TEMP",
        tmp_path / "prior_month_counts.csv.tmp",
    )

    monthly_counts = builder.build_monthly_counts()
    builder.write_features(monthly_counts)

    return pd.read_csv(output)


def test_only_previous_calendar_month_is_counted(
    monkeypatch,
    tmp_path,
):
    result = run_with_synthetic_flights(
        monkeypatch,
        tmp_path,
        sample_flights(),
    )

    # January has no December history. Both February flights see
    # January's one flight, not one another. March sees February's two.
    expected = [0, 1, 1, 2, 2, 2]

    for entity in ("carrier", "origin", "route"):
        column = (
            f"prior_calendar_month_{entity}_"
            "scheduled_flight_count"
        )
        assert result[column].tolist() == expected

    assert result["source_row_number"].tolist() == list(range(6))
    assert "ARR_DELAY" not in result.columns
    assert "severe_delay_120" not in result.columns


def test_adding_a_future_flight_cannot_change_earlier_features(
    monkeypatch,
    tmp_path,
):
    original = run_with_synthetic_flights(
        monkeypatch,
        tmp_path,
        sample_flights(),
    )

    with_future = run_with_synthetic_flights(
        monkeypatch,
        tmp_path,
        sample_flights(include_future=True),
    )

    feature_columns = [
        f"prior_calendar_month_{entity}_scheduled_flight_count"
        for entity in ("carrier", "origin", "route")
    ]

    pd.testing.assert_frame_equal(
        original[feature_columns].reset_index(drop=True),
        with_future.loc[:5, feature_columns].reset_index(drop=True),
    )
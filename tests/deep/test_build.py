"""Tests for ``build_deep_inputs``: fit-only learning, index checks, the final-test guard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from airline_disruption.deep import inputs as di
from airline_disruption.deep.build import build_deep_inputs
from airline_disruption.features.feature_sets import ID_COLUMN
from airline_disruption.sequences import airport_bins as ab
from airline_disruption.sequences import lookback as lb

START = pd.Timestamp("2019-01-01", tz="UTC")
AIRPORTS = ["AAA", "BBB", "CCC", "DDD"]
N = 3000


def make_world(n: int = N, seed: int = 0, mutate=None, late_only_destination: str | None = None) -> SimpleNamespace:
    """A small self-consistent world: flights, the airport table, the sequence index and the row tables.

    ``mutate(flights)`` may edit the flights before anything is built. ``late_only_destination``
    makes the last 20% of flights (the validation rows) fly to an airport nobody flies to earlier.
    """
    rng = np.random.default_rng(seed)
    departure = START + pd.to_timedelta(np.sort(rng.integers(0, 90 * 24 * 3600, n)), unit="s")
    origin = rng.choice(AIRPORTS, n)
    destination = rng.choice(AIRPORTS, n)
    if late_only_destination:
        destination[int(0.8 * n) :] = late_only_destination
    flights = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n, dtype="int64") + 100,
            "origin_airport": origin,
            "destination_airport": destination,
            "scheduled_departure_utc": departure,
            "scheduled_elapsed_time_minutes": rng.integers(45, 300, n).astype("float32"),
            "distance_miles": rng.integers(80, 2800, n).astype("float32"),
        }
    )
    if mutate is not None:
        mutate(flights)

    airports = ab.airport_codes(flights["origin_airport"], flights["destination_airport"])
    grid = ab.grid_for_flights(flights)
    table, _ = ab.build_airport_bins(flights, airports, grid)

    prediction = flights["scheduled_departure_utc"] - pd.Timedelta(hours=2)
    prediction_seconds = ab.to_epoch_seconds(prediction)
    index = pd.DataFrame(
        {
            ID_COLUMN: flights[ID_COLUMN],
            "origin_index": ab.codes_to_index(flights["origin_airport"], airports).astype("int16"),
            "end_bin": lb.end_bins(prediction_seconds, grid).astype("int32"),
        }
    )
    X = pd.DataFrame(
        {
            "origin_airport": pd.Categorical(flights["origin_airport"]),
            "destination_airport": pd.Categorical(flights["destination_airport"]),
            "route": pd.Categorical(flights["origin_airport"] + "-" + flights["destination_airport"]),
            "carrier_identifier": pd.Categorical(rng.choice(["X", "Y"], n)),
            "scheduled_departure_month": flights["scheduled_departure_utc"].dt.month.astype("float32"),
            "scheduled_departure_day_of_week": flights["scheduled_departure_utc"].dt.dayofweek.astype("float32"),
            "scheduled_departure_hour_local": flights["scheduled_departure_utc"].dt.hour.astype("float32"),
            "scheduled_departure_minute_local": flights["scheduled_departure_utc"].dt.minute.astype("float32"),
            "scheduled_arrival_hour_local": rng.integers(0, 24, n).astype("float32"),
            "scheduled_arrival_minute_local": rng.integers(0, 60, n).astype("float32"),
            "scheduled_elapsed_time_minutes": flights["scheduled_elapsed_time_minutes"],
            "distance_miles": flights["distance_miles"],
        }
    )
    meta = pd.DataFrame({ID_COLUMN: flights[ID_COLUMN], "prediction_timestamp_utc": prediction})
    roles = pd.Series(np.where(np.arange(n) < 0.6 * n, "fit", np.where(np.arange(n) < 0.8 * n, "early_stop", "validation")))
    return SimpleNamespace(
        X=X, y=(rng.random(n) < 0.05).astype("int8"), meta=meta, roles=roles, index=index,
        table=table, airports=airports, grid=grid, flights=flights,
    )


def build(w: SimpleNamespace, **kw):
    return build_deep_inputs(w.X, w.y, w.meta, w.roles, w.index, w.table, w.airports, w.grid, **kw)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def test_output_arrays_are_aligned_typed_and_in_row_order() -> None:
    w = make_world()
    inputs, report = build(w)
    n = len(w.X)
    assert inputs.codes.shape == (n, 4) and inputs.numeric.shape == (n, 10)
    assert inputs.codes.dtype == np.int32 and inputs.numeric.dtype == np.float32
    assert inputs.label.dtype == np.int8 and inputs.role.dtype == np.int8 and inputs.airport_index.dtype == np.int16
    assert np.array_equal(inputs.source_row_number, w.meta[ID_COLUMN].to_numpy())
    assert np.array_equal(inputs.label, w.y)
    assert np.all(np.diff(inputs.prediction_seconds) >= 0)
    assert inputs.rows("fit").tolist() == list(range(int(0.6 * n)))
    assert set(report["role"]) == {"fit", "early_stop", "validation"}


def test_airport_index_is_the_flights_own_origin() -> None:
    w = make_world()
    inputs, _ = build(w)
    assert (w.airports[inputs.airport_index] == w.flights["origin_airport"].to_numpy()).all()


def test_the_index_is_found_by_row_id_even_when_the_index_table_is_shuffled() -> None:
    w = make_world()
    reference, _ = build(w)
    w.index = w.index.sample(frac=1.0, random_state=5).reset_index(drop=True)
    shuffled, _ = build(w)
    assert np.array_equal(reference.airport_index, shuffled.airport_index)
    assert np.array_equal(reference.end_bin, shuffled.end_bin)


# ---------------------------------------------------------------------------
# Fit-only learning
# ---------------------------------------------------------------------------
def test_changing_non_fit_rows_does_not_change_anything_learned() -> None:
    w = make_world()
    reference, _ = build(w)
    fit_end = int(0.6 * len(w.X))
    w.X.loc[w.X.index[fit_end:], "distance_miles"] = 40_000.0
    w.X.loc[w.X.index[fit_end:], "scheduled_elapsed_time_minutes"] = 5_000.0
    changed, _ = build(w)
    assert reference.spec.to_json() == changed.spec.to_json()
    assert np.array_equal(reference.numeric[:fit_end], changed.numeric[:fit_end])
    assert changed.numeric[fit_end:, di.NUMERIC_NAMES.index("log_distance_z")].min() > 2  # the change is visible


def test_scaler_ignores_flights_that_happen_after_the_last_fit_prediction_time() -> None:
    """Add a burst of very busy late flights. The table changes late, so the fit-row scaler must not."""
    base = make_world()
    reference, _ = build(base, scaler_sample=10**9)
    last_fit_prediction = base.meta["prediction_timestamp_utc"].iloc[int(0.6 * len(base.X)) - 1]

    def add_late_traffic(flights: pd.DataFrame) -> None:
        late = flights["scheduled_departure_utc"] > last_fit_prediction + pd.Timedelta(hours=8)
        flights.loc[late, "origin_airport"] = "AAA"

    # Same random draws, so early flights are identical; only late flights change airport.
    busy = make_world(mutate=add_late_traffic)
    changed, _ = build(busy, scaler_sample=10**9)
    fit_rows = busy.roles == "fit"
    assert (busy.flights.loc[fit_rows, "origin_airport"] == base.flights.loc[fit_rows, "origin_airport"]).all()
    assert reference.scaler.to_json() == changed.scaler.to_json()


def test_scaler_sampling_is_reproducible_and_seeded() -> None:
    w = make_world()
    a, _ = build(w, scaler_sample=500, seed=1)
    b, _ = build(w, scaler_sample=500, seed=1)
    c, _ = build(w, scaler_sample=500, seed=2)
    assert a.scaler.to_json() == b.scaler.to_json()
    assert a.scaler.to_json() != c.scaler.to_json()


def test_scaled_fit_sequences_have_roughly_zero_mean_and_unit_spread() -> None:
    from airline_disruption.sequences.lookback import gather_sequences

    w = make_world()
    inputs, _ = build(w, scaler_sample=10**9)
    fit = inputs.rows("fit")
    values, mask = gather_sequences(w.table, inputs.airport_index[fit], inputs.end_bin[fit], w.grid)
    scaled = inputs.scaler.apply(values, mask)[mask]
    assert scaled.mean(axis=0) == pytest.approx(0.0, abs=1e-3)
    assert scaled.std(axis=0) == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_a_final_test_row_is_refused() -> None:
    w = make_world()
    w.roles.iloc[-1] = "final_test"
    with pytest.raises(ValueError, match="final test stays locked"):
        build(w)


def test_no_fit_rows_is_refused() -> None:
    w = make_world()
    w.roles[:] = "validation"
    with pytest.raises(ValueError, match="no fit rows"):
        build(w)


def test_mismatched_lengths_are_refused() -> None:
    w = make_world()
    with pytest.raises(ValueError, match="same number of rows"):
        build_deep_inputs(w.X.iloc[:-1], w.y, w.meta, w.roles, w.index, w.table, w.airports, w.grid)


def test_a_row_without_an_index_entry_is_refused() -> None:
    w = make_world()
    w.index = w.index.iloc[1:]
    with pytest.raises(ValueError, match="no entry in the sequence index"):
        build(w)


def test_a_duplicated_index_id_is_refused() -> None:
    w = make_world()
    w.index = pd.concat([w.index, w.index.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build(w)


def test_an_index_that_points_at_the_wrong_airport_is_refused() -> None:
    w = make_world()
    w.index.loc[5, "origin_index"] = (int(w.index.loc[5, "origin_index"]) + 1) % len(w.airports)
    with pytest.raises(ValueError, match="different origin airport"):
        build(w)


def test_a_stale_end_bin_is_refused() -> None:
    w = make_world()
    w.index.loc[9, "end_bin"] += 1  # one bin later than the prediction time allows
    with pytest.raises(ValueError, match="end_bin does not match"):
        build(w)


# ---------------------------------------------------------------------------
# Unseen levels
# ---------------------------------------------------------------------------
def test_the_report_shows_levels_that_first_appear_after_the_fit_period() -> None:
    w = make_world(late_only_destination="ZZZ")
    inputs, report = build(w)
    by_role = report.set_index("role")
    assert by_role.loc["fit", "unknown_share_destination_airport"] == 0.0
    assert by_role.loc["validation", "unknown_share_destination_airport"] == 1.0  # every validation flight goes to ZZZ
    assert by_role.loc["validation", "unknown_share_origin_airport"] == 0.0
    assert (inputs.codes[inputs.rows("validation"), 1] == di.UNKNOWN_CODE).all()

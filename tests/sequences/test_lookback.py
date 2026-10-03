"""Tests for lookback sequences: point-in-time safety, padding and masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.sequences import airport_bins as ab
from airline_disruption.sequences import lookback as lb

BASE = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp())
HOUR = 3600


def small_grid(n_bins: int = 40) -> ab.BinGrid:
    return ab.BinGrid(start_seconds=BASE, bin_hours=3, n_bins=n_bins)


def labelled_table(n_airports: int = 3, n_bins: int = 40) -> np.ndarray:
    """table[a, b, c] = 100*a + b + c/10, so every cell tells you where it came from."""
    a, b, c = np.meshgrid(np.arange(n_airports), np.arange(n_bins), np.arange(3), indexing="ij")
    return (100 * a + b + c / 10).astype("float32")


# ---------------------------------------------------------------------------
# End bin: the newest bin that has CLOSED by the prediction time
# ---------------------------------------------------------------------------
def test_end_bin_at_an_exact_edge_includes_the_bin_that_closes_at_that_instant() -> None:
    grid = small_grid()
    edge = BASE + 3 * HOUR  # bin 0 closes here
    assert lb.end_bins(np.array([edge]), grid)[0] == 1  # bin 0 is readable
    assert lb.end_bins(np.array([edge - 1]), grid)[0] == 0  # one second earlier it is not


def test_the_bin_containing_the_prediction_time_is_never_read() -> None:
    grid = small_grid()
    table = labelled_table()
    t = BASE + 10 * HOUR + 30 * 60  # 10:30 is inside bin 3
    end = lb.end_bins(np.array([t]), grid)
    values, mask = lb.gather_sequences(table, np.array([1]), end, grid, lookback=4, burn_in_hours=0)
    assert end[0] == 3
    assert values[0, -1, 0] == pytest.approx(100 + 2)  # newest position is bin 2, not bin 3
    # Oldest position would be bin -1, before the grid starts: masked even with no burn-in.
    assert mask.tolist() == [[False, True, True, True]]


def test_no_position_in_any_sequence_closes_after_the_prediction_time() -> None:
    grid = small_grid(n_bins=400)
    rng = np.random.default_rng(3)
    t = BASE + rng.integers(0, 390 * 3 * HOUR, size=5000)
    end = lb.end_bins(t, grid)
    lb.assert_no_bin_after_prediction(end, t, grid)  # the runtime guard agrees
    positions = end[:, None] + np.arange(-16, 0)[None, :]
    assert (grid.bin_end_seconds(positions) <= t[:, None]).all()


def test_runtime_guard_accepts_a_bin_that_closes_exactly_at_the_prediction_time() -> None:
    grid = small_grid()
    edge = np.array([BASE + 6 * HOUR])  # bins 0 and 1 close at or before this instant
    lb.assert_no_bin_after_prediction(lb.end_bins(edge, grid), edge, grid)


@pytest.mark.parametrize("minutes_into_bin", [1, 90, 179])
def test_runtime_guard_rejects_an_end_bin_that_is_one_too_late(minutes_into_bin: int) -> None:
    grid = small_grid()
    # Bin 3 spans 09:00-12:00. Reading bin 3 would use flights up to 3 hours after the bin opened.
    t = np.array([BASE + 9 * HOUR + minutes_into_bin * 60])
    end = lb.end_bins(t, grid)
    lb.assert_no_bin_after_prediction(end, t, grid)
    with pytest.raises(ValueError, match="closes after the prediction time"):
        lb.assert_no_bin_after_prediction(end + 1, t, grid)


# ---------------------------------------------------------------------------
# Gathering, padding and masks
# ---------------------------------------------------------------------------
def test_gather_returns_the_lookback_bins_oldest_first() -> None:
    grid, table = small_grid(), labelled_table()
    values, mask = lb.gather_sequences(table, np.array([2]), np.array([10]), grid, lookback=4, burn_in_hours=0)
    assert values.shape == (1, 4, 3) and values.dtype == np.float32
    assert values[0, :, 0].tolist() == [206.0, 207.0, 208.0, 209.0]  # bins 6..9 of airport 2, oldest first
    assert values[0, 0, 1] == pytest.approx(206.1)  # channel order kept
    assert mask.tolist() == [[True, True, True, True]]


def test_early_positions_are_zero_padded_and_masked() -> None:
    grid, table = small_grid(), labelled_table()
    # burn-in of 6 hours = 2 bins, so bins 0 and 1 are unavailable. end_bin 3, lookback 4 -> bins -1, 0, 1, 2.
    values, mask = lb.gather_sequences(table, np.array([1]), np.array([3]), grid, lookback=4, burn_in_hours=6)
    assert mask.tolist() == [[False, False, False, True]]
    assert values[0, :3].sum() == 0  # padded values are exactly zero
    assert values[0, 3, 0] == pytest.approx(102.0)


def test_burn_in_bins_that_exist_in_the_table_are_still_masked() -> None:
    grid = small_grid()
    table = labelled_table() + 1000  # every cell non-zero, so a leaked read would show
    values, mask = lb.gather_sequences(table, np.array([0]), np.array([5]), grid, lookback=4, burn_in_hours=12)
    # first valid bin = 4, so of bins 1..4 only bin 4 is real
    assert mask.tolist() == [[False, False, False, True]]
    assert values[0, :3].sum() == 0
    assert values[0, 3, 0] > 1000


def test_a_real_quiet_bin_is_not_confused_with_padding() -> None:
    grid = small_grid()
    table = labelled_table()
    table[1, 6, :] = 0.0  # a real bin with no scheduled flights
    values, mask = lb.gather_sequences(table, np.array([1]), np.array([7]), grid, lookback=3, burn_in_hours=0)
    assert values[0, 2].sum() == 0 and bool(mask[0, 2])  # newest position is bin 6: a real zero, mask True
    assert values[0, 1].sum() > 0


def test_padding_is_always_a_prefix() -> None:
    grid, table = small_grid(200), labelled_table(2, 200)
    end = np.random.default_rng(1).integers(0, 200, size=2000)
    _, mask = lb.gather_sequences(table, np.zeros(2000, int), end, grid, lookback=16)
    for row in mask:
        first_true = int(np.argmax(row)) if row.any() else len(row)
        assert not row[:first_true].any() and row[first_true:].all()


def test_count_valid_bins_matches_the_mask() -> None:
    grid, table = small_grid(200), labelled_table(2, 200)
    end = np.arange(0, 200)
    _, mask = lb.gather_sequences(table, np.zeros(len(end), int), end, grid, lookback=16)
    assert lb.count_valid_bins(end, grid).tolist() == mask.sum(axis=1).tolist()
    assert lb.count_valid_bins(np.array([0, 8, 12, 24, 100]), grid).tolist() == [0, 0, 4, 16, 16]


def test_vectorised_gather_matches_a_slow_loop() -> None:
    grid = small_grid(60)
    table = np.random.default_rng(7).integers(0, 30, size=(4, 60, 3)).astype("float32")
    rng = np.random.default_rng(8)
    airport = rng.integers(0, 4, 300)
    end = rng.integers(0, 61, 300)
    values, mask = lb.gather_sequences(table, airport, end, grid, lookback=6, burn_in_hours=9)
    first_valid = 3
    for i in range(len(end)):
        for j, b in enumerate(range(end[i] - 6, end[i])):
            if b >= first_valid:
                assert mask[i, j] and np.array_equal(values[i, j], table[airport[i], b])
            else:
                assert not mask[i, j] and not values[i, j].any()


def test_gather_rejects_bad_inputs() -> None:
    grid, table = small_grid(), labelled_table()
    with pytest.raises(ValueError, match="beyond the last bin"):
        lb.gather_sequences(table, np.array([0]), np.array([41]), grid, lookback=4)
    with pytest.raises(ValueError, match="same shape"):
        lb.gather_sequences(table, np.array([0, 1]), np.array([5]), grid, lookback=4)
    values, mask = lb.gather_sequences(table, np.array([0]), np.array([40]), grid, lookback=4)  # newest = last bin: allowed
    assert mask.all() and values[0, -1, 0] == pytest.approx(39.0)


def test_gather_does_not_modify_the_table() -> None:
    grid, table = small_grid(), labelled_table()
    before = table.copy()
    lb.gather_sequences(table, np.array([0, 1]), np.array([3, 30]), grid, lookback=8)
    assert np.array_equal(table, before)


# ---------------------------------------------------------------------------
# The key test: the future cannot change a sequence
# ---------------------------------------------------------------------------
def random_flights(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = np.array(["AAA", "BBB", "CCC", "DDD", "EEE"])
    origin = rng.choice(codes, n)
    destination = np.array([rng.choice(codes[codes != o]) for o in origin])
    departure = pd.to_datetime(BASE + rng.integers(0, 10 * 86400, n), unit="s", utc=True)
    return pd.DataFrame(
        {
            "origin_airport": origin,
            "destination_airport": destination,
            "scheduled_departure_utc": departure,
            "scheduled_elapsed_time_minutes": rng.integers(40, 400, n).astype(float),
        }
    )


def sequences_for(frame: pd.DataFrame, grid: ab.BinGrid, airports: np.ndarray, airport_code: str, t: int):
    table, _ = ab.build_airport_bins(frame, airports, grid)
    index = np.array([list(airports).index(airport_code)])
    end = lb.end_bins(np.array([t]), grid)
    return lb.gather_sequences(table, index, end, grid, lookback=8, burn_in_hours=0)


@pytest.mark.parametrize("cutoff_hours", [30.5, 77.0, 120.25, 200.0])
def test_removing_every_flight_that_departs_at_or_after_the_prediction_time_changes_nothing(cutoff_hours: float) -> None:
    frame = random_flights(1500, seed=11)
    airports = ab.airport_codes(frame["origin_airport"], frame["destination_airport"])
    grid = ab.grid_for_flights(frame)
    cutoff = BASE + int(cutoff_hours * HOUR)

    departure = ab.to_epoch_seconds(frame["scheduled_departure_utc"])
    truncated = frame[departure < cutoff]
    assert 0 < len(truncated) < len(frame)  # the test would be empty otherwise

    for code in airports:
        full_values, full_mask = sequences_for(frame, grid, airports, code, cutoff)
        cut_values, cut_mask = sequences_for(truncated, grid, airports, code, cutoff)
        assert np.array_equal(full_values, cut_values)
        assert np.array_equal(full_mask, cut_mask)


def test_the_same_holds_for_earlier_prediction_times_than_the_cutoff() -> None:
    frame = random_flights(1500, seed=12)
    airports = ab.airport_codes(frame["origin_airport"], frame["destination_airport"])
    grid = ab.grid_for_flights(frame)
    cutoff = BASE + 100 * HOUR
    truncated = frame[ab.to_epoch_seconds(frame["scheduled_departure_utc"]) < cutoff]
    for t in (BASE + 20 * HOUR, BASE + 55 * HOUR + 1234, cutoff - 1):
        a = sequences_for(frame, grid, airports, "BBB", t)
        b = sequences_for(truncated, grid, airports, "BBB", t)
        assert np.array_equal(a[0], b[0])


def test_the_test_can_fail_reading_the_open_bin_would_be_caught() -> None:
    """Guard against a vacuous test: reading the still-open bin DOES change with future flights."""
    frame = random_flights(1500, seed=13)
    airports = ab.airport_codes(frame["origin_airport"], frame["destination_airport"])
    grid = ab.grid_for_flights(frame)
    t = BASE + 100 * HOUR + 30 * 60  # half way through a bin
    truncated = frame[ab.to_epoch_seconds(frame["scheduled_departure_utc"]) < t]
    full_table, _ = ab.build_airport_bins(frame, airports, grid)
    cut_table, _ = ab.build_airport_bins(truncated, airports, grid)
    open_bin = int(lb.end_bins(np.array([t]), grid)[0])
    assert not np.array_equal(full_table[:, open_bin, :], cut_table[:, open_bin, :])


# ---------------------------------------------------------------------------
# Density helper
# ---------------------------------------------------------------------------
def test_prior_counts_match_brute_force_including_window_edges() -> None:
    rng = np.random.default_rng(5)
    key = rng.integers(0, 6, 400)
    seconds = rng.integers(10_000, 90_000, 400)
    window = 7200
    query_key, query_seconds = key.copy(), seconds.copy()
    # Force exact-edge cases: a departure exactly `window` before a query, and exactly at it.
    seconds[0], key[0], query_key[1], query_seconds[1] = 50_000 - window, 3, 3, 50_000
    seconds[2], key[2], query_key[1] = 50_000, 3, 3
    got = lb.prior_counts(key, seconds, query_key, query_seconds, window)
    for i in range(len(query_key)):
        expected = int(((key == query_key[i]) & (seconds >= query_seconds[i] - window) & (seconds < query_seconds[i])).sum())
        assert got[i] == expected
    assert got[1] >= 1  # the flight exactly `window` earlier counts; the one exactly at t does not

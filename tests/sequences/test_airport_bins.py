"""Tests for the airport-by-time-bin schedule table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.features.feature_sets import find_forbidden_columns
from airline_disruption.sequences import airport_bins as ab

DAY = pd.Timestamp("2019-01-01", tz="UTC")


def flights(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """rows: (origin, destination, departure 'YYYY-MM-DD HH:MM' in UTC, scheduled elapsed minutes)."""
    return pd.DataFrame(
        {
            "origin_airport": [r[0] for r in rows],
            "destination_airport": [r[1] for r in rows],
            "scheduled_departure_utc": pd.to_datetime([r[2] for r in rows], utc=True),
            "scheduled_elapsed_time_minutes": [r[3] for r in rows],
        }
    )


def build(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, ab.BinGrid, dict]:
    airports = ab.airport_codes(frame["origin_airport"], frame["destination_airport"])
    grid = ab.grid_for_flights(frame)
    table, stats = ab.build_airport_bins(frame, airports, grid)
    return table, airports, grid, stats


def cell(table, airports, code, bin_index, channel) -> float:
    return float(table[list(airports).index(code), bin_index, ab.CHANNELS.index(channel)])


# ---------------------------------------------------------------------------
# Grid and timestamps
# ---------------------------------------------------------------------------
def test_grid_starts_at_utc_midnight_and_covers_the_last_instant() -> None:
    start = int(DAY.timestamp()) + 5 * 3600 + 17 * 60  # 05:17 UTC
    end = int(DAY.timestamp()) + 2 * 86400 + 60  # 00:01 two days later
    grid = ab.make_grid(start, end, bin_hours=3)
    assert grid.start_seconds == int(DAY.timestamp())
    assert grid.n_bins == 17  # two full days (16 bins) plus the bin holding 00:01
    assert grid.bin_of(np.array([end]))[0] == grid.n_bins - 1


@pytest.mark.parametrize("bad_width", [5, 7, 9, 48])
def test_grid_rejects_a_width_that_does_not_divide_a_day(bad_width: int) -> None:
    with pytest.raises(ValueError, match="divide 24"):
        ab.make_grid(0, 10, bin_hours=bad_width)


def test_bin_edges_belong_to_the_later_bin() -> None:
    grid = ab.make_grid(int(DAY.timestamp()), int(DAY.timestamp()) + 86400, 3)
    base = int(DAY.timestamp())
    assert grid.bin_of(np.array([base + 3 * 3600 - 1, base + 3 * 3600])).tolist() == [0, 1]
    assert grid.bin_end_seconds(np.array([0]))[0] == base + 3 * 3600


def test_first_valid_bin_rounds_the_burn_in_up_to_whole_bins() -> None:
    grid = ab.make_grid(0, 86400, 3)
    assert grid.first_valid_bin(24) == 8
    assert grid.first_valid_bin(25) == 9
    assert grid.first_valid_bin(0) == 0


def test_epoch_seconds_converts_other_zones_to_utc() -> None:
    utc = pd.Series(pd.to_datetime(["2019-01-01 12:00"], utc=True))
    eastern = utc.dt.tz_convert("America/New_York")
    assert ab.to_epoch_seconds(eastern)[0] == ab.to_epoch_seconds(utc)[0] == int(pd.Timestamp("2019-01-01 12:00", tz="UTC").timestamp())


def test_epoch_seconds_rejects_naive_and_missing_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ab.to_epoch_seconds(pd.Series(pd.to_datetime(["2019-01-01 12:00"])))
    with pytest.raises(ValueError, match="Missing"):
        ab.to_epoch_seconds(pd.Series(pd.to_datetime(["2019-01-01 12:00", None], utc=True)))


# ---------------------------------------------------------------------------
# Counts, checked by hand
# ---------------------------------------------------------------------------
HAND = [
    ("AAA", "BBB", "2019-01-01 01:00", 90),   # departs bin 0, arrives 02:30 -> bin 0
    ("AAA", "CCC", "2019-01-01 01:30", 240),  # departs bin 0, arrives 05:30 -> bin 1
    ("AAA", "BBB", "2019-01-01 02:00", 60),   # departs bin 0 (same destination as row 1), arrives 03:00 -> bin 1
    ("AAA", "BBB", "2019-01-01 04:00", 60),   # departs bin 1, arrives 05:00 -> bin 1
    ("BBB", "AAA", "2019-01-01 23:00", 180),  # departs bin 7, arrives 02:00 NEXT UTC day -> bin 8
]


def test_counts_match_a_hand_calculation() -> None:
    table, airports, _, stats = build(flights(HAND))
    assert cell(table, airports, "AAA", 0, "scheduled_departures") == 3
    assert cell(table, airports, "AAA", 1, "scheduled_departures") == 1
    assert cell(table, airports, "BBB", 7, "scheduled_departures") == 1
    assert cell(table, airports, "BBB", 0, "scheduled_arrivals") == 1
    assert cell(table, airports, "BBB", 1, "scheduled_arrivals") == 2
    assert cell(table, airports, "CCC", 1, "scheduled_arrivals") == 1
    assert cell(table, airports, "AAA", 8, "scheduled_arrivals") == 1
    assert stats == {"rows": 5, "arrivals_counted": 5, "arrivals_not_placeable": 0}


def test_distinct_destinations_counts_each_destination_once_per_bin() -> None:
    table, airports, _, _ = build(flights(HAND))
    # Bin 0 at AAA: BBB twice and CCC once -> 2 distinct, although 3 flights.
    assert cell(table, airports, "AAA", 0, "distinct_destinations") == 2
    assert cell(table, airports, "AAA", 1, "distinct_destinations") == 1
    assert cell(table, airports, "BBB", 7, "distinct_destinations") == 1


def test_arrival_crossing_utc_midnight_lands_in_the_next_day() -> None:
    table, airports, grid, _ = build(flights(HAND))
    assert grid.n_bins == 9  # last arrival is in the first bin of the second day
    assert table[list(airports).index("AAA"), 8, 1] == 1


def test_every_departure_is_counted_exactly_once() -> None:
    frame = flights(HAND)
    table, _, _, stats = build(frame)
    assert table[:, :, 0].sum() == len(frame)
    assert table[:, :, 1].sum() == stats["arrivals_counted"]


def test_table_shape_dtype_and_channel_order() -> None:
    table, airports, grid, _ = build(flights(HAND))
    assert table.shape == (len(airports), grid.n_bins, 3)
    assert table.dtype == np.float32
    assert ab.CHANNELS == ("scheduled_departures", "scheduled_arrivals", "distinct_destinations")


@pytest.mark.parametrize("bad_elapsed", [np.nan, 0.0, -30.0, 24 * 60 + 1])
def test_unplaceable_arrival_still_counts_as_a_departure(bad_elapsed: float) -> None:
    frame = flights([("AAA", "BBB", "2019-01-01 01:00", bad_elapsed), ("AAA", "BBB", "2019-01-01 01:10", 60)])
    table, airports, _, stats = build(frame)
    assert cell(table, airports, "AAA", 0, "scheduled_departures") == 2
    assert table[:, :, 1].sum() == 1
    assert stats["arrivals_not_placeable"] == 1


def test_elapsed_time_is_rounded_to_the_nearest_second_not_truncated() -> None:
    # 179.99999 minutes is 180 minutes with float noise: the flight lands at 03:00, the start of bin 1.
    frame = flights([("AAA", "BBB", "2019-01-01 00:00", 180.0 - 1e-6)])
    table, airports, _, _ = build(frame)
    assert cell(table, airports, "BBB", 1, "scheduled_arrivals") == 1
    assert cell(table, airports, "BBB", 0, "scheduled_arrivals") == 0


def test_a_24_hour_elapsed_time_is_still_placeable() -> None:
    frame = flights([("AAA", "BBB", "2019-01-01 00:00", 24 * 60)])
    _, _, _, stats = build(frame)
    assert stats["arrivals_counted"] == 1


# ---------------------------------------------------------------------------
# Only the schedule goes in
# ---------------------------------------------------------------------------
def test_builder_reads_only_schedule_columns() -> None:
    assert find_forbidden_columns(ab.REQUIRED_COLUMNS) == []
    assert find_forbidden_columns(ab.CHANNELS) == []


def test_outcome_columns_in_the_input_are_ignored() -> None:
    plain = flights(HAND)
    with_outcomes = plain.assign(
        cancelled=[1, 0, 1, 0, 1],
        arrival_delay_minutes=[500, -3, 240, 0, 999],
        diverted=[0, 1, 0, 0, 0],
    )
    a, *_ = build(plain)
    b, *_ = build(with_outcomes)
    assert np.array_equal(a, b)  # cancelled and diverted flights count exactly like the others


def test_missing_required_column_is_reported() -> None:
    frame = flights(HAND).drop(columns=["scheduled_elapsed_time_minutes"])
    airports = ab.airport_codes(frame["origin_airport"], frame["destination_airport"])
    with pytest.raises(ValueError, match="Missing columns"):
        ab.build_airport_bins(frame, airports, ab.make_grid(0, 86400))


# ---------------------------------------------------------------------------
# Airport codes
# ---------------------------------------------------------------------------
def test_airport_list_is_sorted_and_unique_across_both_columns() -> None:
    airports = ab.airport_codes(pd.Series(["BBB", "AAA", "BBB"]), pd.Series(["CCC", "AAA", "AAA"]))
    assert airports.tolist() == ["AAA", "BBB", "CCC"]


def test_categorical_and_string_columns_give_the_same_indices() -> None:
    airports = np.array(["AAA", "BBB", "CCC"])
    strings = pd.Series(["CCC", "AAA", "BBB", "AAA"])
    categorical = pd.Series(pd.Categorical(strings, categories=["ZZZ", "CCC", "BBB", "AAA"]))  # unused level, other order
    assert ab.codes_to_index(strings, airports).tolist() == ab.codes_to_index(categorical, airports).tolist() == [2, 0, 1, 0]


def test_unknown_or_missing_airport_is_rejected() -> None:
    airports = np.array(["AAA", "BBB"])
    with pytest.raises(ValueError, match="not in the airport list"):
        ab.codes_to_index(pd.Series(["AAA", "XXX"]), airports)
    with pytest.raises(ValueError, match="Missing"):
        ab.codes_to_index(pd.Series(pd.Categorical(["AAA", None], categories=["AAA"])), airports)
    with pytest.raises(ValueError, match="Missing"):
        ab.codes_to_index(pd.Series(["AAA", None]), airports)


# ---------------------------------------------------------------------------
# Grid from flights, save and load
# ---------------------------------------------------------------------------
def test_grid_from_flights_extends_to_the_last_arrival() -> None:
    just_inside = flights([("AAA", "BBB", "2019-01-01 22:00", 299)])  # lands 02:59 on 2 January
    on_the_edge = flights([("AAA", "BBB", "2019-01-01 22:00", 300)])  # lands 03:00, the next bin
    assert ab.grid_for_flights(just_inside).n_bins == 9  # 8 bins on day one plus the first bin of day two
    assert ab.grid_for_flights(on_the_edge).n_bins == 10


def test_save_and_load_round_trip(tmp_path) -> None:
    table, airports, grid, _ = build(flights(HAND))
    path = tmp_path / "bins.npz"
    ab.save_bin_table(path, table, airports, grid)
    table2, airports2, grid2 = ab.load_bin_table(path)
    assert np.array_equal(table, table2) and airports.tolist() == airports2.tolist() and grid == grid2


def test_loading_a_file_with_another_channel_order_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, table=np.zeros((1, 1, 3), "float32"), airports=np.array(["AAA"]), grid_start_seconds=np.int64(0),
             bin_hours=np.int64(3), n_bins=np.int64(1), channels=np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="Channel order"):
        ab.load_bin_table(path)

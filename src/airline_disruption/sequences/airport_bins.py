"""Airport-by-time-bin schedule table (the raw material of the airport sequences).

Phase 6, Step 5. Sequence version ``phase6_sequences_v1``.

What this builds
----------------
A table with one row per airport and one column per fixed UTC time bin, holding
three counts taken from the flight SCHEDULE only:

    scheduled_departures    flights scheduled to leave the airport in the bin
    scheduled_arrivals      flights scheduled to arrive at the airport in the bin
    distinct_destinations   different destinations among those departures

The table never sees an outcome. It takes the origin, the destination, the
scheduled departure time in UTC and the scheduled elapsed time, and nothing
else. Cancelled and diverted flights are counted like any other flight,
because they were on the schedule. ``REQUIRED_COLUMNS`` is checked against the
outcome-name guard in the tests.

How a sequence is read from it lives in ``lookback.py``. The table itself is a
plain count of the whole file. It is safe because a sequence only reads bins
that ended at or before the prediction time, and a test changes the flights
after that time and checks the sequence does not move.

Why bins of three hours
-----------------------
The source is a sample of about three million flights over 56 months, so one
airport has few flights per hour. Three-hour bins give usable counts and still
show the daily pattern. The width is fixed here in advance, not tuned.

Burn-in
-------
The file starts on 1 January 2019. A flight that departed on 31 December is not
in it, so arrival counts for the first hours are too low. Bins that start
before ``BURN_IN_HOURS`` after the first bin are treated as unavailable and are
masked, not read as zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCE_VERSION = "phase6_sequences_v1"

BIN_HOURS = 3
LOOKBACK_BINS = 16  # 16 bins of 3 hours = 48 hours
BURN_IN_HOURS = 24

CHANNELS = ("scheduled_departures", "scheduled_arrivals", "distinct_destinations")

# Columns the builder reads. Schedule fields only.
ORIGIN_COLUMN = "origin_airport"
DESTINATION_COLUMN = "destination_airport"
DEPARTURE_UTC_COLUMN = "scheduled_departure_utc"
ELAPSED_COLUMN = "scheduled_elapsed_time_minutes"
REQUIRED_COLUMNS = (ORIGIN_COLUMN, DESTINATION_COLUMN, DEPARTURE_UTC_COLUMN, ELAPSED_COLUMN)

# A scheduled elapsed time outside this range is treated as invalid. Such a flight
# still counts as a departure, but its arrival cannot be placed on the timeline.
MAX_PLAUSIBLE_ELAPSED_MINUTES = 24 * 60

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinGrid:
    """Fixed UTC bins. Bin ``k`` covers ``[start + k*width, start + (k+1)*width)`` in epoch seconds."""

    start_seconds: int
    bin_hours: int
    n_bins: int

    @property
    def bin_seconds(self) -> int:
        return self.bin_hours * SECONDS_PER_HOUR

    def first_valid_bin(self, burn_in_hours: int = BURN_IN_HOURS) -> int:
        """Index of the first bin that is safe to read (whole bins after the burn-in)."""
        return math.ceil(burn_in_hours / self.bin_hours)

    def bin_of(self, seconds: np.ndarray) -> np.ndarray:
        """Bin index of each instant (floor). Instants before the grid give negative indices."""
        return (np.asarray(seconds, dtype="int64") - self.start_seconds) // self.bin_seconds

    def bin_end_seconds(self, bin_index: np.ndarray) -> np.ndarray:
        """Instant at which each bin closes (exclusive end)."""
        return self.start_seconds + (np.asarray(bin_index, dtype="int64") + 1) * self.bin_seconds


def make_grid(min_seconds: int, max_seconds: int, bin_hours: int = BIN_HOURS) -> BinGrid:
    """Grid that starts at midnight UTC on the day of ``min_seconds`` and covers ``max_seconds``."""
    if SECONDS_PER_DAY % (bin_hours * SECONDS_PER_HOUR) != 0:
        raise ValueError("bin_hours must divide 24 so that bins line up with UTC days")
    if max_seconds < min_seconds:
        raise ValueError("max_seconds is earlier than min_seconds")
    start = (int(min_seconds) // SECONDS_PER_DAY) * SECONDS_PER_DAY
    width = bin_hours * SECONDS_PER_HOUR
    n_bins = (int(max_seconds) - start) // width + 1
    return BinGrid(start_seconds=start, bin_hours=bin_hours, n_bins=n_bins)


def to_epoch_seconds(stamps: pd.Series) -> np.ndarray:
    """Timezone-aware datetimes to whole seconds since 1970-01-01 UTC (int64).

    Naive timestamps are rejected: guessing a zone here would hide a bug.
    Missing timestamps are rejected too.
    """
    if stamps.isna().any():
        raise ValueError("Missing timestamps")
    if getattr(stamps.dt, "tz", None) is None:
        raise ValueError("Timestamps must be timezone-aware (UTC)")
    naive_utc = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive_utc.to_numpy().astype("datetime64[s]").astype("int64")


# ---------------------------------------------------------------------------
# Airport codes
# ---------------------------------------------------------------------------
def airport_codes(*columns: pd.Series) -> np.ndarray:
    """Sorted unique airport codes across the given columns (strings)."""
    seen: set[str] = set()
    for column in columns:
        if isinstance(column.dtype, pd.CategoricalDtype):
            values = column.cat.remove_unused_categories().cat.categories.astype(str)
        else:
            values = pd.Index(column.dropna().astype(str).unique())
        seen.update(values)
    return np.array(sorted(seen))


def codes_to_index(column: pd.Series, airports: np.ndarray) -> np.ndarray:
    """Position of each row's airport in ``airports`` (int32). An unknown or missing code raises."""
    lookup = pd.Index(airports)
    if isinstance(column.dtype, pd.CategoricalDtype):
        # Map the (few) categories, then index by the integer codes: no 3M-row string array.
        category_index = lookup.get_indexer(column.cat.categories.astype(str)).astype("int32")
        codes = column.cat.codes.to_numpy()
        if (codes < 0).any():
            raise ValueError("Missing airport codes")
        result = category_index[codes]
    else:
        if column.isna().any():
            raise ValueError("Missing airport codes")
        result = lookup.get_indexer(column.astype(str)).astype("int32")
    if (result < 0).any():
        raise ValueError("Airport code not in the airport list")
    return result


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def build_airport_bins(flights: pd.DataFrame, airports: np.ndarray, grid: BinGrid) -> tuple[np.ndarray, dict]:
    """Count scheduled departures, arrivals and distinct destinations per airport and bin.

    Returns ``(table, stats)``. ``table`` has shape ``(n_airports, n_bins, 3)``,
    float32, channel order ``CHANNELS``. ``stats`` counts the rows used and the
    arrivals that could not be placed (missing or implausible elapsed time).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in flights.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_airports, n_bins = len(airports), grid.n_bins
    origin = codes_to_index(flights[ORIGIN_COLUMN], airports).astype("int64")
    destination = codes_to_index(flights[DESTINATION_COLUMN], airports).astype("int64")
    departure = to_epoch_seconds(flights[DEPARTURE_UTC_COLUMN])
    departure_bin = grid.bin_of(departure)
    if departure_bin.min() < 0 or departure_bin.max() >= n_bins:
        raise ValueError("A scheduled departure falls outside the time grid")

    cells = n_airports * n_bins
    departures = np.bincount(origin * n_bins + departure_bin, minlength=cells)

    # Arrivals: scheduled departure plus scheduled elapsed time, in UTC, so a flight that
    # crosses midnight or a time-zone line lands in the right bin.
    elapsed = flights[ELAPSED_COLUMN].to_numpy(dtype="float64")
    placeable = np.isfinite(elapsed) & (elapsed > 0) & (elapsed <= MAX_PLAUSIBLE_ELAPSED_MINUTES)
    arrival_seconds = departure[placeable] + np.rint(elapsed[placeable] * 60).astype("int64")
    arrival_bin = grid.bin_of(arrival_seconds)
    if arrival_bin.max() >= n_bins:
        raise ValueError("A scheduled arrival falls outside the time grid; build the grid from arrivals too")
    arrivals = np.bincount(destination[placeable] * n_bins + arrival_bin, minlength=cells)

    # Distinct destinations per (origin, bin): unique (origin, bin, destination) triples, counted per cell.
    triples = np.unique((origin * n_bins + departure_bin) * n_airports + destination)
    distinct = np.bincount(triples // n_airports, minlength=cells)

    table = np.stack([departures, arrivals, distinct], axis=-1).reshape(n_airports, n_bins, len(CHANNELS)).astype("float32")
    stats = {
        "rows": int(len(flights)),
        "arrivals_counted": int(placeable.sum()),
        "arrivals_not_placeable": int((~placeable).sum()),
    }
    return table, stats


def grid_for_flights(flights: pd.DataFrame, bin_hours: int = BIN_HOURS) -> BinGrid:
    """Grid covering every scheduled departure and every placeable scheduled arrival."""
    departure = to_epoch_seconds(flights[DEPARTURE_UTC_COLUMN])
    elapsed = flights[ELAPSED_COLUMN].to_numpy(dtype="float64")
    placeable = np.isfinite(elapsed) & (elapsed > 0) & (elapsed <= MAX_PLAUSIBLE_ELAPSED_MINUTES)
    latest = int(departure.max())
    if placeable.any():
        latest = max(latest, int((departure[placeable] + np.rint(elapsed[placeable] * 60).astype("int64")).max()))
    return make_grid(int(departure.min()), latest, bin_hours)


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------
def save_bin_table(path: str | Path, table: np.ndarray, airports: np.ndarray, grid: BinGrid) -> None:
    """Write the table and everything needed to read it back (no pickles)."""
    np.savez_compressed(
        path,
        table=table,
        airports=airports,
        grid_start_seconds=np.int64(grid.start_seconds),
        bin_hours=np.int64(grid.bin_hours),
        n_bins=np.int64(grid.n_bins),
        channels=np.array(CHANNELS),
        sequence_version=np.array(SEQUENCE_VERSION),
    )


def load_bin_table(path: str | Path) -> tuple[np.ndarray, np.ndarray, BinGrid]:
    """Read a table written by ``save_bin_table``. Returns ``(table, airports, grid)``."""
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["channels"]) != CHANNELS:
            raise ValueError("Channel order in the file does not match this code version")
        grid = BinGrid(int(data["grid_start_seconds"]), int(data["bin_hours"]), int(data["n_bins"]))
        return data["table"], data["airports"], grid

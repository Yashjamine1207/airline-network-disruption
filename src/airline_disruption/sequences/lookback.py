"""Read fixed-length lookback sequences from the airport-bin table.

Phase 6, Step 5.

The rule that keeps a sequence point-in-time safe
-------------------------------------------------
A flight is predicted at ``T = scheduled departure (UTC) - 2 hours``. Its
sequence is the ``lookback`` bins that CLOSED at or before ``T``.

    end_bin = floor((T - grid_start) / bin_width)

Bins ``0 .. end_bin - 1`` are complete at ``T``. Bin ``end_bin`` contains ``T``
and is still open, so it is never read. The newest position in a sequence is
therefore bin ``end_bin - 1``, whose end time is at or before ``T``.

Padding and masks
-----------------
Positions run oldest to newest. A position is REAL when its bin index is at or
after ``first_valid_bin`` (after the burn-in). Otherwise it is padding: the
values are set to 0 and the mask is False. Padding only ever sits at the oldest
end, so the mask is a run of False followed by a run of True.

Zero is a real value for a real bin (an airport with no scheduled flights in
three hours), so the values alone cannot tell padding from a quiet bin. Always
pass the mask to the model.
"""

from __future__ import annotations

import numpy as np

from airline_disruption.sequences.airport_bins import LOOKBACK_BINS, BinGrid


def end_bins(prediction_seconds: np.ndarray, grid: BinGrid) -> np.ndarray:
    """Index one past the newest bin that has closed by each prediction time."""
    return grid.bin_of(prediction_seconds)


def assert_no_bin_after_prediction(end_bin: np.ndarray, prediction_seconds: np.ndarray, grid: BinGrid) -> None:
    """Raise ValueError if the newest bin of any sequence closes after its prediction time.

    ``end_bin - 1`` is the newest bin read; it closes at ``grid_start + end_bin * width``.
    """
    newest_close = grid.start_seconds + np.asarray(end_bin, dtype="int64") * grid.bin_seconds
    late = newest_close > np.asarray(prediction_seconds, dtype="int64")
    if late.any():
        raise ValueError(f"{int(late.sum()):,} sequences would include a bin that closes after the prediction time")


def count_valid_bins(end_bin: np.ndarray, grid: BinGrid, lookback: int = LOOKBACK_BINS, burn_in_hours: int | None = None) -> np.ndarray:
    """Number of real (non-padding) positions in each sequence, from 0 to ``lookback``."""
    first_valid = _first_valid(grid, burn_in_hours)
    return np.clip(np.asarray(end_bin, dtype="int64") - first_valid, 0, lookback).astype("int16")


def gather_sequences(
    table: np.ndarray,
    airport_index: np.ndarray,
    end_bin: np.ndarray,
    grid: BinGrid,
    lookback: int = LOOKBACK_BINS,
    burn_in_hours: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, mask)`` for a batch of flights.

    ``values`` has shape ``(n, lookback, channels)``, float32, oldest position first,
    padding set to 0. ``mask`` has shape ``(n, lookback)``, True for real positions.
    """
    airport_index = np.asarray(airport_index, dtype="int64")
    end_bin = np.asarray(end_bin, dtype="int64")
    if airport_index.shape != end_bin.shape:
        raise ValueError("airport_index and end_bin must have the same shape")
    if len(end_bin) and (end_bin.max() > table.shape[1]):
        raise ValueError("end_bin is beyond the last bin of the table")
    first_valid = _first_valid(grid, burn_in_hours)

    positions = end_bin[:, None] + np.arange(-lookback, 0, dtype="int64")[None, :]  # oldest first
    mask = positions >= first_valid
    safe = np.where(mask, positions, 0)  # any valid index; the value is overwritten below
    values = table[airport_index[:, None], safe]
    values = np.where(mask[:, :, None], values, 0.0).astype("float32")
    return values, mask


def prior_counts(
    key: np.ndarray,
    departure_seconds: np.ndarray,
    query_key: np.ndarray,
    query_seconds: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """For each query, how many flights with the same key departed in ``[t - window, t)``.

    Used to compare how dense a sequence would be for an airport, a route or a
    carrier at an airport. Sorting once and using two binary searches per query
    keeps it fast for millions of rows.
    """
    key = np.asarray(key, dtype="int64")
    departure_seconds = np.asarray(departure_seconds, dtype="int64")
    query_key = np.asarray(query_key, dtype="int64")
    query_seconds = np.asarray(query_seconds, dtype="int64")

    base = min(int(departure_seconds.min()), int((query_seconds - window_seconds).min()))
    span = max(int(departure_seconds.max()), int(query_seconds.max())) - base + 1
    packed = np.sort(key * span + (departure_seconds - base))
    upper = np.searchsorted(packed, query_key * span + (query_seconds - base), side="left")
    lower = np.searchsorted(packed, query_key * span + (query_seconds - window_seconds - base), side="left")
    return (upper - lower).astype("int32")


def _first_valid(grid: BinGrid, burn_in_hours: int | None) -> int:
    return grid.first_valid_bin() if burn_in_hours is None else grid.first_valid_bin(burn_in_hours)

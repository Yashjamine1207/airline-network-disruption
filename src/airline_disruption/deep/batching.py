"""Batches for the neural networks, built without TensorFlow.

Phase 6, Step 7.

A ``BatchSource`` holds one set of rows (fit, early-stop or validation). For each
batch it returns a dict of NumPy arrays and the labels. The airport sequences are
NOT stored per row: each batch reads its rows' lookbacks from the airport table
(``lookback.gather_sequences``) and scales them. That keeps memory small and lets
later steps change the lookback length without rebuilding anything.

Keeping this free of TensorFlow means the ordering, alignment and shuffling rules
are tested with plain NumPy, and the Keras wrapper in ``keras_models.py`` only
forwards to it.

Input names (they must match the Keras model's input names):
    cat_origin, cat_destination, cat_route, cat_carrier   int32, shape (batch,)
    static_numeric                                        float32, (batch, 10)
    sequence                                              float32, (batch, L, 3)  [sequence models only]
    sequence_mask                                         bool,    (batch, L)     [sequence models only]
"""

from __future__ import annotations

import math

import numpy as np

from airline_disruption.deep.inputs import CATEGORICAL_COLUMNS, DeepInputs
from airline_disruption.sequences.airport_bins import LOOKBACK_BINS, BinGrid
from airline_disruption.sequences.lookback import gather_sequences

CATEGORICAL_INPUT_NAMES = ("cat_origin", "cat_destination", "cat_route", "cat_carrier")
assert len(CATEGORICAL_INPUT_NAMES) == len(CATEGORICAL_COLUMNS)


class BatchSource:
    """Batches of one row set, optionally shuffled per epoch with a fixed seed."""

    def __init__(
        self,
        inputs: DeepInputs,
        rows: np.ndarray,
        table: np.ndarray | None,
        grid: BinGrid | None,
        batch_size: int = 2048,
        lookback: int = LOOKBACK_BINS,
        use_sequence: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        if use_sequence and (table is None or grid is None):
            raise ValueError("A sequence model needs the airport table and its grid")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.inputs = inputs
        self.rows = np.asarray(rows, dtype="int64")
        self.table, self.grid = table, grid
        self.batch_size = batch_size
        self.lookback = lookback
        self.use_sequence = use_sequence
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def order(self, epoch: int = 0) -> np.ndarray:
        """Positions ``0..len(rows)-1`` in the order used during ``epoch``."""
        if not self.shuffle:
            return np.arange(len(self.rows))
        return np.random.default_rng(self.seed + epoch).permutation(len(self.rows))

    def get(self, batch_index: int, order: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """``(x, y)`` for one batch. ``order`` defaults to the unshuffled order."""
        if not 0 <= batch_index < len(self):
            raise IndexError(batch_index)
        order = self.order() if order is None else order
        chosen = self.rows[order[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]]

        x: dict[str, np.ndarray] = {
            name: self.inputs.codes[chosen, i] for i, name in enumerate(CATEGORICAL_INPUT_NAMES)
        }
        x["static_numeric"] = self.inputs.numeric[chosen]
        if self.use_sequence:
            values, mask = gather_sequences(
                self.table, self.inputs.airport_index[chosen], self.inputs.end_bin[chosen], self.grid, self.lookback
            )
            x["sequence"] = self.inputs.scaler.apply(values, mask)
            x["sequence_mask"] = mask
        for name in ("static_numeric", "sequence"):
            # A ReLU network silently turns NaN into 0 and then predicts the base rate, so a NaN
            # must stop the run here instead of producing plausible-looking scores.
            if name in x and not np.isfinite(x[name]).all():
                raise ValueError(f"Batch {batch_index} has non-finite values in {name}")
        return x, self.inputs.label[chosen].astype("float32")

    def labels(self) -> np.ndarray:
        """Labels of all rows in the unshuffled order."""
        return self.inputs.label[self.rows]

    def row_ids(self) -> np.ndarray:
        """``source_row_number`` of all rows in the unshuffled order."""
        return self.inputs.source_row_number[self.rows]

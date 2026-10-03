"""Tests for BatchSource: alignment, shuffling, isolation between row sets, sequence reads."""

from __future__ import annotations

import numpy as np
import pytest

from airline_disruption.deep.batching import BatchSource
from airline_disruption.deep.inputs import ROLE_CODES, DeepInputs, SequenceScaler, StaticSpec
from airline_disruption.sequences.airport_bins import BinGrid
from airline_disruption.sequences.lookback import gather_sequences

N = 50
GRID = BinGrid(start_seconds=0, bin_hours=3, n_bins=200)
SCALER = SequenceScaler([1.0, 2.0, 0.5], [2.0, 1.5, 1.0])


def make_inputs() -> DeepInputs:
    rng = np.random.default_rng(0)
    idx = np.arange(N)
    codes = np.stack([idx, idx + 100, idx + 200, idx + 300], axis=1).astype("int32")  # every cell identifies its row
    numeric = np.stack([idx] + [rng.normal(size=N) for _ in range(9)], axis=1).astype("float32")
    end_bin = np.where(idx % 5 == 0, 3 + idx % 4, 60 + idx).astype("int32")  # every fifth row has a short history
    return DeepInputs(
        codes=codes,
        numeric=numeric,
        label=(idx % 2).astype("int8"),
        role=np.where(idx < 30, ROLE_CODES["fit"], ROLE_CODES["validation"]).astype("int8"),
        airport_index=(idx % 3).astype("int16"),
        end_bin=end_bin,
        source_row_number=(1000 + idx).astype("int64"),
        prediction_seconds=(idx * 3600).astype("int64"),
        spec=StaticSpec({}, {}, {}, {}),
        scaler=SCALER,
    )


def make_table() -> np.ndarray:
    a, b, c = np.meshgrid(np.arange(3), np.arange(200), np.arange(3), indexing="ij")
    return (a * 7 + (b % 11) + c).astype("float32")  # small non-negative counts


def source(rows=None, **kw) -> BatchSource:
    inputs = make_inputs()
    rows = np.arange(N) if rows is None else rows
    return BatchSource(inputs, rows, make_table(), GRID, **{"batch_size": 8, **kw})


def test_unshuffled_batches_cover_every_row_once_in_order() -> None:
    s = source()
    assert len(s) == 7  # 6 full batches of 8 and one of 2
    seen = []
    for i in range(len(s)):
        x, y = s.get(i)
        assert len(y) == len(x["cat_origin"]) == len(x["static_numeric"]) == len(x["sequence"]) == len(x["sequence_mask"])
        seen.extend(x["cat_origin"].tolist())
    assert seen == list(range(N))
    assert len(s.get(6)[1]) == 2


def test_each_batch_row_is_aligned_across_codes_numeric_label_and_sequence() -> None:
    inputs, table = make_inputs(), make_table()
    s = BatchSource(inputs, np.arange(N), table, GRID, batch_size=16)
    x, y = s.get(1)
    rows = np.arange(16, 32)
    assert x["cat_origin"].tolist() == rows.tolist()
    assert x["cat_destination"].tolist() == (rows + 100).tolist()
    assert x["cat_route"].tolist() == (rows + 200).tolist()
    assert x["cat_carrier"].tolist() == (rows + 300).tolist()
    assert x["static_numeric"][:, 0].tolist() == rows.tolist()
    assert y.tolist() == (rows % 2).tolist() and y.dtype == np.float32
    values, mask = gather_sequences(table, inputs.airport_index[rows], inputs.end_bin[rows], GRID)
    assert np.array_equal(x["sequence"], SCALER.apply(values, mask)) and np.array_equal(x["sequence_mask"], mask)


def test_rows_with_short_history_are_padded_and_masked_inside_a_batch() -> None:
    x, _ = source(batch_size=N).get(0)
    short = np.arange(N) % 5 == 0
    assert (x["sequence_mask"][~short]).all()
    assert x["sequence_mask"][short].sum(axis=1).max() < 16  # burn-in leaves fewer than 16 real bins
    assert (x["sequence"][~x["sequence_mask"]] == 0).all()  # padded positions are exactly zero after scaling


def test_a_source_only_ever_returns_its_own_rows() -> None:
    fit_rows = np.flatnonzero(make_inputs().role == ROLE_CODES["fit"])
    validation_rows = np.flatnonzero(make_inputs().role == ROLE_CODES["validation"])
    for rows in (fit_rows, validation_rows):
        s = source(rows, shuffle=True)
        seen = np.concatenate([s.get(i, s.order(3))[0]["cat_origin"] for i in range(len(s))])
        assert sorted(seen.tolist()) == sorted(rows.tolist())
    assert not set(fit_rows) & set(validation_rows)


def test_shuffle_is_a_reproducible_permutation_that_changes_each_epoch() -> None:
    s = source(shuffle=True, seed=7)
    a0, a0_again, a1 = s.order(0), s.order(0), s.order(1)
    assert np.array_equal(a0, a0_again) and not np.array_equal(a0, a1)
    assert sorted(a0.tolist()) == list(range(N)) and sorted(a1.tolist()) == list(range(N))
    assert not np.array_equal(a0, np.arange(N))
    assert not np.array_equal(source(shuffle=True, seed=8).order(0), a0)
    # Every row appears exactly once per epoch, with its own label.
    seen, labels = [], []
    for i in range(len(s)):
        x, y = s.get(i, a1)
        seen.extend(x["cat_origin"].tolist()); labels.extend(y.tolist())
    assert sorted(seen) == list(range(N))
    assert all(label == row % 2 for row, label in zip(seen, labels))


def test_unshuffled_order_ignores_the_epoch() -> None:
    s = source(shuffle=False)
    assert np.array_equal(s.order(0), s.order(5))


def test_a_tabular_only_source_has_no_sequence_inputs_and_needs_no_table() -> None:
    s = BatchSource(make_inputs(), np.arange(N), None, None, batch_size=10, use_sequence=False)
    x, y = s.get(0)
    assert set(x) == {"cat_origin", "cat_destination", "cat_route", "cat_carrier", "static_numeric"}
    assert len(y) == 10


def test_a_sequence_source_without_a_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="airport table"):
        BatchSource(make_inputs(), np.arange(N), None, None, use_sequence=True)


def test_a_shorter_lookback_is_the_newest_slice_of_the_longer_one() -> None:
    long_x, _ = source(batch_size=N, lookback=16).get(0)
    short_x, _ = source(batch_size=N, lookback=6).get(0)
    assert short_x["sequence"].shape == (N, 6, 3)
    assert np.array_equal(short_x["sequence"], long_x["sequence"][:, -6:])
    assert np.array_equal(short_x["sequence_mask"], long_x["sequence_mask"][:, -6:])


def test_labels_and_ids_follow_the_unshuffled_row_order() -> None:
    rows = np.array([4, 9, 2])
    s = source(rows)
    assert s.labels().tolist() == [0, 1, 0]
    assert s.row_ids().tolist() == [1004, 1009, 1002]


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        source(batch_size=0)
    with pytest.raises(IndexError):
        source().get(99)
    with pytest.raises(IndexError):
        source().get(-1)


def test_reading_batches_does_not_change_the_saved_arrays() -> None:
    inputs, table = make_inputs(), make_table()
    before = (inputs.codes.copy(), inputs.numeric.copy(), inputs.end_bin.copy(), table.copy())
    s = BatchSource(inputs, np.arange(N), table, GRID, batch_size=7, shuffle=True)
    for i in range(len(s)):
        s.get(i, s.order(1))
    assert np.array_equal(before[0], inputs.codes) and np.array_equal(before[1], inputs.numeric)
    assert np.array_equal(before[2], inputs.end_bin) and np.array_equal(before[3], table)


@pytest.mark.parametrize("field", ["numeric", "table"])
def test_non_finite_inputs_stop_the_run_instead_of_reaching_the_model(field: str) -> None:
    """A ReLU network turns NaN into 0 and then predicts the base rate. That must never happen quietly."""
    inputs, table = make_inputs(), make_table()
    if field == "numeric":
        inputs.numeric[3, 4] = np.nan
    else:
        table[:] = np.nan
    s = BatchSource(inputs, np.arange(N), table, GRID, batch_size=N)
    with pytest.raises(ValueError, match="non-finite"):
        s.get(0)

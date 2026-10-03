"""Tests for the sequence perturbations (Phase 7A, Step 10). NumPy only."""

from __future__ import annotations

import numpy as np
import pytest

from airline_disruption.deep import perturb


def batch(n: int = 6, length: int = 8, channels: int = 3, real: int = 6):
    rng = np.random.default_rng(0)
    mask = np.zeros((n, length), dtype=bool)
    mask[:, length - real :] = True
    seq = np.where(mask[:, :, None], rng.normal(size=(n, length, channels)), 0.0).astype("float32")
    return seq, mask


def test_remove_sequence_gives_zeros_and_an_all_false_mask_and_leaves_the_input_alone() -> None:
    seq, mask = batch()
    before = seq.copy(), mask.copy()
    new_seq, new_mask = perturb.remove_sequence(seq, mask)
    assert not new_seq.any() and not new_mask.any() and new_seq.shape == seq.shape and new_seq.dtype == seq.dtype
    assert np.array_equal(seq, before[0]) and np.array_equal(mask, before[1])


def test_mask_bins_hides_only_the_requested_positions() -> None:
    seq, mask = batch()
    new_seq, new_mask = perturb.mask_bins(seq, mask, 5, 7)
    assert not new_mask[:, 5:7].any() and (new_seq[:, 5:7] == 0).all()
    assert np.array_equal(new_mask[:, :5], mask[:, :5]) and np.array_equal(new_seq[:, :5], seq[:, :5])
    assert np.array_equal(new_mask[:, 7:], mask[:, 7:]) and np.array_equal(new_seq[:, 7:], seq[:, 7:])


def test_keep_newest_keeps_exactly_k_bins() -> None:
    seq, mask = batch(real=8)
    for k in (1, 3, 8):
        new_seq, new_mask = perturb.keep_newest(seq, mask, k)
        assert new_mask.sum(axis=1).tolist() == [k] * len(mask) and new_mask[:, -k:].all()
        assert np.array_equal(new_seq[:, -k:], seq[:, -k:])
    full_seq, full_mask = perturb.keep_newest(seq, mask, 8)
    assert np.array_equal(full_seq, seq) and full_seq is not seq


def test_shuffle_rows_is_a_deterministic_permutation_that_moves_sequence_and_mask_together() -> None:
    seq, mask = batch()
    seq[:, :, 0] = np.arange(len(seq))[:, None]  # tag each row
    mask[:, 0] = np.arange(len(mask)) % 2 == 0
    a_seq, a_mask = perturb.shuffle_rows(seq, mask, seed=3)
    b_seq, b_mask = perturb.shuffle_rows(seq, mask, seed=3)
    assert np.array_equal(a_seq, b_seq) and np.array_equal(a_mask, b_mask)
    tags = a_seq[:, 0, 0].astype(int)
    assert sorted(tags.tolist()) == list(range(len(seq)))  # every lookback used once
    assert not np.array_equal(tags, np.arange(len(seq)))
    assert np.array_equal(a_mask, mask[tags])  # the mask followed its sequence


def test_bad_arguments_are_rejected() -> None:
    seq, mask = batch()
    with pytest.raises(ValueError, match="start < stop"):
        perturb.mask_bins(seq, mask, 4, 4)
    with pytest.raises(ValueError, match="start < stop"):
        perturb.mask_bins(seq, mask, 0, 9)
    with pytest.raises(ValueError, match="between 1"):
        perturb.keep_newest(seq, mask, 0)
    with pytest.raises(ValueError, match="same batch"):
        perturb.remove_sequence(seq, mask[:-1])


# ---------------------------------------------------------------------------
# Scoring passes: PerturbedSource, build_passes, real_steps, verdict
# ---------------------------------------------------------------------------
from airline_disruption.deep.batching import BatchSource  # noqa: E402
from airline_disruption.deep.perturb import MATERIAL_PR_AUC, PerturbedSource, build_passes, real_steps, verdict  # noqa: E402
from airline_disruption.deep.inputs import ROLE_CODES, DeepInputs, SequenceScaler, StaticSpec  # noqa: E402
from airline_disruption.sequences.airport_bins import BinGrid  # noqa: E402

N = 50
GRID = BinGrid(start_seconds=0, bin_hours=3, n_bins=200)


def make_inputs() -> DeepInputs:
    """Small synthetic inputs (same idea as tests/deep/test_batching.py): every fifth row has a short history."""
    idx = np.arange(N)
    rng = np.random.default_rng(0)
    return DeepInputs(
        codes=np.stack([idx, idx + 100, idx + 200, idx + 300], axis=1).astype("int32"),
        numeric=np.stack([idx] + [rng.normal(size=N) for _ in range(9)], axis=1).astype("float32"),
        label=(idx % 2).astype("int8"),
        role=np.full(N, ROLE_CODES["validation"], dtype="int8"),
        airport_index=(idx % 3).astype("int16"),
        end_bin=np.where(idx % 5 == 0, 3 + idx % 4, 60 + idx).astype("int32"),
        source_row_number=(1000 + idx).astype("int64"),
        prediction_seconds=(idx * 3600).astype("int64"),
        spec=StaticSpec({}, {}, {}, {}),
        scaler=SequenceScaler([1.0, 2.0, 0.5], [2.0, 1.5, 1.0]),
    )


def make_table() -> np.ndarray:
    a, b, c = np.meshgrid(np.arange(3), np.arange(200), np.arange(3), indexing="ij")
    return (a * 7 + (b % 11) + c).astype("float32")


def _sources(change):
    inputs, table = make_inputs(), make_table()
    plain = BatchSource(inputs, np.arange(N), table, GRID, batch_size=16)
    changed = PerturbedSource(inputs, np.arange(N), table, GRID, batch_size=16, change=change)
    return plain, changed


def test_perturbed_source_changes_only_the_sequence() -> None:
    plain, changed = _sources(lambda s, m, b: perturb.remove_sequence(s, m))
    for i in range(len(plain)):
        (xp, yp), (xc, yc) = plain.get(i), changed.get(i)
        assert np.array_equal(yp, yc)
        for name in ("cat_origin", "cat_destination", "cat_route", "cat_carrier", "static_numeric"):
            assert np.array_equal(xp[name], xc[name])
        assert not xc["sequence"].any() and not xc["sequence_mask"].any()
        assert xp["sequence_mask"].any()  # the plain source is untouched by the change


def test_perturbed_source_without_a_change_equals_the_plain_source() -> None:
    plain, same = _sources(None)
    for i in range(len(plain)):
        (xp, _), (xs, _) = plain.get(i), same.get(i)
        assert all(np.array_equal(xp[k], xs[k]) for k in xp)


def test_change_receives_the_batch_index_so_shuffles_differ_between_batches() -> None:
    seen = []
    _, changed = _sources(lambda s, m, b: (seen.append(b) or (s, m)))
    for i in range(len(changed)):
        changed.get(i)
    assert seen == list(range(len(changed)))


def test_build_passes_masks_cover_the_lookback_once_and_names_are_unique() -> None:
    for lookback in (4, 6, 16):
        passes = build_passes(lookback, seed=1)
        names = [p[0] for p in passes]
        assert names[0] == "baseline" and passes[0][2] is None and len(set(names)) == len(names)
        seq, mask = batch(n=5, length=lookback, real=lookback)
        hidden = np.zeros(lookback, dtype=int)
        for name, _, change in passes:
            if name.startswith("mask_"):
                _, new_mask = change(seq, mask, 0)
                hidden += (~new_mask[0]).astype(int)
        assert hidden.tolist() == [1] * lookback  # the four windows are disjoint and together hide every bin
    assert {"no_sequence", "shuffled_lookbacks", "keep_newest_1", "keep_newest_4", "keep_newest_8"} <= set(names)


def test_build_passes_skips_keep_sizes_that_are_not_shorter_than_the_lookback_and_rejects_tiny_lookbacks() -> None:
    names = [p[0] for p in build_passes(8, 0)]
    assert "keep_newest_8" not in names and "keep_newest_4" in names
    with pytest.raises(ValueError):
        build_passes(3, 0)


def test_real_steps_counts_unmasked_bins_per_row_in_row_order() -> None:
    inputs, table = make_inputs(), make_table()
    source = BatchSource(inputs, np.arange(N), table, GRID, batch_size=16)
    counts = real_steps(source)
    assert counts.shape == (N,) and counts.max() == 16
    assert (counts[np.arange(N) % 5 == 0] < 16).all() and (counts[np.arange(N) % 5 != 0] == 16).all()


def test_verdict_needs_the_whole_interval_on_one_side_and_calls_small_changes_tiny() -> None:
    assert "relies" in verdict(-0.01, -0.02, -0.005)
    assert "does better without" in verdict(0.01, 0.005, 0.02)
    assert "no clear change" in verdict(-0.01, -0.02, 0.001)
    assert "no clear change" in verdict(0.01, -0.001, 0.02)
    assert "tiny" in verdict(-MATERIAL_PR_AUC / 10, -MATERIAL_PR_AUC / 5, -MATERIAL_PR_AUC / 50)
    assert "tiny" in verdict(MATERIAL_PR_AUC / 10, MATERIAL_PR_AUC / 50, MATERIAL_PR_AUC / 5)
    assert "no clear change" in verdict(0.0, 0.0, 0.001)  # an interval touching zero is not clear


def test_shuffled_lookbacks_pass_uses_a_different_permutation_for_each_batch_and_is_repeatable() -> None:
    change = {name: c for name, _, c in build_passes(8, seed=5)}["shuffled_lookbacks"]
    seq, mask = batch(n=12, length=8, real=8)
    first, again, other = change(seq, mask, 0)[0], change(seq, mask, 0)[0], change(seq, mask, 1)[0]
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert not np.array_equal(first, seq)  # something actually moved


def test_named_passes_do_what_their_names_say() -> None:
    passes = {name: c for name, _, c in build_passes(16, seed=0)}
    seq, mask = batch(n=6, length=16, real=16)
    s, m = passes["no_sequence"](seq, mask, 0)
    assert not s.any() and not m.any()
    for k in (1, 4, 8):
        s, m = passes[f"keep_newest_{k}"](seq, mask, 0)
        assert m.sum(axis=1).tolist() == [k] * 6 and np.array_equal(s[:, -k:], seq[:, -k:])
    s, m = passes["mask_newest_quarter"](seq, mask, 0)
    assert not m[:, 12:].any() and m[:, :12].all()
    s, m = passes["mask_oldest_quarter"](seq, mask, 0)
    assert not m[:, :4].any() and m[:, 4:].all()

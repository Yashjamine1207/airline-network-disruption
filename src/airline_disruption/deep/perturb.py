"""Input perturbations for the sequence-explainability study (NumPy only, no TensorFlow).

Phase 7A, Step 10. The scope asks for temporal masking, lookback removal and permutation of the sequence
input. Each function takes one batch of ``sequence`` (batch, L, channels) and ``mask`` (batch, L, True =
real step) and returns NEW arrays; the inputs are never modified. Masked steps are set to 0 and the mask is
set to False, which is exactly how the model saw padding during training.

A perturbed score answers "how much does THIS trained network rely on that part of the input". It does not
say the input is causally useful in the world, and a network shown a pattern it never saw in training (for
example a fully masked lookback for a flight that has history) may behave oddly. Both limits are stated next
to the results.
"""

from __future__ import annotations

import numpy as np

from airline_disruption.deep.batching import BatchSource


def _check(sequence: np.ndarray, mask: np.ndarray) -> None:
    if sequence.ndim != 3 or mask.ndim != 2 or sequence.shape[:2] != mask.shape:
        raise ValueError("sequence must be (batch, L, channels) and mask (batch, L) with the same batch and L")


def remove_sequence(sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lookback removal: no real step at all. The network sees only its static inputs."""
    _check(sequence, mask)
    return np.zeros_like(sequence), np.zeros_like(mask)


def mask_bins(sequence: np.ndarray, mask: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Temporal masking: hide positions ``start`` up to (not including) ``stop``. 0 is the oldest bin."""
    _check(sequence, mask)
    length = mask.shape[1]
    if not 0 <= start < stop <= length:
        raise ValueError(f"Need 0 <= start < stop <= {length}")
    new_sequence, new_mask = sequence.copy(), mask.copy()
    new_sequence[:, start:stop] = 0.0
    new_mask[:, start:stop] = False
    return new_sequence, new_mask


def keep_newest(sequence: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Shorter lookback: keep only the newest ``k`` bins and hide the rest."""
    _check(sequence, mask)
    length = mask.shape[1]
    if not 1 <= k <= length:
        raise ValueError(f"k must be between 1 and {length}")
    if k == length:
        return sequence.copy(), mask.copy()
    return mask_bins(sequence, mask, 0, length - k)


def shuffle_rows(sequence: np.ndarray, mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Permutation: give every flight another flight's lookback (sequence and mask move together).

    The static inputs stay with their own flight. The lookbacks keep their realistic values and shapes, but
    no longer describe the airport the flight departs from. If the score does not change, the network was
    not using the lookback of its own airport.
    """
    _check(sequence, mask)
    order = np.random.default_rng(seed).permutation(len(sequence))
    return sequence[order].copy(), mask[order].copy()


# ---------------------------------------------------------------------------
# Scoring passes used by scripts/run_phase7a_sequence_perturbation.py
# ---------------------------------------------------------------------------
MATERIAL_PR_AUC = 0.0005  # display rule only: smaller changes are called "tiny" even when the interval excludes zero


class PerturbedSource(BatchSource):
    """A BatchSource whose sequence input is changed after scaling, batch by batch.

    ``change(sequence, mask, batch_index)`` must return a new ``(sequence, mask)``. Static inputs and labels
    are never touched, so any change in the score comes from the lookback alone.
    """

    def __init__(self, *args, change=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.change = change

    def get(self, batch_index: int, order: np.ndarray | None = None):
        x, y = super().get(batch_index, order)
        if self.change is not None:
            x["sequence"], x["sequence_mask"] = self.change(x["sequence"], x["sequence_mask"], batch_index)
        return x, y


def build_passes(lookback: int, seed: int) -> list[tuple[str, str, object]]:
    """``(name, description, change)`` for every scoring pass. ``change`` is None for the baseline.

    The four ``mask_*_quarter`` passes hide disjoint windows that together cover the whole lookback.
    """
    if lookback < 4:
        raise ValueError("The perturbation study needs a lookback of at least 4 bins")
    passes: list[tuple[str, str, object]] = [
        ("baseline", "trained network, inputs unchanged", None),
        ("no_sequence", "lookback removed (no real step)", lambda s, m, b: remove_sequence(s, m)),
        ("shuffled_lookbacks", "each flight gets another flight's lookback (same batch)",
         lambda s, m, b: shuffle_rows(s, m, seed + b)),
    ]
    for k in sorted({1, 4, 8}):
        if k < lookback:
            passes.append((f"keep_newest_{k}", f"only the newest {k} of {lookback} bins kept",
                           lambda s, m, b, k=k: keep_newest(s, m, k)))
    width = lookback // 4
    names = ("oldest", "second_oldest", "second_newest", "newest")
    for i, name in enumerate(names):
        start = i * width
        stop = lookback if i == len(names) - 1 else (i + 1) * width
        passes.append((f"mask_{name}_quarter", f"bins {start} to {stop - 1} hidden (0 is the oldest)",
                       lambda s, m, b, start=start, stop=stop: mask_bins(s, m, start, stop)))
    return passes


def real_steps(source: BatchSource) -> np.ndarray:
    """Number of real (unmasked) lookback bins for every row of ``source``, in row order."""
    return np.concatenate([source.get(i)[0]["sequence_mask"].sum(axis=1) for i in range(len(source))]).astype("int64")


def verdict(delta: float, ci_low: float, ci_high: float, material: float = MATERIAL_PR_AUC) -> str:
    """One sentence for a change in PR-AUC (perturbed minus baseline) and its bootstrap interval."""
    if ci_high < 0:
        if abs(delta) < material:
            return f"PR-AUC falls by less than {material:g}: statistically clear but tiny"
        return "PR-AUC falls, whole interval below zero: the network relies on this input"
    if ci_low > 0:
        if abs(delta) < material:
            return f"PR-AUC rises by less than {material:g}: statistically clear but tiny"
        return "PR-AUC rises, whole interval above zero: the network does better without this input"
    return "no clear change: the interval includes zero"

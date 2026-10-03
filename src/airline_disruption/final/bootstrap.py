"""Day-level bootstrap for PR-AUC, lift and paired PR-AUC differences.

Phase 7B, Step 12. Whole UTC prediction days are resampled with replacement (flights on one day share their
conditions). Resampling days is the same as giving every flight an integer weight (how many times its day was
drawn), and PR-AUC can be computed with those weights from ONE sort of the scores. That makes a resample cost a
few milliseconds instead of a sort, so 1,000 resamples of several scorers are cheap.

PR-AUC here is average precision as scikit-learn defines it: sum over distinct score thresholds of
(recall step) x (precision at that threshold). Tied scores are one threshold. The tests compare it with
scikit-learn on the same rows and on rows duplicated by the weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DayBootstrap:
    """Fixed day resamples, shared by every score vector so differences are paired."""

    def __init__(self, y, day_codes, n_boot: int = 500, seed: int = 42, confidence: float = 0.95) -> None:
        self.y = np.asarray(y).astype("int64")
        if not np.isin(self.y, (0, 1)).all() or self.y.min() == self.y.max():
            raise ValueError("labels must be 0 or 1 with both classes present")
        codes, _ = pd.factorize(pd.Series(np.asarray(day_codes)))
        if len(codes) != len(self.y):
            raise ValueError("day_codes must have one entry per row")
        self.codes = codes.astype("int64")
        self.n_days = int(self.codes.max()) + 1
        self.n_boot = int(n_boot)
        self.confidence = confidence
        draws = np.random.default_rng(seed).integers(0, self.n_days, size=(self.n_boot, self.n_days))
        self.day_weights = np.stack([np.bincount(d, minlength=self.n_days) for d in draws]).astype("int64") if self.n_boot else np.zeros((0, self.n_days), "int64")
        self.flights_by_day = np.bincount(self.codes, minlength=self.n_days)
        self.events_by_day = np.bincount(self.codes, weights=self.y, minlength=self.n_days)

    def _sorted(self, score):
        s = np.asarray(score, dtype="float64")
        if len(s) != len(self.y) or not np.isfinite(s).all():
            raise ValueError("score must be finite and have one entry per row")
        order = np.argsort(-s, kind="stable")
        s_sorted = s[order]
        ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])  # last row of each run of equal scores
        return order, ends

    @staticmethod
    def _average_precision(weight, y_sorted, ends) -> float:
        tp = np.cumsum(weight * y_sorted)[ends]
        seen = np.cumsum(weight)[ends]
        positives = tp[-1]
        if positives <= 0:
            return float("nan")
        precision = np.divide(tp, seen, out=np.zeros(len(tp), dtype="float64"), where=seen > 0)
        recall = tp / positives
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def point(self, score) -> float:
        """PR-AUC on the real rows (every weight 1)."""
        order, ends = self._sorted(score)
        return self._average_precision(np.ones(len(order), dtype="int64"), self.y[order], ends)

    def draws(self, score) -> tuple[np.ndarray, np.ndarray]:
        """``(pr_auc, lift)`` for every resample."""
        order, ends = self._sorted(score)
        y_sorted, codes_sorted = self.y[order], self.codes[order]
        pr, lift = np.full(self.n_boot, np.nan), np.full(self.n_boot, np.nan)
        for b in range(self.n_boot):
            weights = self.day_weights[b]
            row_weight = weights[codes_sorted]
            pr[b] = self._average_precision(row_weight, y_sorted, ends)
            prevalence = (weights * self.events_by_day).sum() / (weights * self.flights_by_day).sum()
            lift[b] = pr[b] / prevalence if prevalence > 0 else np.nan
        return pr, lift

    def _interval(self, values) -> tuple[float, float]:
        tail = (1 - self.confidence) / 2
        return float(np.nanquantile(values, tail)), float(np.nanquantile(values, 1 - tail))

    def summarise(self, scores: dict[str, np.ndarray], reference: str | None = None) -> pd.DataFrame:
        """One row per scorer: PR-AUC, lift and their intervals; with ``reference`` also the paired PR-AUC difference
        (scorer minus reference), its interval and the share of resamples in which the scorer is better."""
        prevalence = float(self.y.mean())
        cached = {name: self.draws(s) for name, s in scores.items()}
        rows = []
        for name, s in scores.items():
            point = self.point(s)
            pr, lift = cached[name]
            low, high = self._interval(pr)
            lift_low, lift_high = self._interval(lift)
            row = {"scorer": name, "pr_auc": point, "pr_auc_low": low, "pr_auc_high": high, "lift": point / prevalence,
                   "lift_low": lift_low, "lift_high": lift_high}
            if reference is not None and name != reference:
                difference = pr - cached[reference][0]
                d_low, d_high = self._interval(difference)
                row.update({"difference_vs_reference": point - self.point(scores[reference]), "difference_low": d_low,
                            "difference_high": d_high, "share_better": float(np.nanmean(difference > 0))})
            rows.append(row)
        return pd.DataFrame(rows)

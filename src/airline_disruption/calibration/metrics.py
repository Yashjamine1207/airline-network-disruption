"""Calibration metrics and the paired day-level bootstrap for the Brier score.

Phase 7A, Step 9. Definitions used everywhere in the calibration report:

brier              mean((p - y)^2). Lower is better.
brier_skill        1 - brier / brier_of_a_constant. The constant is the event rate of the rows the
                   calibrator was fitted on: what someone could have known before the evaluation
                   period. 0 means "no better than quoting that rate", 1 is perfect. It can be
                   negative.
calibration_slope  b in P(y=1) = expit(a + b * logit(p)). 1 is ideal. Below 1: probabilities are too
                   spread out (over-confident). Above 1: too compressed.
calibration_in_the_large
                   a in P(y=1) = expit(a + logit(p)), slope fixed at 1. 0 is ideal. Positive: the
                   model gives too low a probability on average. Negative: too high.
ece                expected calibration error: the row-weighted mean of |mean predicted - observed
                   rate| over equal-count bins (10 by default). Equal-count bins are used because
                   the scores are heavily skewed (most flights are far below 5%).
mean_predicted / prevalence
                   The average probability against the observed event rate. Their ratio is the
                   simplest summary of calibration-in-the-large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from airline_disruption.calibration.methods import check_labels, check_scores, fit_logistic, to_logit


def _pair(y, p) -> tuple[np.ndarray, np.ndarray]:
    s = check_scores(p, "probabilities")
    return check_labels(y, len(s)), s


def brier(y, p) -> float:
    y_, p_ = _pair(y, p)
    return float(brier_score_loss(y_, p_))


def brier_skill(y, p, reference_rate: float) -> float:
    """1 - Brier(p) / Brier(constant ``reference_rate``), evaluated on the same rows."""
    if not 0.0 < reference_rate < 1.0:
        raise ValueError("reference_rate must be between 0 and 1")
    y_, p_ = _pair(y, p)
    baseline = float(np.mean((reference_rate - y_) ** 2))
    return 1.0 - float(np.mean((p_ - y_) ** 2)) / baseline


def calibration_slope_and_intercept(y, p) -> tuple[float, float]:
    """``(slope, in_the_large)``: the slope of a logistic recalibration and the level shift at slope 1."""
    y_, p_ = _pair(y, p)
    z = to_logit(p_)
    _, slope = fit_logistic(z, y_, slope=True)
    in_the_large, _ = fit_logistic(z, y_, slope=False)
    return slope, in_the_large


def equal_count_bins(p, n_bins: int) -> np.ndarray:
    """Bin numbers 0..n_bins-1 with (nearly) equal row counts. Ties are split by row order, so it is deterministic."""
    p = np.asarray(p, dtype="float64")
    if n_bins < 1 or n_bins > len(p):
        raise ValueError("n_bins must be between 1 and the number of rows")
    order = np.argsort(p, kind="stable")
    bins = np.empty(len(p), dtype="int64")
    for number, chunk in enumerate(np.array_split(order, n_bins)):
        bins[chunk] = number
    return bins


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against observed event rate in equal-count bins."""
    y_, p_ = _pair(y, p)
    frame = pd.DataFrame({"bin": equal_count_bins(p_, n_bins), "y": y_, "p": p_})
    table = (
        frame.groupby("bin")
        .agg(rows=("y", "size"), events=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(int) + 1
    return table


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    table = reliability_table(y, p, n_bins)
    weights = table["rows"] / table["rows"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def evaluate_probabilities(y, p, reference_rate: float, n_bins: int = 10) -> dict[str, float]:
    """Every calibration number for one set of probabilities on one period."""
    y_, p_ = _pair(y, p)
    slope, in_the_large = calibration_slope_and_intercept(y_, p_)
    prevalence = float(y_.mean())
    mean_predicted = float(p_.mean())
    return {
        "rows": int(len(y_)),
        "events": int(y_.sum()),
        "prevalence": prevalence,
        "mean_predicted": mean_predicted,
        "predicted_to_observed": mean_predicted / prevalence,
        "brier": brier(y_, p_),
        "brier_skill": brier_skill(y_, p_, reference_rate),
        "calibration_slope": slope,
        "calibration_in_the_large": in_the_large,
        "ece": expected_calibration_error(y_, p_, n_bins),
        "pr_auc": float(average_precision_score(y_, p_)),
        "roc_auc": float(roc_auc_score(y_, p_)),
        "distinct_scores": int(len(np.unique(p_))),
    }


def paired_day_bootstrap_brier(
    y,
    p_reference,
    p_challenger,
    day_ids,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Uncertainty for ``Brier(challenger) - Brier(reference)``. A NEGATIVE difference favours the challenger.

    Whole days are resampled with replacement (flights on one day share the same operating
    conditions, so a per-flight bootstrap would be too optimistic). Both probability sets are
    scored on the same resamples. Per-day sums make each resample cheap.
    """
    y_, reference = _pair(y, p_reference)
    _, challenger = _pair(y, p_challenger)
    codes, _ = pd.factorize(pd.Series(np.asarray(day_ids)))
    if len(codes) != len(y_):
        raise ValueError("day_ids must have one entry per row")
    n_days = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=n_days).astype("float64")
    error_reference = np.bincount(codes, weights=(reference - y_) ** 2, minlength=n_days)
    error_challenger = np.bincount(codes, weights=(challenger - y_) ** 2, minlength=n_days)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_days, size=(n_boot, n_days))
    total_rows = counts[draws].sum(axis=1)
    differences = (error_challenger[draws].sum(axis=1) - error_reference[draws].sum(axis=1)) / total_rows

    tail = (1.0 - confidence) / 2.0
    brier_reference = float(error_reference.sum() / counts.sum())
    brier_challenger = float(error_challenger.sum() / counts.sum())
    return {
        "brier_reference": brier_reference,
        "brier_challenger": brier_challenger,
        "difference": brier_challenger - brier_reference,
        "ci_low": float(np.quantile(differences, tail)),
        "ci_high": float(np.quantile(differences, 1.0 - tail)),
        "share_challenger_better": float((differences < 0).mean()),
        "n_boot": int(n_boot),
        "n_days": n_days,
    }

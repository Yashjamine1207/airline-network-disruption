"""Classification metrics for rare-event prediction.

Phase 6, Step 4. Every Phase 6 model (LightGBM, LSTM, GRU, Transformer) is scored
with these functions, so the numbers are comparable.

Headline metric: PR-AUC (average precision). Accuracy is never reported.
PR-AUC of a model with no skill equals the event prevalence, and prevalence
differs by split (severe delay is 2.1% in development, 2.7% in 2022, 3.2% in
2023), so ``lift`` (PR-AUC divided by prevalence) is reported next to it.

Functions
---------
binary_metrics       PR-AUC, lift, ROC-AUC, Brier score, prevalence.
ranking_table        Precision, recall, F1 and counts when the top K% of scores is reviewed.
calibration_table    Reliability table in equal-count bins.
paired_day_bootstrap Uncertainty for the PR-AUC difference between two models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Review capacities from the Phase 5-7B scope: top 0.5%, 1%, 5% and 10% of flights.
DEFAULT_FRACTIONS = (0.005, 0.01, 0.05, 0.10)


def _validate(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Return labels as int8 and scores as float64, or raise ValueError."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype="float64")
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("y_true and scores must be one-dimensional and the same length")
    if len(y) == 0:
        raise ValueError("No rows to evaluate")
    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN or infinite values")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be 0 or 1")
    y = y.astype("int8")
    if y.min() == y.max():
        raise ValueError("Only one class present, so PR-AUC and ROC-AUC are undefined")
    return y, s


def binary_metrics(y_true, scores, probabilities: bool = True) -> dict[str, float]:
    """Headline metrics for one model on one evaluation set.

    Set ``probabilities=False`` for scores that are not probabilities (the Brier
    score and mean score are then reported as NaN).
    """
    y, s = _validate(y_true, scores)
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, s))
    return {
        "rows": int(len(y)),
        "events": int(y.sum()),
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence,
        "roc_auc": float(roc_auc_score(y, s)),
        "brier_score": float(brier_score_loss(y, s)) if probabilities else float("nan"),
        "mean_score": float(s.mean()) if probabilities else float("nan"),
    }


def ranking_table(y_true, scores, fractions=DEFAULT_FRACTIONS) -> pd.DataFrame:
    """Results when the top ``fraction`` of flights by score is flagged for review.

    Ties at the cut-off score are handled by expectation: if the cut-off falls
    inside a group of equal scores, that group contributes its events in
    proportion to how many of its members fit in the top K. This makes the
    result independent of row order, which matters for coarse baselines such as
    a carrier-rate lookup that has only 18 distinct scores.

    Columns: fraction, reviewed, events_captured, false_alerts, precision,
    recall, f1, lift (precision divided by prevalence).
    """
    y, s = _validate(y_true, scores)
    n = len(y)
    total_events = int(y.sum())
    prevalence = total_events / n

    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    cumulative_events = np.concatenate([[0], np.cumsum(y[order])])
    ascending = -sorted_scores  # ascending, so searchsorted works

    rows = []
    for fraction in fractions:
        k = min(max(int(round(fraction * n)), 1), n)
        cutoff = sorted_scores[k - 1]
        n_above = int(np.searchsorted(ascending, -cutoff, side="left"))  # scores strictly above the cut-off
        n_through_ties = int(np.searchsorted(ascending, -cutoff, side="right"))  # scores at or above it
        n_tied = n_through_ties - n_above
        events_above = cumulative_events[n_above]
        events_tied = cumulative_events[n_through_ties] - events_above
        captured = float(events_above + events_tied * (k - n_above) / n_tied)

        precision = captured / k
        recall = captured / total_events
        f1 = 2 * precision * recall / (precision + recall) if captured > 0 else 0.0
        rows.append(
            {
                "fraction": fraction,
                "reviewed": k,
                "events_captured": captured,
                "false_alerts": k - captured,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "lift": precision / prevalence,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: mean predicted probability against observed event rate.

    Bins have equal row counts (ranked by score, ties broken by row order), so
    every bin has enough rows even when scores are heavily skewed.
    """
    y, p = _validate(y_true, probabilities)
    ranks = pd.Series(p).rank(method="first")
    bins = pd.qcut(ranks, n_bins, labels=False)
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p})
    table = (
        frame.groupby("bin")
        .agg(rows=("y", "size"), events=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(int) + 1
    return table


def paired_day_bootstrap(
    y_true,
    scores_reference,
    scores_challenger,
    day_ids,
    n_boot: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Uncertainty for ``PR-AUC(challenger) - PR-AUC(reference)``.

    Whole days are resampled with replacement, not single flights, because
    flights on the same day share weather, disruption and airport state. A
    per-flight bootstrap would make the intervals too narrow. Both models are
    scored on the same resamples, so the difference is paired.

    Returns the full-sample values, the percentile interval of the difference,
    and the share of resamples in which the challenger is better.
    """
    y, reference = _validate(y_true, scores_reference)
    _, challenger = _validate(y_true, scores_challenger)
    codes, _ = pd.factorize(pd.Series(np.asarray(day_ids)))
    if len(codes) != len(y):
        raise ValueError("day_ids must have one entry per row")

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    n_days = int(sorted_codes.max()) + 1
    starts = np.searchsorted(sorted_codes, np.arange(n_days), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(n_days), side="right")

    rng = np.random.default_rng(seed)
    differences = np.full(n_boot, np.nan)
    for i in range(n_boot):
        chosen = rng.integers(0, n_days, n_days)
        index = np.concatenate([order[starts[d] : ends[d]] for d in chosen])
        y_sample = y[index]
        if y_sample.min() == y_sample.max():
            continue  # cannot happen with real data; skipped rather than crashing
        differences[i] = average_precision_score(y_sample, challenger[index]) - average_precision_score(
            y_sample, reference[index]
        )

    valid = differences[~np.isnan(differences)]
    tail = (1.0 - confidence) / 2.0
    point_reference = float(average_precision_score(y, reference))
    point_challenger = float(average_precision_score(y, challenger))
    return {
        "pr_auc_reference": point_reference,
        "pr_auc_challenger": point_challenger,
        "difference": point_challenger - point_reference,
        "ci_low": float(np.quantile(valid, tail)),
        "ci_high": float(np.quantile(valid, 1.0 - tail)),
        "share_challenger_better": float((valid > 0).mean()),
        "n_boot": int(len(valid)),
        "n_days": n_days,
    }

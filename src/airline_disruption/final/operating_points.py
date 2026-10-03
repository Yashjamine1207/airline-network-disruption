"""Operating points: score cutoffs learned on 2022 and confusion matrices on any period.

Phase 7B, Step 12. A cutoff is a raw LightGBM score. The calibrators are strictly increasing, so a cutoff in
score space is the same flights as a cutoff in probability space; the probability that a cutoff corresponds to
is shown for reading only.

Two kinds of cutoff, both found on the validation year and then applied unchanged to the final test:
* capacity cutoffs: the lowest score inside the top 0.5%, 1%, 5% and 10% of validation flights.
* the F1-optimal cutoff: the score that maximises F1 on the validation year. For a weak ranking this can flag a
  large share of all flights, so its alert rate is always printed next to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from airline_disruption.explain.errors import top_fraction_cutoffs
from airline_disruption.policy.review import capacity_name


def f1_optimal_cutoff(y, score) -> tuple[float, float]:
    """``(cutoff, f1)``: the score at which flagging ``score >= cutoff`` gives the highest F1."""
    y = np.asarray(y)
    s = np.asarray(score, dtype="float64")
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]  # the last point (recall 0) has no threshold
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def learn_cutoffs(y, score, fractions) -> dict[str, dict]:
    """Operating points from one period (the validation year). Keys: capacity_0.5pct, ..., f1_optimal."""
    cutoffs = top_fraction_cutoffs(score, tuple(fractions))
    points = {f"capacity_{fraction * 100:g}pct": {"kind": "capacity", "fraction": fraction, "cutoff_score": cutoff}
              for fraction, cutoff in cutoffs.items()}
    cutoff, f1 = f1_optimal_cutoff(y, score)
    points["f1_optimal"] = {"kind": "f1_optimal", "fraction": None, "cutoff_score": cutoff, "validation_f1": f1}
    return points


def confusion_at(y, score, cutoff: float) -> dict:
    """Flag every flight with ``score >= cutoff`` and count."""
    y = np.asarray(y)
    flagged = np.asarray(score, dtype="float64") >= cutoff
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"flights": int(len(y)), "events": tp + fn, "flagged": tp + fp, "alert_rate": (tp + fp) / len(y),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
            "precision": precision, "recall": recall, "f1": f1, "specificity": tn / (tn + fp) if tn + fp else np.nan}


def operating_point_table(y, score, points: dict[str, dict], period: str, probability_map=None) -> pd.DataFrame:
    """One row per operating point with its confusion matrix on ``period``.

    ``probability_map`` is an optional function from score to calibrated probability, used only to show which
    probability each cutoff corresponds to.
    """
    rows = []
    for name, point in points.items():
        cutoff = point["cutoff_score"]
        row = {"period": period, "operating_point": name, "cutoff_score": cutoff, **confusion_at(y, score, cutoff)}
        if probability_map is not None:
            row["cutoff_probability"] = float(probability_map(np.array([cutoff]))[0])
        rows.append(row)
    return pd.DataFrame(rows)

"""Error breakdown tables for a probabilistic classifier (Phase 7A, Step 10).

For every group of flights (a carrier, a quarter, a route-volume band, ...) the table shows how often the
event happens, how large the model's probabilities are, and what a review of the top 1% and top 10% of
ALL flights (one global cutoff each, not a cutoff per group) would catch inside the group.

A global cutoff is the honest choice: an analyst ranks every flight together. A group whose flights all
score low is then shown as it is, with a low alert rate and low recall, instead of being rescued by its own
cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

MIN_EVENTS_FOR_PR_AUC = 30


def top_fraction_cutoffs(score, fractions=(0.01, 0.10)) -> dict[float, float]:
    """The lowest score inside the top ``fraction`` of all flights (ties at the cutoff are all included)."""
    s = np.asarray(score, dtype="float64")
    n = len(s)
    ordered = np.sort(s)[::-1]
    return {f: float(ordered[min(max(int(round(f * n)), 1), n) - 1]) for f in fractions}


def group_metrics(
    frame: pd.DataFrame,
    dimension: str,
    cutoffs: dict[float, float],
    group_column: str,
    score_column: str = "score",
    probability_column: str = "probability",
    label_column: str = "label",
    min_events: int = MIN_EVENTS_FOR_PR_AUC,
) -> pd.DataFrame:
    """One row per value of ``group_column``. See the module docstring for the alert columns.

    ``pr_auc`` and ``lift`` are left empty for a group with fewer than ``min_events`` events or fewer than
    ``min_events`` non-events, where they would be noise. ``small_sample`` marks those rows.
    """
    total_rows = len(frame)
    rows = []
    for value, group in frame.groupby(group_column, observed=True, dropna=False, sort=True):
        y = group[label_column].to_numpy()
        s = group[score_column].to_numpy()
        p = group[probability_column].to_numpy()
        events = int(y.sum())
        record = {
            "dimension": dimension, "group": "unseen" if pd.isna(value) else str(value), "rows": int(len(group)),
            "share_of_rows": len(group) / total_rows, "events": events, "prevalence": events / len(group),
            "mean_score": float(s.mean()), "mean_probability": float(p.mean()),
            "probability_to_observed": float(p.mean() / (events / len(group))) if events else np.nan,
            "brier_probability": float(np.mean((p - y) ** 2)),
        }
        for fraction, cutoff in cutoffs.items():
            label = f"top{fraction * 100:g}pct"  # 0.01 gives top1pct, 0.005 gives top0.5pct
            flagged = s >= cutoff
            captured = int((flagged & (y == 1)).sum())
            record[f"alert_rate_{label}"] = float(flagged.mean())
            record[f"precision_{label}"] = captured / flagged.sum() if flagged.any() else np.nan
            record[f"recall_{label}"] = captured / events if events else np.nan
        enough = events >= min_events and (len(group) - events) >= min_events
        record["pr_auc"] = float(average_precision_score(y, s)) if enough else np.nan
        record["lift"] = record["pr_auc"] / record["prevalence"] if enough else np.nan
        record["small_sample"] = not enough
        rows.append(record)
    return pd.DataFrame(rows)


def severity_table(
    frame: pd.DataFrame,
    cutoffs: dict[float, float],
    delay_column: str = "arrival_delay_minutes",
    score_column: str = "score",
    label_column: str = "label",
    edges: tuple[float, ...] = (120, 180, 300, 600, np.inf),
) -> pd.DataFrame:
    """Among EVENTS only: what share of each delay-severity band would a global top-K% review catch.

    Uses the realised arrival delay, so this is a retrospective (post-event) description of who the model
    misses, not a prediction input.
    """
    events = frame[frame[label_column] == 1].copy()
    events["_band"] = pd.cut(events[delay_column], bins=list(edges), right=False)
    rows = []
    for band, group in events.groupby("_band", observed=True):
        record = {"dimension": "event_delay_band_minutes", "group": str(band), "events": int(len(group)),
                  "share_of_events": len(group) / len(events), "small_sample": len(group) < MIN_EVENTS_FOR_PR_AUC}
        for fraction, cutoff in cutoffs.items():
            record[f"recall_top{fraction * 100:g}pct"] = float((group[score_column] >= cutoff).mean())
        rows.append(record)
    return pd.DataFrame(rows)

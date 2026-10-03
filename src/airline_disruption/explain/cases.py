"""Choosing which flights to show as case studies, by rule and not by eye.

Phase 7A, Step 10. The scope asks for true positives, false positives, false negatives, high-confidence
errors and low-risk correct cases, and says the purpose is to expose limits, not to show the model at its
best. Picking cases by hand invites showing the good ones. These functions pick them by a fixed rule:

* the rule uses only the score and the label of each row, never a feature, a carrier or an airport;
* typical cases are taken at fixed quantiles of the score inside each category, so the table shows the
  middle of each group and not only its extremes;
* the extreme cases (the most confident mistakes) are a separate, named category.

Definitions (rank 1 is the highest score; ``n`` is the number of rows):

alert                 rank <= alert_fraction * n   (default 1%, a top-1% review capacity)
true_positive         alert and event
false_positive        alert and no event
false_negative        event and NOT inside the top ``missed_fraction`` (default 10%): missed even at a large capacity
low_risk_correct      no event and in the lower half of the scores
high_confidence_false_positive   the ``extremes`` highest-scoring flights that were not events
high_confidence_false_negative   the ``extremes`` lowest-scoring flights that were events
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = (
    "true_positive",
    "false_positive",
    "false_negative",
    "low_risk_correct",
    "high_confidence_false_positive",
    "high_confidence_false_negative",
)


def score_ranks(score) -> np.ndarray:
    """Rank 1 for the highest score. Ties are broken by row position, so the result is deterministic."""
    s = np.asarray(score, dtype="float64")
    order = np.argsort(-s, kind="stable")
    rank = np.empty(len(s), dtype="int64")
    rank[order] = np.arange(1, len(s) + 1)
    return rank


def select_cases(
    y,
    score,
    alert_fraction: float = 0.01,
    missed_fraction: float = 0.10,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    extremes: int = 3,
) -> pd.DataFrame:
    """Rows to explain. Columns: position, category, pick, rank, score, label.

    ``position`` indexes the arrays passed in. A row is chosen at most once. Extreme cases are chosen
    first, then the typical cases, which skip rows already taken. A category with fewer rows than picks
    simply returns fewer rows.
    """
    y = np.asarray(y).astype("int64")
    s = np.asarray(score, dtype="float64")
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("y and score must be non-empty one-dimensional arrays of the same length")
    if not np.isfinite(s).all():
        raise ValueError("score contains NaN or infinite values")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0 or 1")
    if not 0.0 < alert_fraction < missed_fraction <= 1.0:
        raise ValueError("Need 0 < alert_fraction < missed_fraction <= 1")
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("quantiles must be between 0 and 1")
    if extremes < 0:
        raise ValueError("extremes must not be negative")

    n = len(y)
    rank = score_ranks(s)
    alert_k = max(int(round(alert_fraction * n)), 1)
    review_k = max(int(round(missed_fraction * n)), 1)
    alert = rank <= alert_k
    event = y == 1

    pools = {
        "true_positive": np.flatnonzero(alert & event),
        "false_positive": np.flatnonzero(alert & ~event),
        "false_negative": np.flatnonzero((rank > review_k) & event),
        "low_risk_correct": np.flatnonzero((rank > n / 2.0) & ~event),
    }
    taken: set[int] = set()
    rows: list[dict] = []

    def add(position: int, category: str, pick: str) -> None:
        taken.add(int(position))
        rows.append({"position": int(position), "category": category, "pick": pick, "rank": int(rank[position]),
                     "score": float(s[position]), "label": int(y[position])})

    non_events = np.flatnonzero(~event)
    if extremes and len(non_events):
        for position in non_events[np.argsort(rank[non_events], kind="stable")][:extremes]:
            add(position, "high_confidence_false_positive", "highest_scoring_non_event")
    events = np.flatnonzero(event)
    if extremes and len(events):
        for position in events[np.argsort(-rank[events], kind="stable")][:extremes]:
            add(position, "high_confidence_false_negative", "lowest_scoring_event")

    for category, pool in pools.items():
        if len(pool) == 0:
            continue
        ordered = pool[np.argsort(rank[pool], kind="stable")]  # best score first
        for q in quantiles:
            start = int(round(q * (len(ordered) - 1)))
            for step in range(len(ordered)):  # move on if that row is already chosen
                candidate = int(ordered[(start + step) % len(ordered)])
                if candidate not in taken:
                    add(candidate, category, f"typical_q{int(round(q * 100)):02d}")
                    break
    table = pd.DataFrame(rows, columns=["position", "category", "pick", "rank", "score", "label"])
    order = {c: i for i, c in enumerate(CATEGORIES)}
    table["_o"] = table["category"].map(order)
    return table.sort_values(["_o", "rank"], kind="stable").drop(columns="_o").reset_index(drop=True)

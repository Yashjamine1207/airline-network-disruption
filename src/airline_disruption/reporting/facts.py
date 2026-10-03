"""Facts computed from result tables, so the narrative never repeats a number by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd


def with_alert_shares(groups: pd.DataFrame, alert_column: str = "alert_rate_top1pct") -> pd.DataFrame:
    """Add ``alerts`` (rows times alert rate), ``share_of_alerts`` and ``share_of_events`` to a by-group table."""
    out = groups.copy()
    out["alerts"] = out["rows"] * out[alert_column]
    total = out["alerts"].sum()
    out["share_of_alerts"] = out["alerts"] / total if total > 0 else np.nan
    out["share_of_events"] = out["events"] / out["events"].sum() if out["events"].sum() > 0 else np.nan
    out["share_of_rows"] = out["rows"] / out["rows"].sum()
    return out


def concentration(groups: pd.DataFrame, k: int, alert_column: str = "alert_rate_top1pct", min_rows: int = 1000) -> dict:
    """The ``k`` groups with the most alerts, and the share of alerts and of flights they hold.

    Groups with fewer than ``min_rows`` rows are left out, so a stray day cannot be named as a pattern.
    """
    shaped = with_alert_shares(groups[groups["rows"] >= min_rows], alert_column)
    if shaped.empty:
        return {"groups": [], "share_of_alerts": np.nan, "share_of_rows": np.nan, "share_of_events": np.nan}
    top = shaped.sort_values("alerts", ascending=False).head(k)
    return {"groups": top["group"].tolist(), "share_of_alerts": float(top["share_of_alerts"].sum()),
            "share_of_rows": float(top["share_of_rows"].sum()), "share_of_events": float(top["share_of_events"].sum())}


def groups_without_alerts(groups: pd.DataFrame, alert_column: str = "alert_rate_top1pct", min_rows: int = 1000) -> pd.DataFrame:
    """Groups of at least ``min_rows`` flights where the review flagged nothing (alert rate rounds to zero flights)."""
    big = groups[groups["rows"] >= min_rows]
    return big[(big["rows"] * big[alert_column]) < 0.5]


def is_increasing(values) -> bool:
    """True if the values never fall from one position to the next."""
    v = np.asarray(list(values), dtype="float64")
    return bool(len(v) > 1 and np.all(np.diff(v) >= 0))


def pick(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    """Rows where every named column equals the given value."""
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    return frame[mask]


def one(frame: pd.DataFrame, **conditions) -> pd.Series:
    """The single row matching the conditions. Raises if there are none or several."""
    rows = pick(frame, **conditions)
    if len(rows) != 1:
        raise KeyError(f"expected exactly one row for {conditions}, found {len(rows)}")
    return rows.iloc[0]


CLAIM_LABELS = {
    "lift": "PR-AUC lift", "lift_low": "lower end of the lift interval",
    "difference": "PR-AUC difference against the history baseline", "difference_low": "interval low", "difference_high": "interval high",
    "by_day_precision_at_10pct": "precision at 10% inside each day", "by_day_precision_at_10pct_low": "lower end of that interval",
    "event_rate_2023": "2023 event rate",
}


def claim_lines(claims: dict) -> list[str]:
    """One plain line per pre-registered claim: name, outcome, its key numbers, and the rule it was judged by."""
    lines = []
    for name, claim in claims.items():
        outcome = "supported" if claim.get("supported") else "not supported"
        numbers = {k: v for k, v in claim.items() if k not in ("statement", "supported", "baseline") and isinstance(v, (int, float))}
        detail = "; ".join(f"{CLAIM_LABELS.get(k, k)} {v:.4f}" for k, v in numbers.items())
        baseline = f" Baseline: {claim['baseline']}." if claim.get("baseline") else ""
        lines.append(f"**{name}**: {outcome}. {detail}. Rule: {claim.get('statement', '')}{baseline}")
    return lines

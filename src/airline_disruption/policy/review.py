"""Ranked analyst-review scenarios and an illustrative penalty table (Phase 7B).

The scenario is retrospective: if an analyst could look at the K% of flights with the highest model score,
how many severe events would be among them? It is a way to read a ranking, not an operating procedure.

Three ways to apply a capacity K:

whole_year   one ranking over every flight in the evaluation year; the top K% are reviewed. Seasonal level
             differences count: a summer flight can outrank a winter one.
by_month     the top K% of each calendar month are reviewed. A fixed monthly capacity.
by_day       the top K% of each UTC prediction day are reviewed. A fixed daily capacity, the nearest
             thing to how a review desk would work. Seasonal and weekday level differences cannot help.

Rules, all tested:
* The number reviewed in a group of ``n`` flights is ``ceil(K * n)``, at least 1 (a small float error is
  tolerated, so 5% of 60 flights is 3, not 4). The reviewed total is therefore at least K% of the flights.
* Ties are broken by row order, so the same input always gives the same flagged set.
* Rankings use whatever score is passed in. The Phase 7B script passes the UNCALIBRATED score (Platt is
  monotone and gives the same ranking). No probability threshold is used here.

The penalty table is illustrative. A missed event costs ``w_miss`` and a false alert costs ``w_false``; the
weights are assumptions, never airline costs. The measured quantities (events caught, false alerts) stay in
their own columns, and the break-even ratio is computed from them with no assumption at all.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_CAPACITIES = (0.005, 0.01, 0.05, 0.10)
DEFAULT_RATIOS = (1, 2, 5, 10, 20, 50, 100)
SCHEMES = ("whole_year", "by_month", "by_day")


def capacity_name(fraction: float) -> str:
    """0.005 gives '0.5%', 0.1 gives '10%'."""
    return f"{fraction * 100:g}%"


def review_count(n: int, fraction: float) -> int:
    """Flights reviewed out of ``n``: ceil(fraction * n), at least 1, at most n. Tolerates float error."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if n < 1:
        raise ValueError("n must be at least 1")
    return int(min(n, max(1, math.ceil(fraction * n - 1e-9))))


def review_flags(score, fraction: float, groups=None) -> np.ndarray:
    """Boolean array in the input order: True for the flights an analyst would review.

    With ``groups`` the capacity applies inside every group separately (for example every day).
    """
    s = np.asarray(score, dtype="float64")
    n = len(s)
    if n == 0:
        raise ValueError("no rows")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if groups is None:
        codes = np.zeros(n, dtype="int64")
    else:
        codes, _ = pd.factorize(pd.Series(np.asarray(groups)), use_na_sentinel=False)
        if len(codes) != n:
            raise ValueError("groups must have one entry per row")
    # Sort by group, then by score (highest first), then by row order for ties.
    order = np.lexsort((np.arange(n), -s, codes))
    sorted_codes = codes[order]
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rank_in_group = np.arange(n) - starts[sorted_codes]
    allowed = np.array([review_count(int(size), fraction) for size in sizes])
    flagged_sorted = rank_in_group < allowed[sorted_codes]
    flags = np.zeros(n, dtype=bool)
    flags[order[flagged_sorted]] = True
    return flags


def review_row(y, flags) -> dict:
    """Counts and rates for one set of flagged flights."""
    y = np.asarray(y)
    flags = np.asarray(flags, dtype=bool)
    if len(y) != len(flags):
        raise ValueError("y and flags must have the same length")
    total, events = len(y), int(y.sum())
    reviewed = int(flags.sum())
    captured = int((flags & (y == 1)).sum())
    precision = captured / reviewed if reviewed else np.nan
    prevalence = events / total
    return {
        "flights": total, "events": events, "prevalence": prevalence, "reviewed": reviewed,
        "events_captured": captured, "false_alerts": reviewed - captured, "events_missed": events - captured,
        "precision": precision, "recall": captured / events if events else np.nan,
        "lift": precision / prevalence if events and reviewed else np.nan,
        "break_even_ratio": (reviewed - captured) / captured if captured else np.inf,
    }


def bootstrap_day_intervals(y, flags, day_codes, n_boot: int = 1000, seed: int = 42, confidence: float = 0.95) -> dict:
    """Percentile intervals for precision and recall when whole days are resampled with replacement.

    The flagged set is fixed (it comes from the ranking); only which days count is resampled. Flights on one
    day share their conditions, so resampling flights one by one would make the intervals too narrow.
    """
    y = np.asarray(y)
    flags = np.asarray(flags, dtype=bool)
    codes, _ = pd.factorize(pd.Series(np.asarray(day_codes)))
    n_days = int(codes.max()) + 1
    events = np.bincount(codes, weights=y, minlength=n_days)
    reviewed = np.bincount(codes, weights=flags.astype(float), minlength=n_days)
    captured = np.bincount(codes, weights=(flags & (y == 1)).astype(float), minlength=n_days)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_days, size=(n_boot, n_days))
    rev, cap, ev = reviewed[draws].sum(axis=1), captured[draws].sum(axis=1), events[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision, recall = cap / rev, cap / ev
    tail = (1 - confidence) / 2
    return {
        "precision_low": float(np.nanquantile(precision, tail)), "precision_high": float(np.nanquantile(precision, 1 - tail)),
        "recall_low": float(np.nanquantile(recall, tail)), "recall_high": float(np.nanquantile(recall, 1 - tail)),
        "n_days": n_days, "n_boot": int(n_boot),
    }


def top_k_review(y, score, fractions=DEFAULT_CAPACITIES, month_codes=None, day_codes=None, n_boot: int = 0, seed: int = 42) -> pd.DataFrame:
    """One row per scheme and capacity. ``by_month`` and ``by_day`` appear only when their group codes are given.

    With ``n_boot > 0`` and ``day_codes`` the precision and recall get day-level bootstrap intervals.
    """
    y = np.asarray(y)
    schemes = {"whole_year": None}
    if month_codes is not None:
        schemes["by_month"] = month_codes
    if day_codes is not None:
        schemes["by_day"] = day_codes
    rows = []
    for scheme, groups in schemes.items():
        for fraction in fractions:
            flags = review_flags(score, fraction, groups)
            row = {"scheme": scheme, "capacity": capacity_name(fraction), "fraction": fraction, **review_row(y, flags)}
            if n_boot > 0 and day_codes is not None:
                row.update(bootstrap_day_intervals(y, flags, day_codes, n_boot, seed))
            rows.append(row)
    return pd.DataFrame(rows)


def utility_table(capacity_rows: pd.DataFrame, ratios=DEFAULT_RATIOS, false_alert_weight: float = 1.0) -> pd.DataFrame:
    """ILLUSTRATIVE penalty reduction for every scheme, capacity and assumed miss-to-false-alert ratio.

    penalty(review) = w_miss * missed + w_false * false alerts;  penalty(no review) = w_miss * all events.
    net_reduction   = penalty(no review) - penalty(review) = w_miss * captured - w_false * false alerts.
    random_reduction is the same quantity for a review of the same size that picks flights at random
    (captured = reviewed * prevalence). advantage_over_random = net_reduction - random_reduction.
    One unit is the penalty of one false alert. None of this is a measured airline cost.
    """
    if false_alert_weight <= 0 or any(r <= 0 for r in ratios):
        raise ValueError("weights and ratios must be positive")
    rows = []
    for record in capacity_rows.itertuples():
        random_captured = record.reviewed * record.prevalence
        for ratio in ratios:
            w_miss = ratio * false_alert_weight
            net = w_miss * record.events_captured - false_alert_weight * record.false_alerts
            random_net = w_miss * random_captured - false_alert_weight * (record.reviewed - random_captured)
            rows.append({
                "scheme": record.scheme, "capacity": record.capacity, "fraction": record.fraction,
                "miss_to_false_alert_ratio": ratio, "net_reduction_units": net, "random_review_reduction_units": random_net,
                "advantage_over_random_units": net - random_net, "reviewed": record.reviewed,
                "events_captured": record.events_captured, "false_alerts": record.false_alerts,
                "break_even_ratio": record.break_even_ratio, "pays_off_at_this_ratio": bool(net > 0),
            })
    return pd.DataFrame(rows)

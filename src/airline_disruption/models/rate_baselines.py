"""Transparent historical-rate baselines.

Phase 6, Step 4.

Each baseline scores a flight with the smoothed severe-delay rate of its carrier,
airport or route, measured on the FIT rows only (development, before the
early-stopping period). Rates are fixed lookup tables applied to later periods,
so nothing from the evaluation period enters them.

These are baselines, not features. The Phase 3 decision blocks historical
outcome rates as pre-flight predictors because the source does not say when an
earlier flight's outcome became known. A table built from 2019 to mid-2021 and
applied to 2022 uses only outcomes that were long settled by 2022, so it is a
legitimate comparator. A model that cannot beat "the route's average history"
has learned little.

Smoothing: ``(events + strength * overall_rate) / (flights + strength)``.
``strength`` is fixed in advance (50), not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_STRENGTH = 50.0

# baseline name -> column used as the key
BASELINE_KEYS = {
    "baseline_carrier_rate": "carrier_identifier",
    "baseline_origin_rate": "origin_airport",
    "baseline_destination_rate": "destination_airport",
    "baseline_route_rate": "route",
}


@dataclass(frozen=True)
class RateTable:
    """Smoothed event rate per key, plus the overall rate used for unseen keys."""

    rates: pd.Series  # index: key as string
    overall_rate: float


def fit_rate_table(keys: pd.Series, y: np.ndarray, strength: float = DEFAULT_STRENGTH) -> RateTable:
    """Learn a smoothed rate for each key."""
    y = np.asarray(y, dtype="float64")
    if len(keys) != len(y):
        raise ValueError("keys and y must be the same length")
    overall = float(y.mean())
    grouped = pd.DataFrame({"key": keys.astype(str).to_numpy(), "y": y}).groupby("key")["y"].agg(["sum", "count"])
    rates = (grouped["sum"] + strength * overall) / (grouped["count"] + strength)
    return RateTable(rates=rates, overall_rate=overall)


def apply_rate_table(table: RateTable, keys: pd.Series, fallback: np.ndarray | None = None) -> np.ndarray:
    """Score rows. A key not in the table gets ``fallback`` if given, else the overall rate."""
    # copy=True: pandas 3 hands back a read-only view, and we overwrite unseen keys below.
    mapped = keys.astype(str).map(table.rates).to_numpy(dtype="float64", copy=True)
    unseen = np.isnan(mapped)
    if unseen.any():
        mapped[unseen] = table.overall_rate if fallback is None else np.asarray(fallback, dtype="float64")[unseen]
    return mapped


def build_rate_baselines(
    fit_frame: pd.DataFrame,
    y_fit: np.ndarray,
    apply_frame: pd.DataFrame,
    strength: float = DEFAULT_STRENGTH,
) -> dict[str, np.ndarray]:
    """Scores for every baseline on ``apply_frame``, learned from ``fit_frame`` only.

    Returns ``baseline_prevalence`` (a constant), then carrier, origin,
    destination and route rates. A route never seen in ``fit_frame`` falls back
    to its origin airport's rate; other unseen keys fall back to the overall rate.
    """
    overall = float(np.asarray(y_fit, dtype="float64").mean())
    scores: dict[str, np.ndarray] = {"baseline_prevalence": np.full(len(apply_frame), overall)}
    for name, column in BASELINE_KEYS.items():
        table = fit_rate_table(fit_frame[column], y_fit, strength)
        fallback = scores["baseline_origin_rate"] if name == "baseline_route_rate" else None
        scores[name] = apply_rate_table(table, apply_frame[column], fallback)
    return scores

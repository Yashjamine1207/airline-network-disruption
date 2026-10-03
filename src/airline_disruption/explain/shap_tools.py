"""SHAP values for the tuned LightGBM, using LightGBM's own TreeSHAP.

Phase 7A, Step 10. ``booster.predict(X, pred_contrib=True)`` returns exact Shapley values for trees
(Lundberg et al., TreeSHAP), one column per feature plus a last column with the expected value, all in
LOG-ODDS. It is the same algorithm as the ``shap`` package's TreeExplainer, without another dependency.

What a SHAP value means here: how much a feature moved this flight's log-odds away from the average flight,
given the model. It describes the model, not the world. A large value for an airport says the model
associates that airport with higher risk in its training data. It does not say the airport causes delay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature families for the summary. Every base_no_year column belongs to exactly one.
FAMILIES: dict[str, tuple[str, ...]] = {
    "calendar and clock": (
        "scheduled_departure_month", "scheduled_departure_day_of_week", "scheduled_departure_hour_local",
        "scheduled_departure_minute_local", "scheduled_arrival_hour_local", "scheduled_arrival_minute_local",
    ),
    "trip length": ("scheduled_elapsed_time_minutes", "distance_miles"),
    "airport": ("origin_airport", "destination_airport"),
    "route": ("route",),
    "carrier": ("carrier_identifier",),
    "prior-month schedule volume": (
        "prior_calendar_month_carrier_scheduled_flight_count", "prior_calendar_month_origin_scheduled_flight_count",
        "prior_calendar_month_route_scheduled_flight_count",
    ),
    "prior-month network degree": (
        "prior_month_origin_out_degree", "prior_month_origin_weighted_out_degree",
        "prior_month_destination_in_degree", "prior_month_destination_weighted_in_degree",
    ),
}
_FAMILY_OF = {feature: family for family, features in FAMILIES.items() for feature in features}


def family_of(feature: str) -> str:
    return _FAMILY_OF.get(feature, "other")


def contributions(model, X: pd.DataFrame, num_iteration: int | None = None) -> tuple[pd.DataFrame, float]:
    """``(per-feature SHAP values in log-odds, expected value)`` for the rows of ``X``."""
    booster = model.booster_
    raw = booster.predict(X, num_iteration=num_iteration, pred_contrib=True)
    raw = np.asarray(raw, dtype="float64")
    if raw.ndim != 2 or raw.shape[1] != X.shape[1] + 1:
        raise ValueError(f"Unexpected SHAP output shape {raw.shape} for {X.shape[1]} features")
    frame = pd.DataFrame(raw[:, :-1], columns=list(X.columns), index=X.index)
    return frame, float(raw[0, -1])


def check_additivity(model, X: pd.DataFrame, contrib: pd.DataFrame, expected_value: float,
                     num_iteration: int | None = None, tolerance: float = 1e-6) -> float:
    """Largest gap between (expected value + sum of SHAP values) and the model's raw log-odds.

    Raises ValueError if it exceeds ``tolerance``. The check is what makes the numbers trustworthy: SHAP
    values that do not add up to the model's own output are wrong.
    """
    raw = model.booster_.predict(X, num_iteration=num_iteration, raw_score=True)
    gap = float(np.max(np.abs(contrib.sum(axis=1).to_numpy() + expected_value - np.asarray(raw))))
    if gap > tolerance:
        raise ValueError(f"SHAP values do not add up to the model output (largest gap {gap:.3g})")
    return gap


def global_importance(contrib: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (log-odds), share of the total, and family. Largest first."""
    mean_abs = contrib.abs().mean().sort_values(ascending=False)
    table = pd.DataFrame({"feature": mean_abs.index, "mean_abs_shap": mean_abs.to_numpy()})
    table["share_of_total"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    table["family"] = table["feature"].map(family_of)
    return table.reset_index(drop=True)


def family_importance(importance: pd.DataFrame) -> pd.DataFrame:
    table = importance.groupby("family", as_index=False)["mean_abs_shap"].sum()
    table["share_of_total"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def numeric_effect(X: pd.Series, shap: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Mean SHAP value by equal-count bin of a numeric feature: which way does the model lean as the value rises."""
    valid = X.notna()
    if valid.sum() < n_bins:
        return pd.DataFrame(columns=["feature", "bin", "value_low", "value_high", "rows", "mean_shap"])
    bins = pd.qcut(X[valid].rank(method="first"), n_bins, labels=False)
    frame = pd.DataFrame({"value": X[valid].to_numpy(), "shap": shap[valid].to_numpy(), "bin": bins.to_numpy()})
    out = frame.groupby("bin").agg(value_low=("value", "min"), value_high=("value", "max"), rows=("shap", "size"),
                                   mean_shap=("shap", "mean")).reset_index()
    out["bin"] = out["bin"].astype(int) + 1
    out.insert(0, "feature", X.name)
    return out


def categorical_effect(X: pd.Series, shap: pd.Series, min_rows: int = 500, top: int = 10) -> pd.DataFrame:
    """Levels with the highest and lowest mean SHAP value (levels with at least ``min_rows`` rows)."""
    frame = pd.DataFrame({"level": X.astype("object").where(X.notna(), "unseen").to_numpy(), "shap": shap.to_numpy()})
    grouped = frame.groupby("level").agg(rows=("shap", "size"), mean_shap=("shap", "mean")).reset_index()
    grouped = grouped[grouped["rows"] >= min_rows]
    if grouped.empty:
        return pd.DataFrame(columns=["feature", "direction", "level", "rows", "mean_shap"])
    highest = grouped.nlargest(top, "mean_shap").assign(direction="highest")
    lowest = grouped.nsmallest(top, "mean_shap").assign(direction="lowest")
    out = pd.concat([highest, lowest], ignore_index=True)
    out.insert(0, "feature", X.name)
    return out[["feature", "direction", "level", "rows", "mean_shap"]]


def top_contributions(contrib_row: pd.Series, value_row: pd.Series, k: int = 6) -> list[tuple[str, str, float]]:
    """The ``k`` features with the largest absolute SHAP value for one flight: (feature, value, shap)."""
    order = contrib_row.abs().sort_values(ascending=False).index[:k]
    return [(f, str(value_row[f]), float(contrib_row[f])) for f in order]

"""Group labels for the error breakdown and the review-capacity tables (Phase 7A and 7B).

One function builds every grouping so the error analysis (Step 10) and the alert scenarios (Step 11) cut
the flights in exactly the same way. The labels use only the flight's own schedule features, its score,
and route volumes counted in the FIT rows (never from the rows being described).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = {
    "scheduled_departure_month", "scheduled_departure_day_of_week", "scheduled_departure_hour_local",
    "scheduled_departure_minute_local", "scheduled_arrival_hour_local", "scheduled_arrival_minute_local",
    "scheduled_elapsed_time_minutes", "distance_miles",
    "prior_calendar_month_carrier_scheduled_flight_count", "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count", "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree", "prior_month_destination_in_degree", "prior_month_destination_weighted_in_degree",
}
CATEGORICAL_FEATURES = ("origin_airport", "destination_airport", "route", "carrier_identifier")
ROUTE_VOLUME_EDGES = [-1, 0, 99, 999, np.inf]
ROUTE_VOLUME_LABELS = ["unseen route (0 fit-period flights)", "1-99", "100-999", "1,000 or more"]


def band_label(hour: float) -> str:
    """Local scheduled departure hour (0 to 23) to a five-way band."""
    if hour < 6:
        return "00-05"
    if hour < 10:
        return "06-09"
    if hour < 14:
        return "10-13"
    if hour < 18:
        return "14-17"
    return "18-23"


def build_dimensions(X_val: pd.DataFrame, X_fit: pd.DataFrame, score, period_labels, period_name: str = "quarter_2022") -> dict[str, pd.Series]:
    """One label per validation row for every dimension (each a Series with a fresh 0..n-1 index).

    ``X_val`` and ``X_fit`` hold the model's features (categories already restricted to the fit rows, so a level
    never seen in the fit rows is missing). ``score`` and ``period_labels`` line up with the rows of ``X_val``.
    """
    X_val = X_val.reset_index(drop=True)
    score = np.asarray(score, dtype="float64")
    if len(score) != len(X_val) or len(period_labels) != len(X_val):
        raise ValueError("score and period_labels must have one entry per row of X_val")
    origin = X_val["origin_airport"].astype("object").where(X_val["origin_airport"].notna(), "unseen")
    top_origins = origin.value_counts().index[:25]
    route_counts = X_fit["route"].value_counts()
    volume = X_val["route"].astype("object").map(route_counts).fillna(0)
    volume_band = pd.cut(volume, ROUTE_VOLUME_EDGES, labels=ROUTE_VOLUME_LABELS)
    unseen = X_val[list(CATEGORICAL_FEATURES)].isna().any(axis=1)
    missing_numeric = X_val[[c for c in X_val.columns if c in NUMERIC_FEATURES]].isna().any(axis=1)
    completeness = np.select([unseen & missing_numeric, unseen, missing_numeric],
                             ["unseen category and missing number", "unseen category", "missing number"], "complete")
    decile = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False) + 1
    return {
        period_name: pd.Series(list(period_labels)),
        "carrier": X_val["carrier_identifier"].astype("object").where(X_val["carrier_identifier"].notna(), "unseen"),
        "origin_airport_top25": origin.where(origin.isin(top_origins), "all other airports"),
        "route_volume_in_fit_period": volume_band.astype("object"),
        "scheduled_departure_local_hours": X_val["scheduled_departure_hour_local"].map(band_label),
        "day_of_week": X_val["scheduled_departure_day_of_week"].astype("int64").astype(str),
        "missing_or_unseen_values": pd.Series(completeness),
        "score_decile": decile.map(lambda d: f"decile {int(d):02d}" + (" (lowest)" if d == 1 else " (highest)" if d == 10 else "")),
    }

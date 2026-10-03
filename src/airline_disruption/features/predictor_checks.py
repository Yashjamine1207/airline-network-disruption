"""Checks on the Phase 6 predictor table.

Phase 6, Step 3. Pure functions on DataFrames, no file input/output.

Four checks:

1. ``check_alignment``: predictor rows and cohort rows are the same flights,
   with the same UTC prediction timestamps and the same local flight dates.
2. ``null_rate_table``: missing values per feature, development and validation only.
3. ``unseen_category_table``: validation categories never seen in development.
4. ``verify_prior_month_features``: recompute the seven prior-month features from
   the schedule columns and compare them with the stored values.

The final-test split is deliberately left out of checks 2 and 3. They describe
features, not labels, but nothing that shapes preprocessing should be read off
the locked period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMN = "source_row_number"


def to_utc(series: pd.Series) -> pd.Series:
    """Return a timezone-aware UTC datetime series.

    Accepts ISO-8601 strings or an already timezone-aware column. A naive
    datetime column is refused because its zone is unknown.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("Naive datetime column: the timezone is unknown, refusing to assume UTC")
    return pd.to_datetime(series, utc=True, format="ISO8601")


# ---------------------------------------------------------------------------
# 1. Alignment between the cohort table and the predictor table
# ---------------------------------------------------------------------------
def check_alignment(cohort: pd.DataFrame, predictors: pd.DataFrame) -> None:
    """Raise ValueError unless the two tables describe the same flights.

    ``cohort`` needs: source_row_number, prediction_timestamp_utc, flight_date.
    ``predictors`` needs: source_row_number, prediction_timestamp_utc, FL_DATE.
    Row order does not matter.
    """
    for name, table in (("cohort", cohort), ("predictors", predictors)):
        if table[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} table has duplicate {ID_COLUMN} values")

    cohort_ids = pd.Index(cohort[ID_COLUMN])
    predictor_ids = pd.Index(predictors[ID_COLUMN])
    only_cohort = len(cohort_ids.difference(predictor_ids))
    only_predictors = len(predictor_ids.difference(cohort_ids))
    if only_cohort or only_predictors:
        raise ValueError(
            f"Tables do not describe the same flights: {only_cohort:,} ids only in cohort, "
            f"{only_predictors:,} ids only in predictors"
        )

    merged = cohort[[ID_COLUMN, "prediction_timestamp_utc", "flight_date"]].merge(
        predictors[[ID_COLUMN, "prediction_timestamp_utc", "FL_DATE"]],
        on=ID_COLUMN,
        suffixes=("_cohort", "_predictors"),
        validate="one_to_one",
    )
    time_mismatch = int((to_utc(merged["prediction_timestamp_utc_cohort"]) != to_utc(merged["prediction_timestamp_utc_predictors"])).sum())
    date_mismatch = int((pd.to_datetime(merged["flight_date"]) != pd.to_datetime(merged["FL_DATE"])).sum())
    if time_mismatch or date_mismatch:
        raise ValueError(
            f"Tables disagree: {time_mismatch:,} different prediction timestamps, {date_mismatch:,} different flight dates"
        )


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
def null_rate_table(
    frame: pd.DataFrame,
    split: pd.Series,
    columns: list[str] | tuple[str, ...],
    splits: tuple[str, ...] = ("development", "validation"),
) -> pd.DataFrame:
    """Share of missing values per feature and split.

    ``split`` must share ``frame``'s index. Returns one row per feature with a
    ``null_rate_<split>`` column for each requested split.
    """
    labels = split.astype(str).to_numpy()
    rows = []
    for column in columns:
        missing = frame[column].isna().to_numpy()
        row: dict[str, object] = {"feature": column}
        for name in splits:
            mask = labels == name
            row[f"null_rate_{name}"] = float(missing[mask].mean()) if mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Categories the model has never seen
# ---------------------------------------------------------------------------
def unseen_category_table(
    frame: pd.DataFrame,
    split: pd.Series,
    columns: list[str] | tuple[str, ...],
    fit_split: str = "development",
    eval_split: str = "validation",
) -> pd.DataFrame:
    """For each categorical column, how much of ``eval_split`` uses categories absent from ``fit_split``."""
    labels = split.astype(str).to_numpy()
    fit_mask = labels == fit_split
    eval_mask = labels == eval_split
    rows = []
    for column in columns:
        fit_values = frame.loc[fit_mask, column].dropna()
        eval_values = frame.loc[eval_mask, column].dropna()
        known = pd.Index(fit_values.astype(str).unique())
        unseen_flags = ~eval_values.astype(str).isin(known)
        rows.append(
            {
                "feature": column,
                f"categories_{fit_split}": int(len(known)),
                f"categories_{eval_split}": int(eval_values.astype(str).nunique()),
                f"unseen_categories_in_{eval_split}": int(eval_values.astype(str)[unseen_flags].nunique()),
                f"unseen_row_share_{eval_split}": float(unseen_flags.mean()) if len(eval_values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Independent recomputation of the prior-month features
# ---------------------------------------------------------------------------
# (stored column, entity column, "count" | "nunique", other column for nunique, definition text)
PRIOR_MONTH_CHECKS = (
    (
        "prior_calendar_month_carrier_scheduled_flight_count",
        "carrier_identifier",
        "count",
        None,
        "scheduled flights of the carrier in the previous UTC month",
    ),
    (
        "prior_calendar_month_origin_scheduled_flight_count",
        "origin_airport",
        "count",
        None,
        "scheduled flights leaving the origin in the previous UTC month",
    ),
    (
        "prior_calendar_month_route_scheduled_flight_count",
        "route",
        "count",
        None,
        "scheduled flights on the route in the previous UTC month",
    ),
    (
        "prior_month_origin_out_degree",
        "origin_airport",
        "nunique",
        "destination_airport",
        "distinct destinations served from the origin in the previous UTC month",
    ),
    (
        "prior_month_origin_weighted_out_degree",
        "origin_airport",
        "count",
        None,
        "scheduled flights leaving the origin in the previous UTC month",
    ),
    (
        "prior_month_destination_in_degree",
        "destination_airport",
        "nunique",
        "origin_airport",
        "distinct origins serving the destination in the previous UTC month",
    ),
    (
        "prior_month_destination_weighted_in_degree",
        "destination_airport",
        "count",
        None,
        "scheduled flights arriving at the destination in the previous UTC month",
    ),
)

_MONTH_STRIDE = 100_000  # month index stays far below this, so entity*stride+month is unique


def _codes(series: pd.Series) -> np.ndarray:
    """Integer codes for a column. Missing values become -1."""
    return series.astype("category").cat.codes.to_numpy(dtype="int64")


def verify_prior_month_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute each prior-month feature and compare it with the stored column.

    The month of a row is the UTC month of ``scheduled_departure_utc``. The
    previous month is the calendar month before it. Rows in the first month of
    the data have no previous month in the file, so they are reported
    separately and excluded from the comparison.

    ``frame`` needs: scheduled_departure_utc, carrier_identifier,
    origin_airport, destination_airport, route, and the seven stored columns.

    Returns one row per feature: comparable rows, exact matches, mismatches,
    stored nulls among comparable rows, and the mismatch rate. A mismatch does
    not prove leakage. It means the stored feature is not exactly "count or
    degree in the previous UTC month of this file", and the Phase 3 build
    script should be read to see why.
    """
    stamps = to_utc(frame["scheduled_departure_utc"])
    month = (stamps.dt.year * 12 + stamps.dt.month).to_numpy(dtype="int64")
    first_month = int(month.min())
    has_previous_month = (month - 1) >= first_month

    rows = []
    for stored_column, entity_column, method, other_column, definition in PRIOR_MONTH_CHECKS:
        entity = _codes(frame[entity_column])
        present = entity >= 0

        # Aggregate per (entity, month).
        key = entity * _MONTH_STRIDE + month
        if method == "count":
            per_key = pd.Series(key[present]).value_counts()
        else:
            other = _codes(frame[other_column])
            pairs = pd.DataFrame({"key": key[present], "other": other[present]}).drop_duplicates()
            per_key = pairs.groupby("key").size()

        # Look up the same entity one month earlier. No flights that month means 0.
        previous_key = entity * _MONTH_STRIDE + (month - 1)
        expected = per_key.reindex(previous_key).fillna(0).to_numpy(dtype="float64")

        stored = pd.to_numeric(frame[stored_column], errors="coerce").to_numpy(dtype="float64")
        comparable = has_previous_month & present
        stored_null = np.isnan(stored)
        matches = comparable & ~stored_null & (stored == expected)
        mismatches = comparable & ~stored_null & (stored != expected)
        first_month_rows = (~has_previous_month) & present

        comparable_rows = int(comparable.sum())
        rows.append(
            {
                "feature": stored_column,
                "definition": definition,
                "comparable_rows": comparable_rows,
                "matches": int(matches.sum()),
                "mismatches": int(mismatches.sum()),
                "stored_null_in_comparable_rows": int((comparable & stored_null).sum()),
                "mismatch_rate": float(mismatches.sum() / comparable_rows) if comparable_rows else float("nan"),
                "first_month_rows": int(first_month_rows.sum()),
                "first_month_stored_null": int((first_month_rows & stored_null).sum()),
            }
        )
    return pd.DataFrame(rows)

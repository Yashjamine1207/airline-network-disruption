"""Row selection and benchmark alignment for the neural-network runs (no TensorFlow).

Phase 6, Step 7. The training script uses these functions so the checks are tested with
plain NumPy and pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airline_disruption.features.feature_sets import ID_COLUMN


def require_two_classes(labels: np.ndarray, name: str) -> None:
    """Raise ValueError if ``labels`` has fewer than two classes (a smoke subsample can be too small)."""
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"The {name} rows contain only one class ({len(labels):,} rows). "
            "A smoke-run subsample may be too small for a rare event."
        )


def subsample_rows(rows: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    """A sorted random subset of ``rows`` (time order kept). ``None`` or a large limit returns all rows."""
    rows = np.asarray(rows)
    if max_rows is None or len(rows) <= max_rows:
        return rows
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    keep = np.sort(np.random.default_rng(seed).choice(len(rows), size=max_rows, replace=False))
    return rows[keep]


def align_scores(
    row_ids: np.ndarray,
    labels: np.ndarray,
    other: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Scores from ``other[column]`` for ``row_ids``, in that order.

    ``other`` is a predictions table (columns ``source_row_number``, ``label`` and ``column``), for
    example the LightGBM benchmark's. Raises ValueError if a row is missing or a label disagrees:
    two models are only comparable when they scored the same rows with the same labels.
    """
    for needed in (ID_COLUMN, "label", column):
        if needed not in other.columns:
            raise ValueError(f"The predictions table has no column {needed!r}")
    if other[ID_COLUMN].duplicated().any():
        raise ValueError("The predictions table has duplicate source_row_number values")
    positions = pd.Index(other[ID_COLUMN]).get_indexer(row_ids)
    if (positions < 0).any():
        raise ValueError(f"{int((positions < 0).sum()):,} validation rows are missing from the predictions table")
    if not np.array_equal(other["label"].to_numpy()[positions].astype("int64"), np.asarray(labels, dtype="int64")):
        raise ValueError("Labels in the predictions table differ from the labels of the same rows")
    scores = other[column].to_numpy(dtype="float64")[positions]
    if not np.isfinite(scores).all():
        raise ValueError(f"Column {column!r} contains NaN or infinite scores")
    return scores


def verdict(difference: float, ci_low: float, ci_high: float, challenger: str, reference: str) -> str:
    """One plain sentence describing a paired difference (challenger minus reference)."""
    if ci_low > 0:
        return f"{challenger} is better than {reference}: the whole 95% interval is above zero."
    if ci_high < 0:
        return f"{reference} is better than {challenger}: the whole 95% interval is below zero."
    return f"No clear difference between {challenger} and {reference}: the 95% interval includes zero (point estimate {difference:+.5f})."


# ---------------------------------------------------------------------------
# The 2020 ablation: which fit rows a run trains on
# ---------------------------------------------------------------------------
FIT_VARIANTS = ("full", "no2020", "matched")
SHOCK_YEAR = 2020


def calendar_year_utc(prediction_seconds: np.ndarray) -> np.ndarray:
    """Calendar year (UTC) of each prediction time given in seconds since 1970-01-01 UTC."""
    seconds = np.asarray(prediction_seconds, dtype="int64")
    return seconds.astype("datetime64[s]").astype("datetime64[Y]").astype("int64") + 1970


def fit_variant_rows(rows: np.ndarray, prediction_seconds: np.ndarray, variant: str, seed: int) -> np.ndarray:
    """The fit rows a run trains on, for the "train with 2020 versus without 2020" ablation.

    ``rows`` are positions of the fit rows and ``prediction_seconds`` is indexed by those
    positions. Variants:

    full      every fit row.
    no2020    fit rows whose prediction time is not in calendar year 2020 (UTC).
    matched   a random subset of the fit rows with exactly as many rows as ``no2020`` has.

    Without ``matched``, "no 2020" would confound two things: 2020 is gone AND there is less
    data. ``matched`` removes the same number of rows but spread over all years, so it
    isolates the effect of the amount of data. The result is always in time order.
    """
    if variant not in FIT_VARIANTS:
        raise ValueError(f"Unknown fit variant {variant!r}; choose from {FIT_VARIANTS}")
    rows = np.asarray(rows)
    if variant == "full":
        return rows
    in_shock = calendar_year_utc(np.asarray(prediction_seconds)[rows]) == SHOCK_YEAR
    if not in_shock.any():
        raise ValueError(f"The fit rows contain no rows from {SHOCK_YEAR}, so there is nothing to remove")
    kept = rows[~in_shock]
    if len(kept) == 0:
        raise ValueError(f"Every fit row is from {SHOCK_YEAR}")
    if variant == "no2020":
        return kept
    return subsample_rows(rows, len(kept), seed)


def fit_composition(prediction_seconds: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Rows, events and event rate of the fit rows by UTC calendar year."""
    frame = pd.DataFrame({"year": calendar_year_utc(prediction_seconds), "event": np.asarray(labels, dtype="int64")})
    table = frame.groupby("year").agg(rows=("event", "size"), events=("event", "sum")).reset_index()
    table["event_rate"] = table["events"] / table["rows"]
    table["share_of_fit_rows"] = table["rows"] / table["rows"].sum()
    return table


def decide_drop_2020(ci_low_vs_full: float, ci_low_vs_matched: float) -> bool:
    """DEC-021: training without 2020 is adopted only if it beats BOTH comparisons clearly.

    ``ci_low_vs_full`` is the lower end of the 95% interval of (no2020 minus full). It is not enough:
    fewer rows can change a score for reasons that have nothing to do with 2020. So the
    no-2020 model must also beat a model trained on the same NUMBER of random rows
    (``ci_low_vs_matched``). Both lower ends must be above zero. NaN counts as "no".
    """
    return bool(np.isfinite(ci_low_vs_full) and np.isfinite(ci_low_vs_matched) and ci_low_vs_full > 0.0 and ci_low_vs_matched > 0.0)

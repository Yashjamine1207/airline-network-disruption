"""Refit the selected LightGBM exactly as the benchmark did and keep everything analysis needs.

Phase 7A, Step 10. The calibration script (Step 9) contains the same steps inline. This helper lets the
explanation and error-analysis scripts use the same rows, roles, category levels and model without copying
the code again. Only development and 2022 rows are loaded; the final test cannot load.

The result is checked against the stored validation scores by the caller (``check_against_stored``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from airline_disruption.deep import protocol as proto
from airline_disruption.features.feature_sets import ID_COLUMN
from airline_disruption.models import lightgbm_benchmark as lb
from airline_disruption.sequences import airport_bins as ab


@dataclass
class Refit:
    fit: lb.FitResult
    X: pd.DataFrame  # fit rows, then early-stop rows, then validation rows; categories restricted to the fit rows
    n_fit: int
    n_stop: int
    y_fit: np.ndarray
    y_stop: np.ndarray
    y_val: np.ndarray
    ids_stop: np.ndarray
    ids_val: np.ndarray
    seconds_stop: np.ndarray
    seconds_val: np.ndarray
    meta_val: pd.DataFrame
    s_stop: np.ndarray
    s_val: np.ndarray
    seconds_fit: np.ndarray | None = None  # UTC prediction time of the fit rows, seconds since 1970 (Step 12)

    @property
    def X_fit(self) -> pd.DataFrame:
        return self.X.iloc[: self.n_fit]

    @property
    def X_stop(self) -> pd.DataFrame:
        return self.X.iloc[self.n_fit : self.n_fit + self.n_stop]

    @property
    def X_val(self) -> pd.DataFrame:
        return self.X.iloc[self.n_fit + self.n_stop :]


def refit_selected_lightgbm(root: Path, feature_set: str, params: dict, seed: int, n_jobs: int,
                            max_fit_rows: int | None = None, target: str = lb.TARGET,
                            eligibility_column: str = lb.ELIGIBILITY_COLUMN) -> Refit:
    """Load rows, assign roles, fit LightGBM on the fit rows (early stopping on the early-stop rows), score both.

    ``target`` and ``eligibility_column`` default to severe delay. Step 12 also refits the cancellation baseline.
    """
    frame = lb.load_model_frame(root, feature_set, target=target, eligibility_column=eligibility_column)  # development + 2022 only
    roles = lb.assign_roles(frame.meta, lb.PROTOCOL_PHASE6)
    lb.check_role_chronology(frame.meta, roles)
    lb.mask_unavailable_history(frame.X, frame.meta)

    fit_pos = np.flatnonzero((roles == lb.ROLE_FIT).to_numpy())
    stop_pos = np.flatnonzero((roles == lb.ROLE_EARLY_STOP).to_numpy())
    val_pos = np.flatnonzero((roles == lb.ROLE_VALIDATION).to_numpy())
    seconds = ab.to_epoch_seconds(frame.meta["prediction_timestamp_utc"])
    fit_pos = proto.subsample_rows(fit_pos, max_fit_rows, seed)

    order = np.concatenate([fit_pos, stop_pos, val_pos])
    X = frame.X.iloc[order].reset_index(drop=True)
    fit_mask = np.zeros(len(X), dtype=bool)
    fit_mask[: len(fit_pos)] = True
    lb.apply_fit_categories(X, fit_mask)
    n_fit, n_stop = len(fit_pos), len(stop_pos)
    fit = lb.fit_lightgbm(X.iloc[:n_fit], frame.y[fit_pos], X.iloc[n_fit : n_fit + n_stop], frame.y[stop_pos],
                          n_jobs=n_jobs, seed=seed, params=params)
    s_stop = lb.predict_scores(fit, X.iloc[n_fit : n_fit + n_stop])
    s_val = lb.predict_scores(fit, X.iloc[n_fit + n_stop :])
    ids = frame.meta[ID_COLUMN].to_numpy()
    return Refit(
        fit=fit, X=X, n_fit=n_fit, n_stop=n_stop, y_fit=frame.y[fit_pos], y_stop=frame.y[stop_pos], y_val=frame.y[val_pos],
        ids_stop=ids[stop_pos], ids_val=ids[val_pos], seconds_stop=seconds[stop_pos], seconds_val=seconds[val_pos],
        meta_val=frame.meta.iloc[val_pos].reset_index(drop=True), s_stop=s_stop, s_val=s_val,
        seconds_fit=seconds[fit_pos],
    )


def check_against_stored(refit: Refit, stored_path: Path, column: str, tolerance: float) -> dict:
    """Compare the refitted validation scores with the stored ones. Raises ValueError beyond ``tolerance`` (PR-AUC)."""
    from airline_disruption.evaluation.classification import binary_metrics

    stored = proto.align_scores(refit.ids_val, refit.y_val, pd.read_parquet(stored_path), column)
    difference = abs(binary_metrics(refit.y_val, refit.s_val)["pr_auc"] - binary_metrics(refit.y_val, stored)["pr_auc"])
    result = {"checked": True, "file": str(stored_path), "column": column, "pr_auc_difference": float(difference),
              "largest_score_difference": float(np.max(np.abs(refit.s_val - stored)))}
    if difference > tolerance:
        raise ValueError("The refitted model does not reproduce the stored validation scores, so a different model would be analysed.")
    return result

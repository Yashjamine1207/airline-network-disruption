"""One calibration experiment as plain functions on arrays (used by the script and by the tests).

Phase 7A, Step 9. An experiment fits every method on one set of rows (labels and scores) and applies
the fitted calibrators to the evaluation rows. The evaluation labels are used only afterwards, for
metrics. ``fit_and_apply`` takes no evaluation labels at all, which is how the tests show that
they cannot influence a calibrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airline_disruption.calibration import guards
from airline_disruption.calibration.methods import Calibrator, fit_calibrators
from airline_disruption.calibration.metrics import evaluate_probabilities


def fit_and_apply(
    methods,
    scores_fit,
    y_fit,
    seconds_fit,
    scores_apply,
    seconds_apply,
    name: str,
    ids_fit=None,
    ids_apply=None,
) -> tuple[dict[str, Calibrator], dict[str, np.ndarray]]:
    """Fit on the calibration rows, apply to later rows. Returns ``(calibrators, probabilities by method)``.

    The guards run first: no final-test rows, calibration strictly before application, and (when row
    ids are given) no shared rows. There is no argument for the labels of the rows being scored.
    """
    guards.reject_final_test_rows(seconds_fit, f"{name} calibration")
    guards.reject_final_test_rows(seconds_apply, f"{name} evaluation")
    guards.require_calibration_before_evaluation(seconds_fit, seconds_apply, name)
    if ids_fit is not None and ids_apply is not None:
        guards.require_disjoint(ids_fit, ids_apply, name)
    calibrators = fit_calibrators(methods, scores_fit, y_fit)
    return calibrators, {m: c.transform(scores_apply) for m, c in calibrators.items()}


def metrics_table(
    experiment: str,
    fit_rows_label: str,
    y,
    probabilities: dict[str, np.ndarray],
    periods: dict[str, np.ndarray],
    reference_rate: float,
    n_bins: int,
) -> pd.DataFrame:
    """One row per (method, period): every calibration metric. ``periods`` maps a name to a boolean row mask."""
    y = np.asarray(y)
    rows = []
    for method, p in probabilities.items():
        for period, mask in periods.items():
            if not mask.any():
                continue
            values = evaluate_probabilities(y[mask], np.asarray(p)[mask], reference_rate, n_bins)
            rows.append({"experiment": experiment, "calibrator_fitted_on": fit_rows_label, "method": method,
                         "period": period, **values})
    return pd.DataFrame(rows)

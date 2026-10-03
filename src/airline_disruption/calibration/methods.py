"""Calibrators: uncalibrated, Platt (sigmoid) and isotonic.

Phase 7A, Step 9. A calibrator maps a model score to a probability. It is fitted on one period
and applied to later periods. Nothing here reads a date or a file: the caller decides which rows
are used for fitting (see ``guards.py``), so these classes are easy to test with plain arrays.

Methods (``METHODS``)
---------------------
uncalibrated  The score as it is. Always kept as the reference.
platt         Logistic regression of the label on logit(score): p = expit(a + b * logit(score)).
              Two numbers. Strictly increasing when b > 0, so it never changes the ranking of
              flights, only the probability attached to each rank. No smoothed targets are used
              (Platt's original paper smooths the labels); with hundreds of thousands of rows the
              difference is negligible.
isotonic      A non-decreasing step function fitted by pool-adjacent-violators. No shape
              assumption, but it needs more rows, it can overfit, and it merges many scores into
              the same value (ties). Ties can lower PR-AUC and blur the ranking.

Scores outside the fitted range are clipped to the nearest fitted value (isotonic) or follow the
fitted line (Platt). Every output is kept inside [1e-6, 1 - 1e-6], so a logit is always finite.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

METHODS = ("uncalibrated", "platt", "isotonic")
PROBABILITY_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def check_scores(scores, name: str = "scores") -> np.ndarray:
    """Return ``scores`` as a float64 vector, or raise ValueError if they are not probabilities."""
    s = np.asarray(scores, dtype="float64")
    if s.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(s) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(s).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if s.min() < 0.0 or s.max() > 1.0:
        raise ValueError(f"{name} must lie in [0, 1] (found {s.min():.4g} to {s.max():.4g})")
    return s


def check_labels(labels, expected_length: int | None = None, name: str = "labels") -> np.ndarray:
    """Return labels as int8 zeros and ones. Raises ValueError for anything else or for one class only."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and len(y) != expected_length:
        raise ValueError(f"{name} and scores must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f"{name} must be 0 or 1")
    y = y.astype("int8")
    if len(y) == 0 or y.min() == y.max():
        raise ValueError(f"{name} contain only one class, so a calibrator cannot be fitted")
    return y


def clip_probabilities(p, floor: float = PROBABILITY_FLOOR) -> np.ndarray:
    return np.clip(np.asarray(p, dtype="float64"), floor, 1.0 - floor)


def to_logit(p, floor: float = PROBABILITY_FLOOR) -> np.ndarray:
    """log(p / (1 - p)) after clipping to [floor, 1 - floor]."""
    q = clip_probabilities(p, floor)
    return np.log(q / (1.0 - q))


def from_logit(z) -> np.ndarray:
    return expit(np.asarray(z, dtype="float64"))


def fit_logistic(z: np.ndarray, y: np.ndarray, slope: bool = True, max_iter: int = 60, tol: float = 1e-10):
    """Unpenalised logistic regression by Newton-Raphson (no scikit-learn penalty, no scaling).

    ``slope=True``:  P(y = 1) = expit(a + b * z). Returns ``(a, b)``. This is the calibration
                     slope and intercept when ``z`` is the logit of a model's probability.
    ``slope=False``: P(y = 1) = expit(a + z), the slope fixed at 1. Returns ``(a, 1.0)``. This is
                     "calibration in the large": how far the average level is off.

    Step-halving keeps every iteration from making the fit worse. Raises ValueError if it does
    not converge (for example with completely separated data).
    """
    z = np.asarray(z, dtype="float64")
    y = np.asarray(y, dtype="float64")
    design = np.column_stack([np.ones_like(z), z]) if slope else np.ones((len(z), 1))
    offset = np.zeros_like(z) if slope else z
    beta = np.zeros(design.shape[1])
    if not slope:
        beta[0] = float(np.log(y.mean() / (1.0 - y.mean()))) - float(np.mean(z))  # a sensible start

    def log_likelihood(b: np.ndarray) -> float:
        eta = design @ b + offset
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current = log_likelihood(beta)
    for _ in range(max_iter):
        mu = expit(design @ beta + offset)
        gradient = design.T @ (y - mu)
        hessian = (design * (mu * (1.0 - mu))[:, None]).T @ design + 1e-12 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            value = log_likelihood(candidate)
            if value >= current - 1e-12:
                break
            scale /= 2.0
        else:
            raise ValueError("Logistic fit could not improve the likelihood")
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            break
    else:
        raise ValueError("Logistic fit did not converge")
    if not np.isfinite(beta).all():
        raise ValueError("Logistic fit produced non-finite coefficients")
    return (float(beta[0]), float(beta[1])) if slope else (float(beta[0]), 1.0)


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------
class Calibrator:
    """Common interface. ``fit`` returns self; ``transform`` returns probabilities for new scores."""

    name = "base"

    def __init__(self) -> None:
        self.fit_rows: int | None = None
        self.fit_events: int | None = None

    def fit(self, scores, labels) -> "Calibrator":
        s = check_scores(scores)
        y = check_labels(labels, len(s))
        self.fit_rows, self.fit_events = int(len(y)), int(y.sum())
        self._fit(s, y)
        return self

    def transform(self, scores) -> np.ndarray:
        if self.fit_rows is None:
            raise ValueError(f"The {self.name} calibrator has not been fitted")
        return clip_probabilities(self._transform(check_scores(scores)))

    def describe(self) -> dict:
        return {"method": self.name, "fit_rows": self.fit_rows, "fit_events": self.fit_events}

    def _fit(self, s: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _transform(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityCalibrator(Calibrator):
    """The uncalibrated score. Fitting only records how many rows were shown."""

    name = "uncalibrated"

    def _fit(self, s, y) -> None:
        return None

    def _transform(self, s):
        return s

    def transform(self, scores) -> np.ndarray:
        """The score itself, unclipped, so the reference metrics equal the ones reported for the model."""
        if self.fit_rows is None:
            raise ValueError("The uncalibrated calibrator has not been fitted")
        return check_scores(scores).copy()


class PlattCalibrator(Calibrator):
    """Sigmoid calibration on the logit of the score."""

    name = "platt"

    def __init__(self) -> None:
        super().__init__()
        self.intercept: float | None = None
        self.slope: float | None = None

    def _fit(self, s, y) -> None:
        a, b = fit_logistic(to_logit(s), y, slope=True)
        if not b > 0.0:
            raise ValueError(
                f"The Platt slope is {b:.4g}, not positive. The calibrated ranking would not match the model's ranking."
            )
        self.intercept, self.slope = a, b

    def _transform(self, s):
        return from_logit(self.intercept + self.slope * to_logit(s))

    def describe(self) -> dict:
        return {**super().describe(), "intercept": self.intercept, "slope": self.slope}


class IsotonicCalibrator(Calibrator):
    """Monotone step function. Scores outside the fitted range take the nearest fitted value."""

    name = "isotonic"

    def __init__(self) -> None:
        super().__init__()
        self._model: IsotonicRegression | None = None

    def _fit(self, s, y) -> None:
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self._model.fit(s, y)

    def _transform(self, s):
        return self._model.predict(s)

    def describe(self) -> dict:
        knots = int(len(self._model.X_thresholds_)) if self._model is not None else None
        return {**super().describe(), "steps": knots}


_FACTORIES = {"uncalibrated": IdentityCalibrator, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def make_calibrator(method: str) -> Calibrator:
    if method not in _FACTORIES:
        raise ValueError(f"Unknown calibration method {method!r}; choose from {METHODS}")
    return _FACTORIES[method]()


def fit_calibrators(methods, scores, labels) -> dict[str, Calibrator]:
    """Fit every method in ``methods`` on the same rows. The result is keyed by method name."""
    return {m: make_calibrator(m).fit(scores, labels) for m in methods}

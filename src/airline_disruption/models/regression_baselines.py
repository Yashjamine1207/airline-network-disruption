"""Training-only global baselines for completed-flight arrival delay.

The caller is responsible for selecting eligible completed-flight labels
and passing labels from the development/training period only.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

RegressionBaselineStrategy = Literal["mean", "median"]


def fit_global_delay_baseline(
    training_delays: Sequence[float] | pd.Series | np.ndarray,
    strategy: RegressionBaselineStrategy,
) -> DummyRegressor:
    """Fit a global mean or median arrival-delay baseline.

    Delay values may be negative: an early arrival is a valid outcome.
    Missing, non-numeric, and infinite outcomes are rejected rather
    than silently removed or replaced.
    """
    if strategy not in ("mean", "median"):
        raise ValueError("strategy must be 'mean' or 'median'.")

    delays = pd.Series(training_delays).reset_index(drop=True)

    if delays.empty:
        raise ValueError("Training delays cannot be empty.")

    numeric_delays = pd.to_numeric(delays, errors="coerce")

    if numeric_delays.isna().any():
        raise ValueError(
            "Training delays must not contain missing or non-numeric values."
        )

    values = numeric_delays.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Training delays must contain only finite values.")

    # These baselines learn only from training outcomes, not flight features.
    placeholder_features = np.zeros((len(values), 1), dtype=np.uint8)

    model = DummyRegressor(strategy=strategy)
    model.fit(placeholder_features, values)
    return model


def predict_global_delay(
    model: DummyRegressor,
    number_of_flights: int,
) -> np.ndarray:
    """Predict the fitted constant arrival delay, in minutes."""
    if isinstance(number_of_flights, bool) or not isinstance(
        number_of_flights, int
    ):
        raise TypeError("number_of_flights must be an integer.")

    if number_of_flights < 0:
        raise ValueError("number_of_flights cannot be negative.")

    if not hasattr(model, "constant_"):
        raise ValueError("The baseline must be fitted before prediction.")

    if number_of_flights == 0:
        return np.empty(0, dtype=np.float64)

    placeholder_features = np.zeros(
        (number_of_flights, 1), dtype=np.uint8
    )
    return model.predict(placeholder_features)
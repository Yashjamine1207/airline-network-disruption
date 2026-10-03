"""Tests for global arrival-delay regression baselines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from airline_disruption.models.regression_baselines import (
    fit_global_delay_baseline,
    predict_global_delay,
)


def test_mean_baseline_uses_training_mean() -> None:
    model = fit_global_delay_baseline(
        training_delays=[-10, 0, 30, 100],
        strategy="mean",
    )

    predictions = predict_global_delay(model, number_of_flights=3)

    np.testing.assert_allclose(predictions, [30.0, 30.0, 30.0])


def test_median_baseline_uses_training_median() -> None:
    model = fit_global_delay_baseline(
        training_delays=[-10, 0, 30, 100],
        strategy="median",
    )

    predictions = predict_global_delay(model, number_of_flights=2)

    np.testing.assert_allclose(predictions, [15.0, 15.0])


def test_zero_prediction_rows_returns_empty_array() -> None:
    model = fit_global_delay_baseline([0, 10], strategy="median")

    predictions = predict_global_delay(model, number_of_flights=0)

    assert predictions.shape == (0,)


@pytest.mark.parametrize(
    "delays",
    [
        [],
        [0, None],
        [0, "not-a-delay"],
        [0, np.inf],
        [0, -np.inf],
        pd.Series([0, pd.NA], dtype="Float64"),
    ],
)
def test_invalid_training_delays_are_rejected(delays: object) -> None:
    with pytest.raises(ValueError):
        fit_global_delay_baseline(delays, strategy="mean")


def test_invalid_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_global_delay_baseline([0, 10], strategy="constant")  # type: ignore[arg-type]


def test_negative_prediction_count_is_rejected() -> None:
    model = fit_global_delay_baseline([0, 10], strategy="mean")

    with pytest.raises(ValueError):
        predict_global_delay(model, number_of_flights=-1)


def test_unfitted_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        predict_global_delay(DummyRegressor(strategy="mean"), 2)
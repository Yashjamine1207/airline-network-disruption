"""Tests for the Phase 4 training-period prevalence baseline."""

import numpy as np
import pandas as pd
import pytest

from airline_disruption.models.baselines import (
    fit_prevalence_baseline,
    predict_event_probability,
)


def test_predictions_use_training_prevalence() -> None:
    model = fit_prevalence_baseline([0, 0, 0, 1])

    probabilities = predict_event_probability(model, number_of_flights=3)

    np.testing.assert_allclose(probabilities, [0.25, 0.25, 0.25])


def test_prediction_can_return_zero_rows() -> None:
    model = fit_prevalence_baseline([0, 1])

    probabilities = predict_event_probability(model, number_of_flights=0)

    assert probabilities.shape == (0,)


@pytest.mark.parametrize(
    "labels",
    [
        [],
        [0, 0, 0],
        [1, 1],
        [0, 1, None],
        [0, 1, 2],
        pd.Series([0, 1, pd.NA], dtype="Int64"),
    ],
)
def test_invalid_training_labels_are_rejected(labels: object) -> None:
    with pytest.raises(ValueError):
        fit_prevalence_baseline(labels)


def test_negative_prediction_count_is_rejected() -> None:
    model = fit_prevalence_baseline([0, 1])

    with pytest.raises(ValueError):
        predict_event_probability(model, number_of_flights=-1)


def test_unfitted_model_is_rejected() -> None:
    from sklearn.dummy import DummyClassifier

    with pytest.raises(ValueError):
        predict_event_probability(
            DummyClassifier(strategy="prior"),
            number_of_flights=2,
        )
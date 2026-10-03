"""Simple, training-only baselines for Phase 4 classification tasks.

This module does not read project datasets or decide temporal splits.
The caller must pass labels from the development/training period only.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


def fit_prevalence_baseline(
    training_labels: Sequence[int] | pd.Series | np.ndarray,
) -> DummyClassifier:
    """Fit a constant-probability baseline to binary training labels.

    The fitted model predicts the majority class as a hard label and
    returns the training event prevalence as its class-1 probability.

    Missing labels must be excluded by the caller according to the
    documented target cohort; they are never silently treated as zero.
    """
    labels = pd.Series(training_labels).reset_index(drop=True)

    if labels.empty:
        raise ValueError("Training labels cannot be empty.")

    numeric_labels = pd.to_numeric(labels, errors="coerce")

    if numeric_labels.isna().any():
        raise ValueError("Training labels must not contain missing or invalid values.")

    if not numeric_labels.isin([0, 1]).all():
        raise ValueError("Training labels must contain only binary values 0 and 1.")

    if numeric_labels.nunique() != 2:
        raise ValueError("Training labels must contain both classes, 0 and 1.")

    # DummyClassifier requires a feature matrix, but this baseline does
    # not learn from flight features. One placeholder column is sufficient.
    placeholder_features = np.zeros((len(numeric_labels), 1), dtype=np.uint8)

    model = DummyClassifier(strategy="prior")
    model.fit(placeholder_features, numeric_labels.to_numpy(dtype=np.int8))
    return model


def predict_event_probability(
    model: DummyClassifier,
    number_of_flights: int,
) -> np.ndarray:
    """Return the baseline probability of class 1 for each flight."""
    if isinstance(number_of_flights, bool) or not isinstance(
        number_of_flights, int
    ):
        raise TypeError("number_of_flights must be an integer.")

    if number_of_flights < 0:
        raise ValueError("number_of_flights cannot be negative.")

    if not hasattr(model, "classes_"):
        raise ValueError("The baseline must be fitted before prediction.")

    positive_class_indices = np.flatnonzero(model.classes_ == 1)
    if len(positive_class_indices) != 1:
        raise ValueError("The fitted baseline does not contain class 1.")

    placeholder_features = np.zeros((number_of_flights, 1), dtype=np.uint8)
    probabilities = model.predict_proba(placeholder_features)

    return probabilities[:, positive_class_indices[0]]
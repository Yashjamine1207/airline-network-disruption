"""Tests for the Phase 4 base-feature contract and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    build_base_preprocessor,
    select_base_features,
)


def make_predictors() -> pd.DataFrame:
    """Small synthetic table using the real Phase 3 column names."""
    data: dict[str, object] = {
        column: [1.0, 3.0, np.nan]
        for column in BASE_NUMERIC_COLUMNS
    }

    data.update(
        {
            column: ["A", "B", None]
            for column in BASE_CATEGORICAL_COLUMNS
        }
    )
    data["source_row_number"] = [10, 11, 12]
    data["prior_month_origin_out_degree"] = [5, 6, 7]
    return pd.DataFrame(data)


def test_selector_keeps_only_approved_base_features() -> None:
    selected = select_base_features(make_predictors())

    assert list(selected.columns) == [
        *BASE_NUMERIC_COLUMNS,
        *BASE_CATEGORICAL_COLUMNS,
    ]
    assert selected.loc[2, "origin_airport"] == "__MISSING__"


def test_selector_rejects_outcome_columns() -> None:
    predictors = make_predictors()
    predictors["severe_delay_120"] = [0, 1, 0]

    with pytest.raises(ValueError, match="Outcome columns"):
        select_base_features(predictors)


def test_selector_rejects_missing_required_column() -> None:
    predictors = make_predictors().drop(columns="route")

    with pytest.raises(ValueError, match="route"):
        select_base_features(predictors)


def test_selector_rejects_infinite_numeric_value() -> None:
    predictors = make_predictors()
    predictors.loc[0, "distance_miles"] = np.inf

    with pytest.raises(ValueError, match="distance_miles"):
        select_base_features(predictors)


def test_transformer_fits_training_statistics_only() -> None:
    training = select_base_features(make_predictors())

    validation = training.iloc[[0]].copy()
    validation["scheduled_departure_year"] = np.nan
    validation["origin_airport"] = "UNSEEN_AIRPORT"

    transformer = build_base_preprocessor()
    transformer.fit(training)

    train_matrix = transformer.transform(training)
    validation_matrix = transformer.transform(validation)

    # The training values [1, 3, missing] have median 2.
    imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 2.0
    assert train_matrix.shape[0] == 3
    assert validation_matrix.shape[0] == 1
    assert validation_matrix.shape[1] == train_matrix.shape[1]
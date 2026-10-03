"""Tests for airline_disruption.features.predictor_table (Phase 6, Step 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.features.predictor_table import read_predictor_csv


@pytest.fixture()
def csv_path(tmp_path):
    frame = pd.DataFrame(
        {
            "source_row_number": [1, 2, 3, 4, 5, 6, 7],
            "prediction_timestamp_utc": [f"2019-01-0{d} 08:00:00+00:00" for d in range(1, 8)],
            "FL_DATE": [f"2019-01-0{d}" for d in range(1, 8)],
            "scheduled_departure_utc": [f"2019-01-0{d} 10:00:00+00:00" for d in range(1, 8)],
            # Different airports in each chunk of 3 rows, so category sets differ per chunk.
            "origin_airport": ["ATL", "ATL", "JFK", "LAX", "LAX", "SFO", "ATL"],
            "distance_miles": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
            # Integer in the first chunk, missing in the second: dtype must survive concat.
            "prior_count": [5, 6, 7, np.nan, 9, 10, 11],
            "extra_unused": list("abcdefg"),
        }
    )
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _read(path, chunksize):
    return read_predictor_csv(
        path,
        columns=[
            "source_row_number",
            "prediction_timestamp_utc",
            "FL_DATE",
            "scheduled_departure_utc",
            "origin_airport",
            "distance_miles",
            "prior_count",
        ],
        categorical_columns=["origin_airport"],
        float32_columns=["distance_miles"],
        chunksize=chunksize,
    )


def test_chunked_read_matches_single_read(csv_path) -> None:
    path, _ = csv_path
    chunked = _read(path, chunksize=3)
    single = _read(path, chunksize=1_000)
    pd.testing.assert_frame_equal(chunked, single)


def test_categories_are_unified_across_chunks(csv_path) -> None:
    path, original = csv_path
    frame = _read(path, chunksize=3)
    assert isinstance(frame["origin_airport"].dtype, pd.CategoricalDtype)
    assert set(frame["origin_airport"].cat.categories) == {"ATL", "JFK", "LAX", "SFO"}
    assert frame["origin_airport"].astype(str).tolist() == original["origin_airport"].tolist()


def test_types_are_compact_and_timezone_aware(csv_path) -> None:
    path, _ = csv_path
    frame = _read(path, chunksize=3)
    assert frame["distance_miles"].dtype == np.float32
    assert str(frame["prediction_timestamp_utc"].dt.tz) == "UTC"
    assert str(frame["scheduled_departure_utc"].dt.tz) == "UTC"
    assert pd.api.types.is_datetime64_dtype(frame["FL_DATE"])
    assert frame["FL_DATE"].dt.tz is None


def test_values_and_order_survive(csv_path) -> None:
    path, original = csv_path
    frame = _read(path, chunksize=3)
    assert frame["source_row_number"].tolist() == original["source_row_number"].tolist()
    assert frame["prior_count"].isna().tolist() == [False, False, False, True, False, False, False]
    assert frame["prior_count"].dtype == np.float64
    assert frame["scheduled_departure_utc"].iloc[0] == pd.Timestamp("2019-01-01 10:00:00", tz="UTC")


def test_only_requested_columns_are_returned(csv_path) -> None:
    path, _ = csv_path
    frame = _read(path, chunksize=3)
    assert "extra_unused" not in frame.columns
    assert list(frame.columns)[0] == "source_row_number"

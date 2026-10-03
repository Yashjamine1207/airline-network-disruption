"""Focused Phase 3 tests for timestamp conversion and predictor leakage."""

from pathlib import Path

import pandas as pd

from scripts.build_prediction_timestamps import process_chunk


ROOT = Path(__file__).resolve().parents[2]
PREDICTORS = ROOT / "data/features/base/base_predictors.csv"
LABELS = ROOT / "data/processed/targets/model_labels.csv"

TIMEZONES = {
    "JFK": "America/New_York",
    "LAS": "America/Los_Angeles",
}


def convert(date: str, clock: int, origin: str) -> pd.Series:
    """Convert one synthetic flight using the actual project function."""
    source = pd.DataFrame({
        "FL_DATE": [date],
        "CRS_DEP_TIME": [clock],
        "ORIGIN": [origin],
    })
    return process_chunk(source, 0, TIMEZONES).iloc[0]


def test_normal_departure_has_two_hour_prediction_horizon():
    row = convert("2019-01-01", 30, "JFK")

    assert row["timestamp_conversion_status"] == "ok"
    assert row["scheduled_departure_utc"] == "2019-01-01T05:30:00Z"
    assert row["prediction_timestamp_utc"] == "2019-01-01T03:30:00Z"


def test_2400_rolls_to_following_local_day():
    row = convert("2019-01-01", 2400, "JFK")

    assert row["timestamp_conversion_status"] == "ok"
    assert row["scheduled_departure_local"] == "2019-01-02T00:00:00"
    assert row["scheduled_departure_utc"] == "2019-01-02T05:00:00Z"
    assert row["prediction_timestamp_utc"] == "2019-01-02T03:00:00Z"


def test_spring_dst_nonexistent_time_is_not_guessed():
    row = convert("2019-03-10", 230, "JFK")

    assert row["timestamp_conversion_status"] == "dst_nonexistent"
    assert pd.isna(row["prediction_timestamp_utc"])


def test_autumn_dst_ambiguous_time_is_not_guessed():
    row = convert("2019-11-03", 140, "LAS")

    assert row["timestamp_conversion_status"] == "dst_ambiguous"
    assert pd.isna(row["prediction_timestamp_utc"])


def test_unmapped_origin_is_not_guessed():
    row = convert("2019-01-01", 1200, "UNKNOWN")

    assert row["timestamp_conversion_status"] == "unmapped_origin_timezone"
    assert pd.isna(row["prediction_timestamp_utc"])


def test_predictor_file_excludes_labels_and_raw_outcomes():
    predictor_columns = set(pd.read_csv(PREDICTORS, nrows=0).columns)
    label_columns = set(pd.read_csv(LABELS, nrows=0).columns)

    prohibited = {
        "cancelled_target",
        "completed_flight",
        "arrival_delay_minutes",
        "severe_delay_60",
        "severe_delay_90",
        "severe_delay_120",
        "severe_delay_180",
        "ARR_DELAY",
        "CANCELLED",
        "DIVERTED",
        "DEP_DELAY",
        "CANCELLATION_CODE",
    }

    assert "prediction_timestamp_utc" in predictor_columns
    assert not (predictor_columns & prohibited)
    assert "severe_delay_120" in label_columns
    assert "cancelled_target" in label_columns
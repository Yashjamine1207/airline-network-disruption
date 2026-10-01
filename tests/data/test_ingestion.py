"""
Tests for flight-data ingestion and standardisation.

These tests verify:
- required raw-source fields are enforced;
- airport-timezone mappings are validated;
- raw source columns are renamed to canonical project fields;
- source row numbering is auditable;
- origin and destination timezones are joined correctly;
- missing timezone mappings remain visible rather than being guessed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from airline_disruption.data.ingestion import (
    REQUIRED_RAW_COLUMNS,
    load_airport_timezone_mapping,
    standardise_flight_chunk,
    validate_raw_flight_columns,
)


def build_valid_raw_flight_chunk() -> pd.DataFrame:
    """Create a minimal valid raw-source chunk for deterministic tests."""
    raw_record = {
        column_name: pd.NA
        for column_name in REQUIRED_RAW_COLUMNS
    }

    raw_record.update(
        {
            "FL_DATE": "2023-06-15",
            "AIRLINE": "Example Airline",
            "AIRLINE_DOT": "Example Airline Inc.",
            "AIRLINE_CODE": "EX",
            "DOT_CODE": 12345,
            "FL_NUMBER": 100,
            "ORIGIN": "JFK",
            "ORIGIN_CITY": "New York, NY",
            "DEST": "ORD",
            "DEST_CITY": "Chicago, IL",
            "CRS_DEP_TIME": 900,
            "CRS_ARR_TIME": 1030,
            "CANCELLED": 0.0,
            "DIVERTED": 0.0,
            "CRS_ELAPSED_TIME": 150.0,
        }
    )

    return pd.DataFrame([raw_record])


def build_valid_airport_timezone_mapping() -> pd.DataFrame:
    """Create a controlled airport timezone mapping for tests."""
    return pd.DataFrame(
        {
            "airport_code": ["JFK", "ORD"],
            "airport_iana_timezone": [
                "America/New_York",
                "America/Chicago",
            ],
            "airport_timezone_mapping_status": [
                "mapped_validated",
                "mapped_validated",
            ],
        }
    )


def test_validate_raw_flight_columns_accepts_complete_schema() -> None:
    """A complete raw source schema should pass validation."""
    raw_chunk = build_valid_raw_flight_chunk()

    validate_raw_flight_columns(raw_chunk.columns)


def test_validate_raw_flight_columns_rejects_missing_field() -> None:
    """A missing source field must stop ingestion explicitly."""
    raw_chunk = build_valid_raw_flight_chunk().drop(
        columns=["FL_DATE"]
    )

    with pytest.raises(ValueError, match="FL_DATE"):
        validate_raw_flight_columns(raw_chunk.columns)


def test_standardise_flight_chunk_renames_and_joins_timezones() -> None:
    """Raw fields should be standardised and joined to both airport timezones."""
    raw_chunk = build_valid_raw_flight_chunk()
    timezone_mapping = build_valid_airport_timezone_mapping()

    result = standardise_flight_chunk(
        raw_chunk=raw_chunk,
        airport_timezone_mapping=timezone_mapping,
        source_row_start=100,
    )

    assert result.loc[0, "source_row_number"] == 100
    assert result.loc[0, "flight_date_local"] == pd.Timestamp(
    "2023-06-15"
)

    assert result.loc[0, "origin_airport"] == "JFK"
    assert result.loc[0, "destination_airport"] == "ORD"

    assert result.loc[0, "origin_iana_timezone"] == "America/New_York"
    assert result.loc[0, "destination_iana_timezone"] == "America/Chicago"

    assert bool(result.loc[0, "origin_timezone_found"])
    assert bool(result.loc[0, "destination_timezone_found"])

    assert "FL_DATE" not in result.columns
    assert "flight_date_local" in result.columns
    assert "scheduled_departure_time_local_hhmm" in result.columns
    assert "scheduled_arrival_time_local_hhmm" in result.columns


def test_standardise_flight_chunk_preserves_missing_timezone_visibility() -> None:
    """A missing airport mapping must remain explicit in the output."""
    raw_chunk = build_valid_raw_flight_chunk()

    raw_chunk.loc[
        0,
        "ORIGIN",
    ] = "ZZZ"

    timezone_mapping = build_valid_airport_timezone_mapping()

    result = standardise_flight_chunk(
        raw_chunk=raw_chunk,
        airport_timezone_mapping=timezone_mapping,
        source_row_start=0,
    )

    assert not bool(result.loc[0, "origin_timezone_found"])
    assert pd.isna(result.loc[0, "origin_iana_timezone"])

    assert bool(result.loc[0, "destination_timezone_found"])
    assert result.loc[0, "destination_iana_timezone"] == "America/Chicago"


def test_load_airport_timezone_mapping_rejects_duplicate_airports(
    tmp_path,
) -> None:
    """Duplicate airport codes would make timezone joins ambiguous."""
    mapping_path = tmp_path / "duplicate_airports.csv"

    pd.DataFrame(
        {
            "airport_iata_code": ["JFK", "JFK"],
            "iana_timezone": [
                "America/New_York",
                "America/New_York",
            ],
            "mapping_status": [
                "mapped_validated",
                "mapped_validated",
            ],
        }
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_airport_timezone_mapping(mapping_path)


def test_load_airport_timezone_mapping_rejects_unresolved_airport(
    tmp_path,
) -> None:
    """Unresolved airport timezones must stop ingestion."""
    mapping_path = tmp_path / "unresolved_airport.csv"

    pd.DataFrame(
        {
            "airport_iata_code": ["JFK"],
            "iana_timezone": [pd.NA],
            "mapping_status": ["requires_review"],
        }
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="unresolved"):
        load_airport_timezone_mapping(mapping_path)
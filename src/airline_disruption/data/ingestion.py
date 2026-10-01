"""
Flight-data ingestion and standardisation utilities.

This module reads the immutable raw Kaggle CSV in chunks, applies a controlled
raw-to-canonical field mapping, joins the project airport-timezone mapping, and
writes standardised Parquet partitions.

It does not create targets, historical features, leakage-prone aggregations,
weather joins, model inputs, or predictions.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


RAW_TO_CANONICAL_COLUMN_MAP = {
    "FL_DATE": "flight_date_local",
    "AIRLINE": "airline_name",
    "AIRLINE_DOT": "airline_dot_name",
    "AIRLINE_CODE": "reporting_airline_code",
    "DOT_CODE": "reporting_airline_dot_id",
    "FL_NUMBER": "flight_number",
    "ORIGIN": "origin_airport",
    "ORIGIN_CITY": "origin_city",
    "DEST": "destination_airport",
    "DEST_CITY": "destination_city",
    "CRS_DEP_TIME": "scheduled_departure_time_local_hhmm",
    "DEP_TIME": "actual_departure_time_local_hhmm",
    "DEP_DELAY": "departure_delay_minutes",
    "TAXI_OUT": "taxi_out_minutes",
    "WHEELS_OFF": "wheels_off_time_local_hhmm",
    "WHEELS_ON": "wheels_on_time_local_hhmm",
    "TAXI_IN": "taxi_in_minutes",
    "CRS_ARR_TIME": "scheduled_arrival_time_local_hhmm",
    "ARR_TIME": "actual_arrival_time_local_hhmm",
    "ARR_DELAY": "arrival_delay_minutes",
    "CANCELLED": "cancelled_indicator",
    "CANCELLATION_CODE": "cancellation_reason_code",
    "DIVERTED": "diverted_indicator",
    "CRS_ELAPSED_TIME": "scheduled_elapsed_minutes",
    "ELAPSED_TIME": "actual_elapsed_minutes",
    "AIR_TIME": "air_time_minutes",
    "DISTANCE": "distance_miles",
    "DELAY_DUE_CARRIER": "delay_due_carrier_minutes",
    "DELAY_DUE_WEATHER": "delay_due_weather_minutes",
    "DELAY_DUE_NAS": "delay_due_national_air_system_minutes",
    "DELAY_DUE_SECURITY": "delay_due_security_minutes",
    "DELAY_DUE_LATE_AIRCRAFT": "delay_due_late_aircraft_minutes",
}


REQUIRED_RAW_COLUMNS = tuple(RAW_TO_CANONICAL_COLUMN_MAP)


def validate_raw_flight_columns(
    raw_columns: pd.Index,
) -> None:
    """
    Confirm that all required raw fields are present before ingestion.

    Parameters
    ----------
    raw_columns:
        Column names read from the raw CSV header.

    Raises
    ------
    ValueError
        If a required source column is absent.
    """
    missing_columns = sorted(
        set(REQUIRED_RAW_COLUMNS) - set(raw_columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw flight source is missing required columns: "
            f"{missing_columns}"
        )


def load_airport_timezone_mapping(
    airport_timezone_mapping_path: Path,
) -> pd.DataFrame:
    """
    Load and validate the controlled airport-to-IANA-timezone mapping.

    Parameters
    ----------
    airport_timezone_mapping_path:
        CSV path containing project airport reference data.

    Returns
    -------
    pd.DataFrame
        Airport code and validated IANA timezone columns.

    Raises
    ------
    ValueError
        If required mapping fields are missing, duplicated, or incomplete.
    """
    mapping = pd.read_csv(airport_timezone_mapping_path)

    required_mapping_columns = {
        "airport_iata_code",
        "iana_timezone",
        "mapping_status",
    }

    missing_mapping_columns = sorted(
        required_mapping_columns - set(mapping.columns)
    )

    if missing_mapping_columns:
        raise ValueError(
            "Airport timezone mapping is missing required columns: "
            f"{missing_mapping_columns}"
        )

    mapping = mapping[
        [
            "airport_iata_code",
            "iana_timezone",
            "mapping_status",
        ]
    ].copy()

    mapping["airport_iata_code"] = (
        mapping["airport_iata_code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    mapping["iana_timezone"] = (
        mapping["iana_timezone"]
        .astype("string")
        .str.strip()
    )

    duplicate_airports = mapping.loc[
        mapping["airport_iata_code"].duplicated(
            keep=False
        ),
        "airport_iata_code",
    ].unique()

    if len(duplicate_airports) > 0:
        raise ValueError(
            "Airport timezone mapping contains duplicate airport codes: "
            f"{sorted(duplicate_airports.tolist())}"
        )

    unresolved_airports = mapping.loc[
        mapping["iana_timezone"].isna()
        | mapping["iana_timezone"].eq(""),
        "airport_iata_code",
    ].tolist()

    if unresolved_airports:
        raise ValueError(
            "Airport timezone mapping contains unresolved airports: "
            f"{sorted(unresolved_airports)}"
        )

    return mapping.rename(
        columns={
            "airport_iata_code": "airport_code",
            "iana_timezone": "airport_iana_timezone",
            "mapping_status": "airport_timezone_mapping_status",
        }
    )


def standardise_flight_chunk(
    raw_chunk: pd.DataFrame,
    airport_timezone_mapping: pd.DataFrame,
    source_row_start: int,
) -> pd.DataFrame:
    """
    Standardise one raw flight-data chunk and join airport timezones.

    Parameters
    ----------
    raw_chunk:
        One raw CSV chunk using the source schema.
    airport_timezone_mapping:
        Validated airport timezone mapping from the project reference file.
    source_row_start:
        Zero-based raw source row position of the first record in this chunk.

    Returns
    -------
    pd.DataFrame
        Standardised flight records with source-row and timezone audit fields.
    """
    validate_raw_flight_columns(raw_chunk.columns)

    standardised_chunk = raw_chunk.rename(
        columns=RAW_TO_CANONICAL_COLUMN_MAP
    ).copy()

    standardised_chunk.insert(
        0,
        "source_row_number",
        range(
            source_row_start,
            source_row_start + len(standardised_chunk),
        ),
    )

    standardised_chunk["flight_date_local"] = pd.to_datetime(
        standardised_chunk["flight_date_local"],
        errors="coerce",
    ).dt.normalize()

    standardised_chunk["origin_airport"] = (
        standardised_chunk["origin_airport"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    standardised_chunk["destination_airport"] = (
        standardised_chunk["destination_airport"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    origin_timezone_mapping = airport_timezone_mapping.rename(
        columns={
            "airport_code": "origin_airport",
            "airport_iana_timezone": "origin_iana_timezone",
            "airport_timezone_mapping_status": (
                "origin_timezone_mapping_status"
            ),
        }
    )

    destination_timezone_mapping = airport_timezone_mapping.rename(
        columns={
            "airport_code": "destination_airport",
            "airport_iana_timezone": "destination_iana_timezone",
            "airport_timezone_mapping_status": (
                "destination_timezone_mapping_status"
            ),
        }
    )

    standardised_chunk = standardised_chunk.merge(
        origin_timezone_mapping,
        on="origin_airport",
        how="left",
        validate="many_to_one",
    )

    standardised_chunk = standardised_chunk.merge(
        destination_timezone_mapping,
        on="destination_airport",
        how="left",
        validate="many_to_one",
    )

    standardised_chunk["origin_timezone_found"] = (
        standardised_chunk["origin_iana_timezone"].notna()
    )

    standardised_chunk["destination_timezone_found"] = (
        standardised_chunk["destination_iana_timezone"].notna()
    )

    return standardised_chunk


def iter_standardised_flight_chunks(
    raw_flights_csv_path: Path,
    airport_timezone_mapping: pd.DataFrame,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """
    Yield standardised raw-flight chunks without loading the full CSV.

    Parameters
    ----------
    raw_flights_csv_path:
        Immutable raw Kaggle CSV.
    airport_timezone_mapping:
        Controlled airport-to-timezone mapping.
    chunk_size:
        Number of raw rows per chunk.

    Yields
    ------
    pd.DataFrame
        Standardised flight chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    source_row_start = 0

    for raw_chunk in pd.read_csv(
        raw_flights_csv_path,
        chunksize=chunk_size,
        low_memory=False,
    ):
        standardised_chunk = standardise_flight_chunk(
            raw_chunk=raw_chunk,
            airport_timezone_mapping=airport_timezone_mapping,
            source_row_start=source_row_start,
        )

        yield standardised_chunk

        source_row_start += len(raw_chunk)
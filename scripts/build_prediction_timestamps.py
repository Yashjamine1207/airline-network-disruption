"""
Build auditable pre-flight prediction timestamps.

Uses:
- FL_DATE, CRS_DEP_TIME, and ORIGIN from the immutable flight CSV.
- The existing validated airport-to-IANA-timezone mapping.

Output:
- One row per source flight.
- Original local date/time preserved.
- Scheduled departure in origin-local time and UTC.
- Prediction timestamp = scheduled departure UTC minus two hours.
- Explicit status for invalid, unmapped, and DST-problem rows.

No flight outcomes are read. The raw file and mapping are not modified.
"""

from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FLIGHTS = ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"
MAPPING = ROOT / "data/external/airport_timezone_mapping.csv"
OUTPUT = ROOT / "data/interim/flight_prediction_timestamps.csv"
TEMP_OUTPUT = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000


def load_timezone_mapping() -> dict[str, str]:
    """Load only validated, unambiguous airport-timezone assignments."""
    if not MAPPING.is_file():
        raise FileNotFoundError(f"Timezone mapping not found: {MAPPING}")

    mapping = pd.read_csv(MAPPING, dtype="string")

    required = {
        "airport_iata_code",
        "iana_timezone",
        "mapping_status",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Timezone mapping lacks: {sorted(missing)}")

    mapping["airport_iata_code"] = (
        mapping["airport_iata_code"].str.strip().str.upper()
    )
    mapping["iana_timezone"] = mapping["iana_timezone"].str.strip()

    validated = mapping.loc[
        mapping["mapping_status"].isin(
            ["mapped_validated", "mapped_validated_manual_override"]
        )
        & mapping["airport_iata_code"].notna()
        & mapping["iana_timezone"].notna(),
        ["airport_iata_code", "iana_timezone"],
    ].copy()

    duplicates = validated["airport_iata_code"].duplicated(keep=False)
    if duplicates.any():
        airports = validated.loc[
            duplicates, "airport_iata_code"
        ].unique().tolist()
        raise ValueError(
            f"Duplicate validated airport mappings: {airports[:10]}"
        )

    if validated.empty:
        raise ValueError("No mapped_validated airports found.")

    # Check the IANA names before processing three million flight rows.
    for name in validated["iana_timezone"].unique():
        try:
            ZoneInfo(str(name))
        except ZoneInfoNotFoundError as exc:
            raise RuntimeError(
                f"IANA timezone data unavailable for {name!r}. "
                "On Windows, install it with: python -m pip install tzdata"
            ) from exc

    result = dict(
        zip(
            validated["airport_iata_code"].astype(str),
            validated["iana_timezone"].astype(str),
        )
    )
    print(f"Validated airport timezones loaded: {len(result):,}")
    return result


def process_chunk(
    chunk: pd.DataFrame,
    first_row_number: int,
    airport_timezones: dict[str, str],
) -> pd.DataFrame:
    """Convert one chunk of source scheduled departures to UTC."""
    chunk = chunk.reset_index(drop=True)
    count = len(chunk)
    index = chunk.index

    origins = chunk["ORIGIN"].astype("string").str.strip().str.upper()
    zones = origins.map(airport_timezones).astype("string")

    dates = pd.to_datetime(
        chunk["FL_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    clock = pd.to_numeric(chunk["CRS_DEP_TIME"], errors="coerce")
    integer_clock = clock.notna() & clock.eq(np.floor(clock))

    hour = np.floor(clock / 100)
    minute = clock % 100

    normal_clock = (
        integer_clock
        & hour.between(0, 23)
        & minute.between(0, 59)
    )
    midnight_2400 = integer_clock & clock.eq(2400)
    valid_clock = normal_clock | midnight_2400

    # The configured 2400 rule means midnight on the FOLLOWING
    # origin-local calendar day; preserve the original source value.
    safe_hour = hour.where(normal_clock, 0)
    safe_minute = minute.where(normal_clock, 0)

    local_departure = (
        dates
        + pd.to_timedelta(safe_hour, unit="h")
        + pd.to_timedelta(safe_minute, unit="m")
        + pd.to_timedelta(midnight_2400.astype("int8"), unit="D")
    )
    local_departure = local_departure.where(
        dates.notna() & valid_clock
    )

    status = pd.Series("ok", index=index, dtype="string")
    status.loc[dates.isna()] = "invalid_flight_date"
    status.loc[dates.notna() & ~valid_clock] = "invalid_clock_time"
    status.loc[
        dates.notna() & valid_clock & zones.isna()
    ] = "unmapped_origin_timezone"

    utc_departure = pd.Series(
        pd.NaT,
        index=index,
        dtype="datetime64[ns, UTC]",
    )
    dst_ambiguous = pd.Series(False, index=index, dtype="bool")
    dst_nonexistent = pd.Series(False, index=index, dtype="bool")

    eligible = status.eq("ok")

    # Work by timezone, not row by row. This keeps the full run practical.
    for timezone_name, row_index in zones.loc[eligible].groupby(
        zones.loc[eligible]
    ).groups.items():
        local_group = local_departure.loc[row_index]

        localized = local_group.dt.tz_localize(
            str(timezone_name),
            ambiguous="NaT",
            nonexistent="NaT",
        )

        utc_departure.loc[row_index] = (
            localized.dt.tz_convert("UTC")
        )

        # Investigate only times that could not be localized.
        failed_index = localized.index[localized.isna()]
        if len(failed_index) == 0:
            continue

        failed_local = local_departure.loc[failed_index]

        ambiguous = failed_local.dt.tz_localize(
            str(timezone_name),
            ambiguous="NaT",
            nonexistent="shift_forward",
        ).isna()

        nonexistent = failed_local.dt.tz_localize(
            str(timezone_name),
            ambiguous=True,
            nonexistent="NaT",
        ).isna()

        dst_ambiguous.loc[failed_index] = ambiguous
        dst_nonexistent.loc[failed_index] = nonexistent

        status.loc[failed_index[ambiguous.to_numpy()]] = (
            "dst_ambiguous"
        )
        status.loc[failed_index[nonexistent.to_numpy()]] = (
            "dst_nonexistent"
        )

    prediction_time = utc_departure - pd.Timedelta(hours=2)

    output = pd.DataFrame({
        "source_row_number": np.arange(
            first_row_number,
            first_row_number + count,
            dtype=np.int64,
        ),
        "FL_DATE": chunk["FL_DATE"],
        "CRS_DEP_TIME": chunk["CRS_DEP_TIME"],
        "ORIGIN": chunk["ORIGIN"],
        "origin_iana_timezone": zones,
        "scheduled_departure_local": local_departure.dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "scheduled_departure_utc": utc_departure.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "prediction_timestamp_utc": prediction_time.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "clock_value_2400": midnight_2400,
        "dst_ambiguous": dst_ambiguous,
        "dst_nonexistent": dst_nonexistent,
        "timestamp_conversion_status": status,
        "timestamp_code_version": "phase3_prediction_timestamps_v1",
    })

    if output.loc[
        output["timestamp_conversion_status"].eq("ok"),
        "prediction_timestamp_utc",
    ].isna().any():
        raise ValueError(
            "An 'ok' row has no prediction timestamp."
        )

    return output


def main() -> None:
    """Stream the flight file and atomically publish the finished output."""
    if not FLIGHTS.is_file():
        raise FileNotFoundError(f"Flight file not found: {FLIGHTS}")

    airport_timezones = load_timezone_mapping()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    total_rows = 0
    status_counts: Counter[str] = Counter()

    try:
        reader = pd.read_csv(
            FLIGHTS,
            usecols=["FL_DATE", "CRS_DEP_TIME", "ORIGIN"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            result = process_chunk(
                chunk,
                total_rows,
                airport_timezones,
            )
            result.to_csv(
                TEMP_OUTPUT,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )

            total_rows += len(result)
            status_counts.update(
                result["timestamp_conversion_status"].value_counts()
                .to_dict()
            )
            print(
                f"Processed chunk {chunk_number}: "
                f"{total_rows:,} flights"
            )

        if total_rows != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; got {total_rows:,}."
            )

        os.replace(TEMP_OUTPUT, OUTPUT)

    except Exception:
        TEMP_OUTPUT.unlink(missing_ok=True)
        raise

    print(f"\nFinished: {OUTPUT}")
    print(f"Rows: {total_rows:,}")
    print("Conversion status:")
    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count:,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

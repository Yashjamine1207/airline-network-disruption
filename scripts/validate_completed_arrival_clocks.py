"""Cross-check candidate gate-arrival UTC against destination ARR_TIME.

Produces a separate audited table. Does not alter target labels or
the original candidate outcome-time table. Final-test rows stay locked.
"""

from itertools import zip_longest
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"
CANDIDATES = ROOT / "data/interim/completed_outcome_times.csv"
MAPPING = ROOT / "data/external/airport_timezone_mapping.csv"
OUTPUT = ROOT / "data/interim/validated_completed_arrivals.csv"
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000


def load_zones() -> dict[str, str]:
    mapping = pd.read_csv(MAPPING, dtype="string")

    approved = mapping.loc[
        mapping["mapping_status"].isin([
            "mapped_validated",
            "mapped_validated_manual_override",
        ])
    ]

    if approved["airport_iata_code"].duplicated().any():
        raise ValueError("Duplicate validated airport mappings.")

    return dict(zip(
        approved["airport_iata_code"].astype(str),
        approved["iana_timezone"].astype(str),
    ))


def validate_chunk(raw, candidate, offset, zones):
    raw = raw.reset_index(drop=True)
    candidate = candidate.reset_index(drop=True)

    if len(raw) != len(candidate):
        raise ValueError("Raw and candidate chunk sizes differ.")

    ids = pd.to_numeric(
        candidate["source_row_number"], errors="coerce"
    ).to_numpy()
    if not np.array_equal(
        ids, np.arange(offset, offset + len(raw))
    ):
        raise ValueError("Raw and candidate rows are misaligned.")

    status = candidate["outcome_time_status"].astype(
        "string"
    ).copy()
    eligible = status.eq("usable_candidate")

    utc = pd.to_datetime(
        candidate["candidate_gate_arrival_utc"],
        utc=True,
        errors="coerce",
    )
    if utc.loc[eligible].isna().any():
        raise ValueError("Usable candidate has invalid UTC time.")

    destination_zone = (
        raw["DEST"].astype("string").map(zones)
    )
    reported = pd.to_numeric(
        raw["ARR_TIME"], errors="coerce"
    )

    hour = np.floor(reported / 100)
    minute = reported % 100
    valid_clock = (
        reported.notna()
        & (
            (
                hour.between(0, 23)
                & minute.between(0, 59)
            )
            | reported.eq(2400)
        )
    )

    matches = pd.Series(
        False, index=raw.index, dtype="bool"
    )

    candidates_with_zone = eligible & destination_zone.notna()

    for zone, row_index in destination_zone.loc[
        candidates_with_zone
    ].groupby(
        destination_zone.loc[candidates_with_zone]
    ).groups.items():
        local = utc.loc[row_index].dt.tz_convert(
            str(zone)
        )

        expected_minutes = (
            local.dt.hour * 60 + local.dt.minute
        )
        reported_minutes = (
            hour.loc[row_index] * 60
            + minute.loc[row_index]
        ).where(
            ~reported.loc[row_index].eq(2400),
            0,
        )

        difference = (
            expected_minutes - reported_minutes
        ).abs()
        circular_difference = pd.concat(
            [difference, 1440 - difference],
            axis=1,
        ).min(axis=1)

        matches.loc[row_index] = (
            valid_clock.loc[row_index]
            & circular_difference.le(1)
        ).fillna(False)

    status.loc[
        eligible & destination_zone.isna()
    ] = "unmapped_destination_timezone"

    status.loc[
        eligible & ~valid_clock
    ] = "invalid_reported_arrival_clock"

    status.loc[
        eligible
        & destination_zone.notna()
        & valid_clock
        & ~matches
    ] = "arrival_clock_mismatch"

    status.loc[
        eligible & matches
    ] = "validated_event_time"

    validated_utc = utc.where(
        status.eq("validated_event_time")
    )

    return pd.DataFrame({
        "source_row_number": ids.astype("int64"),
        "validated_gate_arrival_utc": (
            validated_utc.dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        ),
        "arrival_time_validation_status": status,
        "arrival_time_validation_version": (
            "phase3_arrival_clock_v1"
        ),
    })


def main():
    for path in (RAW, CANDIDATES, MAPPING):
        if not path.is_file():
            raise FileNotFoundError(path)

    zones = load_zones()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    raw_reader = pd.read_csv(
        RAW,
        usecols=["DEST", "ARR_TIME"],
        chunksize=CHUNK_SIZE,
    )
    candidate_reader = pd.read_csv(
        CANDIDATES,
        usecols=[
            "source_row_number",
            "candidate_gate_arrival_utc",
            "outcome_time_status",
        ],
        chunksize=CHUNK_SIZE,
    )

    total = 0
    counts = {}

    try:
        for number, pair in enumerate(
            zip_longest(
                raw_reader, candidate_reader
            ),
            start=1,
        ):
            if pair[0] is None or pair[1] is None:
                raise ValueError(
                    "Input files have unequal row counts."
                )

            result = validate_chunk(
                pair[0], pair[1], total, zones
            )
            result.to_csv(
                TEMP,
                mode="w" if number == 1 else "a",
                header=(number == 1),
                index=False,
            )

            for name, count in result[
                "arrival_time_validation_status"
            ].value_counts().items():
                counts[name] = (
                    counts.get(name, 0) + int(count)
                )

            total += len(result)
            print(f"Validated {total:,} source rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; "
                f"found {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nDestination-clock validation saved.")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count:,}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
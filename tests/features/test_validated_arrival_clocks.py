"""Test destination-clock checks and final-test protection."""

import pandas as pd

from scripts.validate_completed_arrival_clocks import (
    validate_chunk,
)


ZONES = {
    "JFK": "America/New_York",
    "LAS": "America/Los_Angeles",
}


def check(
    destination,
    reported_hhmm,
    candidate_utc,
    status="usable_candidate",
):
    raw = pd.DataFrame({
        "DEST": [destination],
        "ARR_TIME": [reported_hhmm],
    })
    candidate = pd.DataFrame({
        "source_row_number": [0],
        "candidate_gate_arrival_utc": [candidate_utc],
        "outcome_time_status": [status],
    })
    return validate_chunk(raw, candidate, 0, ZONES).iloc[0]


def test_matching_destination_clock_is_validated():
    # January 1 at 19:00 UTC is 14:00 in New York.
    row = check("JFK", 1400, "2019-01-01T19:00:00Z")

    assert row["arrival_time_validation_status"] == (
        "validated_event_time"
    )
    assert row["validated_gate_arrival_utc"] == (
        "2019-01-01T19:00:00Z"
    )


def test_one_minute_difference_is_within_tolerance():
    row = check("JFK", 1401, "2019-01-01T19:00:00Z")

    assert row["arrival_time_validation_status"] == (
        "validated_event_time"
    )


def test_larger_difference_is_excluded():
    row = check("JFK", 1402, "2019-01-01T19:00:00Z")

    assert row["arrival_time_validation_status"] == (
        "arrival_clock_mismatch"
    )
    assert pd.isna(row["validated_gate_arrival_utc"])


def test_utc_to_local_handles_dst_fallback():
    # The UTC instant selects the second occurrence of 01:40 in LA.
    row = check("LAS", 140, "2019-11-03T09:40:00Z")

    assert row["arrival_time_validation_status"] == (
        "validated_event_time"
    )


def test_locked_final_test_stays_locked():
    row = check(
        "JFK",
        1400,
        None,
        status="final_test_locked",
    )

    assert row["arrival_time_validation_status"] == (
        "final_test_locked"
    )
    assert pd.isna(row["validated_gate_arrival_utc"])
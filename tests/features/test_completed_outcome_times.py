"""Test candidate outcome times and final-test protection."""

import pandas as pd

from scripts.build_completed_outcome_times import process_chunk


SCHEDULED_UTC = "2019-01-01T16:00:00Z"


def make_row(
    *,
    date="2019-01-01",
    cancelled=0,
    diverted=0,
    arrival_delay=60,
    departure_delay=30,
    elapsed=150,
    scheduled_elapsed=120,
    scheduled_utc=SCHEDULED_UTC,
    conversion_status="ok",
):
    raw = pd.DataFrame({
        "FL_DATE": [date],
        "CANCELLED": [cancelled],
        "DIVERTED": [diverted],
        "ARR_DELAY": [arrival_delay],
        "DEP_DELAY": [departure_delay],
        "ELAPSED_TIME": [elapsed],
        "CRS_ELAPSED_TIME": [scheduled_elapsed],
    })

    times = pd.DataFrame({
        "source_row_number": [0],
        "scheduled_departure_utc": [scheduled_utc],
        "timestamp_conversion_status": [conversion_status],
    })

    return process_chunk(raw, times, offset=0).iloc[0]


def test_consistent_completed_flight_gets_gate_arrival_time():
    row = make_row()

    assert row["outcome_time_status"] == "usable_candidate"
    assert row["candidate_gate_arrival_utc"] == (
        "2019-01-01T19:00:00Z"
    )


def test_inconsistent_timing_does_not_get_outcome_time():
    row = make_row(arrival_delay=200)

    assert row["outcome_time_status"] == (
        "inconsistent_timing_fields"
    )
    assert pd.isna(row["candidate_gate_arrival_utc"])


def test_cancelled_flight_does_not_get_arrival_time():
    row = make_row(cancelled=1)

    assert row["outcome_time_status"] == "cancelled_no_arrival"
    assert pd.isna(row["candidate_gate_arrival_utc"])


def test_ambiguous_scheduled_time_is_excluded():
    row = make_row(
        scheduled_utc=None,
        conversion_status="dst_ambiguous",
    )

    assert row["outcome_time_status"] == (
        "unresolved_scheduled_timestamp"
    )
    assert pd.isna(row["candidate_gate_arrival_utc"])


def test_final_test_outcome_stays_locked():
    row = make_row(date="2023-02-01")

    assert row["outcome_time_status"] == "final_test_locked"
    assert pd.isna(row["candidate_gate_arrival_utc"])
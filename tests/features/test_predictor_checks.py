"""Tests for airline_disruption.features.predictor_checks (Phase 6, Step 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.features.predictor_checks import (
    check_alignment,
    null_rate_table,
    to_utc,
    unseen_category_table,
    verify_prior_month_features,
)


# ---------------------------------------------------------------------------
# to_utc
# ---------------------------------------------------------------------------
def test_to_utc_parses_iso_strings_with_offsets() -> None:
    result = to_utc(pd.Series(["2022-01-01 06:50:00+00:00", "2022-01-01 06:50:00-05:00"]))
    assert str(result.dt.tz) == "UTC"
    assert result.iloc[1] == pd.Timestamp("2022-01-01 11:50:00", tz="UTC")


def test_to_utc_converts_other_timezones() -> None:
    local = pd.Series(pd.to_datetime(["2022-01-01 00:30:00"])).dt.tz_localize("America/New_York")
    assert to_utc(local).iloc[0] == pd.Timestamp("2022-01-01 05:30:00", tz="UTC")


def test_to_utc_refuses_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="Naive"):
        to_utc(pd.Series(pd.to_datetime(["2022-01-01 00:30:00"])))


# ---------------------------------------------------------------------------
# check_alignment
# ---------------------------------------------------------------------------
def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": [1, 2, 3],
            "prediction_timestamp_utc": pd.to_datetime(
                ["2019-01-01 08:00:00", "2019-01-01 09:00:00", "2019-01-02 10:00:00"], utc=True
            ),
            "flight_date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-02"]),
        }
    )


def _predictors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": [1, 2, 3],
            "prediction_timestamp_utc": ["2019-01-01 08:00:00+00:00", "2019-01-01 09:00:00+00:00", "2019-01-02 10:00:00+00:00"],
            "FL_DATE": ["2019-01-01", "2019-01-01", "2019-01-02"],
        }
    )


def test_alignment_passes_for_matching_tables() -> None:
    check_alignment(_cohort(), _predictors())


def test_alignment_ignores_row_order() -> None:
    check_alignment(_cohort(), _predictors().iloc[::-1].reset_index(drop=True))


def test_alignment_fails_when_an_id_is_missing() -> None:
    with pytest.raises(ValueError, match="same flights"):
        check_alignment(_cohort(), _predictors().iloc[:2])


def test_alignment_fails_on_duplicate_ids() -> None:
    predictors = _predictors()
    predictors.loc[2, "source_row_number"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        check_alignment(_cohort(), predictors)


def test_alignment_fails_on_a_different_prediction_timestamp() -> None:
    predictors = _predictors()
    predictors.loc[1, "prediction_timestamp_utc"] = "2019-01-01 09:01:00+00:00"
    with pytest.raises(ValueError, match="1 different prediction timestamps"):
        check_alignment(_cohort(), predictors)


def test_alignment_fails_on_a_different_flight_date() -> None:
    predictors = _predictors()
    predictors.loc[0, "FL_DATE"] = "2019-01-02"
    with pytest.raises(ValueError, match="1 different flight dates"):
        check_alignment(_cohort(), predictors)


# ---------------------------------------------------------------------------
# null_rate_table and unseen_category_table
# ---------------------------------------------------------------------------
def test_null_rate_table_by_split() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0], "b": [1, 2, 3, 4, 5]})
    split = pd.Series(["development", "development", "development", "validation", "final_test_locked"])
    table = null_rate_table(frame, split, ["a", "b"]).set_index("feature")
    assert table.loc["a", "null_rate_development"] == pytest.approx(1 / 3)
    assert table.loc["a", "null_rate_validation"] == 1.0
    assert table.loc["b", "null_rate_development"] == 0.0
    # The final-test split is not reported unless it is asked for.
    assert "null_rate_final_test_locked" not in table.columns


def test_unseen_category_table() -> None:
    frame = pd.DataFrame({"airport": ["A", "A", "B", "B", "C", "D", "A", "E"]})
    split = pd.Series(["development"] * 3 + ["validation"] * 4 + ["final_test_locked"])
    # development knows A, B. validation has B, C, D, A -> C and D are unseen (2 of 4 rows).
    table = unseen_category_table(frame, split, ["airport"]).iloc[0]
    assert table["categories_development"] == 2
    assert table["categories_validation"] == 4
    assert table["unseen_categories_in_validation"] == 2
    assert table["unseen_row_share_validation"] == pytest.approx(0.5)


def test_unseen_category_table_ignores_the_final_test_split() -> None:
    # "E" appears only in the final-test row. It must not change any validation number.
    frame = pd.DataFrame({"airport": ["A", "B", "E"]})
    split = pd.Series(["development", "validation", "final_test_locked"])
    table = unseen_category_table(frame, split, ["airport"]).iloc[0]
    assert table["unseen_categories_in_validation"] == 1  # B is unseen; E never counted


# ---------------------------------------------------------------------------
# verify_prior_month_features
# ---------------------------------------------------------------------------
def _schedule() -> pd.DataFrame:
    """Seven flights. January (UTC) is the first month, so only the three February rows are comparable.

    January:  A->X (C1), A->X again (C1), A->Y (C1), B->X (C2)
    February: A->X (C1), B->Y (C2), A->Z (C1)

    The repeated A->X flight makes "number of flights" differ from "number of
    distinct neighbours", so a degree cannot pass as a count.

    Hand-computed values for the three February rows, in order:
      carrier count      C1:3  C2:1  C1:3
      origin count       A:3   B:1   A:3
      route count        A-X:2 B-Y:0 A-Z:0
      origin out-degree  A:2 (X,Y)  B:1 (X)  A:2
      origin weighted    same as origin count: 3, 1, 3
      dest in-degree     X:2 (A,B)  Y:1 (A)  Z:0
      dest weighted      X:3  Y:1  Z:0
    """
    return pd.DataFrame(
        {
            "scheduled_departure_utc": [
                "2019-01-05 10:00:00+00:00",
                "2019-01-10 10:00:00+00:00",
                "2019-01-20 10:00:00+00:00",
                "2019-01-31 23:30:00+00:00",  # still January in UTC
                "2019-02-01 00:30:00+00:00",  # already February in UTC
                "2019-02-10 12:00:00+00:00",
                "2019-02-15 12:00:00+00:00",
            ],
            "carrier_identifier": ["C1", "C1", "C1", "C2", "C1", "C2", "C1"],
            "origin_airport": ["A", "A", "A", "B", "A", "B", "A"],
            "destination_airport": ["X", "X", "Y", "X", "X", "Y", "Z"],
            "route": ["A-X", "A-X", "A-Y", "B-X", "A-X", "B-Y", "A-Z"],
        }
    )


FEBRUARY_FIRST_ROW = 4  # index of the first February row in _schedule()


def _with_correct_stored_values(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    stored = {
        "prior_calendar_month_carrier_scheduled_flight_count": [3, 1, 3],
        "prior_calendar_month_origin_scheduled_flight_count": [3, 1, 3],
        "prior_calendar_month_route_scheduled_flight_count": [2, 0, 0],
        "prior_month_origin_out_degree": [2, 1, 2],
        "prior_month_origin_weighted_out_degree": [3, 1, 3],
        "prior_month_destination_in_degree": [2, 1, 0],
        "prior_month_destination_weighted_in_degree": [3, 1, 0],
    }
    for column, february_values in stored.items():
        # January rows have no previous month in the file; the stored value there is irrelevant.
        frame[column] = [np.nan] * FEBRUARY_FIRST_ROW + february_values
    return frame


def test_verification_matches_hand_computed_values() -> None:
    result = verify_prior_month_features(_with_correct_stored_values(_schedule())).set_index("feature")
    assert (result["comparable_rows"] == 3).all()
    assert (result["matches"] == 3).all()
    assert (result["mismatches"] == 0).all()
    assert (result["first_month_rows"] == 4).all()
    assert (result["first_month_stored_null"] == 4).all()
    assert len(result) == 7


def test_verification_detects_a_wrong_stored_value() -> None:
    frame = _with_correct_stored_values(_schedule())
    frame.loc[FEBRUARY_FIRST_ROW, "prior_calendar_month_origin_scheduled_flight_count"] = 99  # should be 3
    result = verify_prior_month_features(frame).set_index("feature")
    row = result.loc["prior_calendar_month_origin_scheduled_flight_count"]
    assert row["mismatches"] == 1
    assert row["matches"] == 2
    assert row["mismatch_rate"] == pytest.approx(1 / 3)
    # Other features are unaffected.
    assert result.loc["prior_calendar_month_route_scheduled_flight_count", "mismatches"] == 0


def test_verification_uses_the_utc_month_not_the_local_date() -> None:
    # Flight at 2019-01-31 23:30 UTC counts as January; flight at 2019-02-01 00:30 UTC as February.
    # If the check used any other month rule, the hand-computed values above would not match.
    result = verify_prior_month_features(_with_correct_stored_values(_schedule()))
    assert (result["mismatches"] == 0).all()


def test_verification_reports_stored_nulls_in_comparable_rows() -> None:
    frame = _with_correct_stored_values(_schedule())
    frame.loc[FEBRUARY_FIRST_ROW + 2, "prior_month_destination_in_degree"] = np.nan
    row = verify_prior_month_features(frame).set_index("feature").loc["prior_month_destination_in_degree"]
    assert row["stored_null_in_comparable_rows"] == 1
    assert row["mismatches"] == 0


def test_verification_treats_no_flights_last_month_as_zero() -> None:
    # Route B-Y had no January flights, so a stored 0 must match and a stored 5 must not.
    frame = _with_correct_stored_values(_schedule())
    assert verify_prior_month_features(frame).set_index("feature").loc["prior_calendar_month_route_scheduled_flight_count", "mismatches"] == 0
    frame.loc[FEBRUARY_FIRST_ROW + 1, "prior_calendar_month_route_scheduled_flight_count"] = 5
    assert verify_prior_month_features(frame).set_index("feature").loc["prior_calendar_month_route_scheduled_flight_count", "mismatches"] == 1

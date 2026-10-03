"""Tests for airline_disruption.validation.temporal_splits (Phase 6, Step 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.validation.temporal_splits import (
    REGIME_2019,
    REGIME_2020,
    REGIME_2021_2022,
    REGIME_2023,
    REGIME_OTHER,
    SPLIT_DEVELOPMENT,
    SPLIT_EMBARGOED,
    SPLIT_FINAL_TEST,
    SPLIT_VALIDATION,
    UtcSplitBoundaries,
    apply_utc_embargo,
    assign_regime,
    boundaries_from_config,
    chronology_ok,
    to_boolean,
)

BOUNDARIES = UtcSplitBoundaries(
    validation_start_utc=pd.Timestamp("2022-01-01T00:00:00Z"),
    final_test_start_utc=pd.Timestamp("2023-01-01T00:00:00Z"),
)


def _stamps(values: list[str]) -> pd.Series:
    return pd.Series(pd.to_datetime(values, utc=True))


def _run(labels: list[str], times: list[str]) -> list[str]:
    result = apply_utc_embargo(pd.Series(labels), _stamps(times), BOUNDARIES)
    return [str(v) for v in result]


# ---------------------------------------------------------------------------
# Embargo rule: every boundary case
# ---------------------------------------------------------------------------
def test_development_row_after_validation_start_is_embargoed() -> None:
    # Real pattern from the audit: local date 2021-12-31, UTC prediction time 2022-01-01.
    assert _run([SPLIT_DEVELOPMENT], ["2022-01-01 06:50:00+00:00"]) == [SPLIT_EMBARGOED]


def test_development_row_just_before_boundary_is_kept() -> None:
    assert _run([SPLIT_DEVELOPMENT], ["2021-12-31 23:59:59+00:00"]) == [SPLIT_DEVELOPMENT]


def test_development_row_exactly_on_boundary_is_embargoed() -> None:
    # The boundary instant belongs to validation, so a development row there is on the wrong side.
    assert _run([SPLIT_DEVELOPMENT], ["2022-01-01 00:00:00+00:00"]) == [SPLIT_EMBARGOED]


def test_validation_row_exactly_on_boundary_is_kept() -> None:
    assert _run([SPLIT_VALIDATION], ["2022-01-01 00:00:00+00:00"]) == [SPLIT_VALIDATION]


def test_validation_row_early_in_january_is_kept() -> None:
    # Real pattern from the audit: earliest validation prediction time was 2022-01-01 03:59 UTC.
    assert _run([SPLIT_VALIDATION], ["2022-01-01 03:59:00+00:00"]) == [SPLIT_VALIDATION]


def test_validation_row_before_validation_start_is_embargoed() -> None:
    # Cannot happen for US airports (UTC is ahead of local time), but the rule must hold
    # for any airport, so a validation-labelled row with a December 2021 UTC time is removed.
    assert _run([SPLIT_VALIDATION], ["2021-12-31 23:59:59+00:00"]) == [SPLIT_EMBARGOED]


def test_validation_row_after_final_start_is_embargoed() -> None:
    assert _run([SPLIT_VALIDATION], ["2023-01-01 07:35:00+00:00"]) == [SPLIT_EMBARGOED]


def test_validation_row_exactly_on_final_start_is_embargoed() -> None:
    assert _run([SPLIT_VALIDATION], ["2023-01-01 00:00:00+00:00"]) == [SPLIT_EMBARGOED]


def test_final_row_kept_and_never_widened() -> None:
    assert _run([SPLIT_FINAL_TEST], ["2023-01-01 05:59:00+00:00"]) == [SPLIT_FINAL_TEST]


def test_final_row_before_final_start_is_embargoed() -> None:
    assert _run([SPLIT_FINAL_TEST], ["2022-12-31 23:00:00+00:00"]) == [SPLIT_EMBARGOED]


def test_missing_timestamp_is_embargoed() -> None:
    times = pd.Series(pd.to_datetime(["2020-01-01 00:00:00+00:00", None], utc=True))
    result = apply_utc_embargo(pd.Series([SPLIT_DEVELOPMENT, SPLIT_DEVELOPMENT]), times, BOUNDARIES)
    assert [str(v) for v in result] == [SPLIT_DEVELOPMENT, SPLIT_EMBARGOED]


def test_index_is_preserved_and_input_not_modified() -> None:
    labels = pd.Series([SPLIT_DEVELOPMENT, SPLIT_VALIDATION], index=[10, 20])
    times = _stamps(["2021-01-01 00:00:00+00:00", "2022-06-01 00:00:00+00:00"])
    times.index = [10, 20]
    labels_before = labels.copy()
    result = apply_utc_embargo(labels, times, BOUNDARIES)
    assert list(result.index) == [10, 20]
    pd.testing.assert_series_equal(labels, labels_before)


def test_unknown_split_label_raises() -> None:
    with pytest.raises(ValueError, match="Unexpected split labels"):
        apply_utc_embargo(pd.Series(["train"]), _stamps(["2020-01-01 00:00:00+00:00"]), BOUNDARIES)


def test_naive_timestamps_raise() -> None:
    naive = pd.Series(pd.to_datetime(["2020-01-01 00:00:00"]))
    with pytest.raises(ValueError, match="timezone-aware"):
        apply_utc_embargo(pd.Series([SPLIT_DEVELOPMENT]), naive, BOUNDARIES)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        apply_utc_embargo(
            pd.Series([SPLIT_DEVELOPMENT, SPLIT_VALIDATION]),
            _stamps(["2020-01-01 00:00:00+00:00"]),
            BOUNDARIES,
        )


def test_non_utc_timezone_is_converted_before_comparing() -> None:
    # 2021-12-31 19:30 in New York (UTC-5) is 2022-01-01 00:30 UTC, so it is past the boundary.
    times = pd.Series(pd.to_datetime(["2021-12-31 19:30:00-05:00"])).dt.tz_convert("America/New_York")
    result = apply_utc_embargo(pd.Series([SPLIT_DEVELOPMENT]), times, BOUNDARIES)
    assert [str(v) for v in result] == [SPLIT_EMBARGOED]


# ---------------------------------------------------------------------------
# Chronology: the property the embargo exists to guarantee
# ---------------------------------------------------------------------------
def test_chronology_fails_before_embargo_and_passes_after() -> None:
    rng = np.random.default_rng(0)
    n = 60_000  # about 35 rows per day, so boundary rows exist on both sides
    # Local flight dates fix the split; UTC time is the local date plus 0-30 hours,
    # which reproduces the boundary straddling seen in the real manifest.
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 1700, n)), unit="D")
    offsets = pd.to_timedelta(rng.integers(0, 30 * 60, n), unit="min")
    stamps = pd.Series((dates + offsets).tz_localize("UTC"))
    labels = pd.Series(
        np.where(
            dates < pd.Timestamp("2022-01-01"),
            SPLIT_DEVELOPMENT,
            np.where(dates < pd.Timestamp("2023-01-01"), SPLIT_VALIDATION, SPLIT_FINAL_TEST),
        )
    )

    assert not chronology_ok(labels, stamps)  # overlap exists before the embargo

    embargoed = apply_utc_embargo(labels, stamps, BOUNDARIES)
    assert (embargoed == SPLIT_EMBARGOED).sum() > 0
    assert chronology_ok(embargoed, stamps)  # strictly ordered afterwards


def test_embargo_never_moves_a_row_between_retained_splits() -> None:
    rng = np.random.default_rng(1)
    stamps = pd.Series(
        pd.to_datetime("2020-06-01", utc=True) + pd.to_timedelta(rng.integers(0, 1200 * 24 * 60, 2000), unit="min")
    )
    labels = pd.Series(rng.choice([SPLIT_DEVELOPMENT, SPLIT_VALIDATION, SPLIT_FINAL_TEST], 2000))
    result = apply_utc_embargo(labels, stamps, BOUNDARIES).astype(str)
    kept = result != SPLIT_EMBARGOED
    # A retained row must keep the label it started with.
    assert (result[kept].to_numpy() == labels[kept].to_numpy()).all()


# ---------------------------------------------------------------------------
# Boundaries object and config reader
# ---------------------------------------------------------------------------
def test_boundaries_reject_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        UtcSplitBoundaries(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01", tz="UTC"))


def test_boundaries_reject_wrong_order() -> None:
    with pytest.raises(ValueError, match="earlier"):
        UtcSplitBoundaries(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"))


def test_boundaries_reject_non_utc_timezone() -> None:
    with pytest.raises(ValueError, match="UTC"):
        UtcSplitBoundaries(
            pd.Timestamp("2022-01-01", tz="America/New_York"), pd.Timestamp("2023-01-01", tz="UTC")
        )


def test_boundaries_from_config_quoted_dates(tmp_path) -> None:
    config = tmp_path / "validation.yaml"
    config.write_text(
        'primary_temporal_split:\n  validation_start: "2022-01-01"\n  validation_end: "2022-12-31"\n',
        encoding="utf-8",
    )
    boundaries = boundaries_from_config(config)
    assert boundaries.validation_start_utc == pd.Timestamp("2022-01-01T00:00:00Z")
    # validation_end is inclusive, so the final period starts the next day.
    assert boundaries.final_test_start_utc == pd.Timestamp("2023-01-01T00:00:00Z")


def test_boundaries_from_config_unquoted_dates(tmp_path) -> None:
    config = tmp_path / "validation.yaml"
    config.write_text(
        "primary_temporal_split:\n  validation_start: 2022-01-01\n  validation_end: 2022-12-31\n",
        encoding="utf-8",
    )
    boundaries = boundaries_from_config(config)
    assert boundaries.final_test_start_utc == pd.Timestamp("2023-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# Regimes and boolean helper
# ---------------------------------------------------------------------------
def test_regime_mapping() -> None:
    dates = pd.Series(["2019-06-01", "2020-03-15", "2021-12-31", "2022-01-01", "2023-08-31", "2018-12-31"])
    regimes = [str(v) for v in assign_regime(dates)]
    assert regimes == [REGIME_2019, REGIME_2020, REGIME_2021_2022, REGIME_2021_2022, REGIME_2023, REGIME_OTHER]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1, 0, 1]),
        pd.Series([1.0, 0.0, 1.0]),
        pd.Series([True, False, True]),
        pd.Series([1.0, np.nan, 1.0]),
    ],
)
def test_to_boolean_handles_common_encodings(values: pd.Series) -> None:
    result = to_boolean(values)
    assert result.dtype == bool
    assert result.iloc[0] and result.iloc[2]
    assert not result.iloc[1]


def test_to_boolean_object_column_with_missing_value() -> None:
    # read_csv gives an object column when True/False text is mixed with blanks.
    values = pd.Series([True, False, np.nan, True], dtype="object")
    assert to_boolean(values).tolist() == [True, False, False, True]
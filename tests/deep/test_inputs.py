"""Tests for the neural-network inputs: fit-only learning, categories, cycles, scaling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from airline_disruption.deep import inputs as di
from airline_disruption.features.feature_sets import BASE_NO_YEAR_COLUMNS, find_forbidden_columns


def frame(n: int = 400, seed: int = 0, airports=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = rng.choice(airports, n)
    dest = rng.choice(airports, n)
    return pd.DataFrame(
        {
            "origin_airport": pd.Categorical(origin),
            "destination_airport": pd.Categorical(dest),
            "route": pd.Categorical([f"{o}-{d}" for o, d in zip(origin, dest)]),
            "carrier_identifier": pd.Categorical(rng.choice(["X", "Y"], n)),
            "scheduled_departure_month": rng.integers(1, 13, n).astype("float32"),
            "scheduled_departure_day_of_week": rng.integers(0, 7, n).astype("float32"),
            "scheduled_departure_hour_local": rng.integers(0, 24, n).astype("float32"),
            "scheduled_departure_minute_local": rng.integers(0, 60, n).astype("float32"),
            "scheduled_arrival_hour_local": rng.integers(0, 24, n).astype("float32"),
            "scheduled_arrival_minute_local": rng.integers(0, 60, n).astype("float32"),
            "scheduled_elapsed_time_minutes": rng.integers(40, 400, n).astype("float32"),
            "distance_miles": rng.integers(80, 2800, n).astype("float32"),
        }
    )


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------
def test_static_inputs_are_built_from_exactly_the_base_no_year_columns() -> None:
    assert set(di.CATEGORICAL_COLUMNS) | set(di.NUMERIC_SOURCE_COLUMNS) == set(BASE_NO_YEAR_COLUMNS)
    assert "scheduled_departure_year" not in di.NUMERIC_SOURCE_COLUMNS
    assert find_forbidden_columns(di.NUMERIC_SOURCE_COLUMNS + di.CATEGORICAL_COLUMNS) == []


def test_output_shapes_dtypes_and_names_agree() -> None:
    X = frame()
    spec = di.fit_static_spec(X)
    codes, numeric = di.transform_static(X, spec)
    assert codes.shape == (len(X), 4) and codes.dtype == np.int32
    assert numeric.shape == (len(X), len(di.NUMERIC_NAMES)) and numeric.dtype == np.float32
    assert np.isfinite(numeric).all()


# ---------------------------------------------------------------------------
# Learning happens on the fit rows only
# ---------------------------------------------------------------------------
def test_vocabulary_medians_and_scaling_come_from_the_fit_rows_only() -> None:
    fit = frame(600, seed=1)
    other = frame(600, seed=2)
    other["distance_miles"] = 50_000.0  # wildly different rows that must not leak into the spec
    other["scheduled_elapsed_time_minutes"] = 9_000.0
    other["origin_airport"] = pd.Categorical(["ZZZ"] * len(other))

    spec = di.fit_static_spec(fit)
    reference = di.fit_static_spec(fit.copy())
    assert spec.to_json() == reference.to_json()
    assert "ZZZ" not in spec.vocabularies["origin_airport"]
    assert spec.medians["distance_miles"] == pytest.approx(float(fit["distance_miles"].median()))

    # Transforming the other rows does not change the spec, and their extreme values show up as extreme.
    before = spec.to_json()
    codes, numeric = di.transform_static(other, spec)
    assert spec.to_json() == before
    assert (codes[:, 0] == di.UNKNOWN_CODE).all()
    assert numeric[:, di.NUMERIC_NAMES.index("log_distance_z")].min() > 3


def test_fit_stats_are_exactly_the_mean_and_std_of_the_logged_fit_values() -> None:
    X = frame(500, seed=3)
    spec = di.fit_static_spec(X)
    logged = np.log1p(X["distance_miles"].to_numpy(dtype="float64"))
    assert spec.log_mean["distance_miles"] == pytest.approx(logged.mean())
    assert spec.log_std["distance_miles"] == pytest.approx(logged.std())
    _, numeric = di.transform_static(X, spec)
    z = numeric[:, di.NUMERIC_NAMES.index("log_distance_z")]
    assert z.mean() == pytest.approx(0.0, abs=1e-4) and z.std() == pytest.approx(1.0, abs=1e-3)


def test_constant_column_does_not_divide_by_zero() -> None:
    X = frame(100)
    X["distance_miles"] = 500.0
    spec = di.fit_static_spec(X)
    _, numeric = di.transform_static(X, spec)
    assert np.isfinite(numeric).all() and spec.log_std["distance_miles"] > 0


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
def test_rare_and_unseen_levels_share_the_unknown_code() -> None:
    fit = frame(300)
    fit["origin_airport"] = pd.Categorical(["AAA"] * 250 + ["BBB"] * 45 + ["RARE"] * 5)  # RARE below the minimum
    spec = di.fit_static_spec(fit, min_level_count=20)
    assert spec.vocabularies["origin_airport"] == ["AAA", "BBB"]  # most frequent first

    later = frame(6, seed=9)
    later["origin_airport"] = pd.Categorical(["AAA", "BBB", "RARE", "NEVER_SEEN", "AAA", "BBB"])
    codes, _ = di.transform_static(later, spec)
    assert codes[:, 0].tolist() == [1, 2, 0, 0, 1, 2]


def test_vocabulary_order_is_by_frequency_then_name_and_stable() -> None:
    fit = frame(300)
    fit["carrier_identifier"] = pd.Categorical(["B"] * 100 + ["A"] * 100 + ["C"] * 100)  # a three-way tie
    spec = di.fit_static_spec(fit)
    assert spec.vocabularies["carrier_identifier"] == ["A", "B", "C"]


def test_categorical_and_plain_string_columns_encode_identically() -> None:
    levels = ["AAA", "BBB"]
    strings = pd.Series(["BBB", "AAA", "ZZZ", "AAA"])
    categorical = pd.Series(pd.Categorical(strings, categories=["ZZZ", "AAA", "BBB", "UNUSED"]))
    assert di.encode_categories(strings, levels).tolist() == di.encode_categories(categorical, levels).tolist() == [2, 1, 0, 1]


def test_missing_category_maps_to_unknown() -> None:
    column = pd.Series(pd.Categorical(["AAA", None, "BBB"], categories=["AAA", "BBB"]))
    assert di.encode_categories(column, ["AAA", "BBB"]).tolist() == [1, 0, 2]


def test_vocabulary_sizes_count_the_unknown_slot() -> None:
    spec = di.fit_static_spec(frame(400))
    sizes = spec.vocabulary_sizes()
    assert all(sizes[c] == len(spec.vocabularies[c]) + 1 for c in di.CATEGORICAL_COLUMNS)


@pytest.mark.parametrize("n_levels, expected", [(2, 4), (20, 9), (380, 32), (9000, 32)])
def test_embedding_width_rule(n_levels: int, expected: int) -> None:
    assert di.embedding_dim(n_levels) == expected


# ---------------------------------------------------------------------------
# Cyclical numbers and missing values
# ---------------------------------------------------------------------------
def one_row(**changes) -> pd.DataFrame:
    row = frame(1, seed=4)
    for key, value in changes.items():
        row[key] = np.float32(value)
    return row


def numeric_of(row: pd.DataFrame, spec: di.StaticSpec, name: str) -> float:
    return float(di.transform_static(row, spec)[1][0, di.NUMERIC_NAMES.index(name)])


def test_clock_times_wrap_around_midnight() -> None:
    spec = di.fit_static_spec(frame(300))
    late = one_row(scheduled_departure_hour_local=23, scheduled_departure_minute_local=59)
    midnight = one_row(scheduled_departure_hour_local=0, scheduled_departure_minute_local=0)
    noon = one_row(scheduled_departure_hour_local=12, scheduled_departure_minute_local=0)
    def distance(a: pd.DataFrame, b: pd.DataFrame) -> float:  # straight-line distance on the clock circle
        return math.hypot(
            numeric_of(a, spec, "departure_time_sin") - numeric_of(b, spec, "departure_time_sin"),
            numeric_of(a, spec, "departure_time_cos") - numeric_of(b, spec, "departure_time_cos"),
        )

    assert distance(late, midnight) < 0.01  # one minute apart across midnight
    assert distance(late, noon) > 1.9  # opposite sides of the clock
    assert numeric_of(midnight, spec, "departure_time_cos") == pytest.approx(1.0)
    assert numeric_of(noon, spec, "departure_time_cos") == pytest.approx(-1.0)


def test_minutes_are_part_of_the_clock_time() -> None:
    spec = di.fit_static_spec(frame(300))
    six = one_row(scheduled_arrival_hour_local=6, scheduled_arrival_minute_local=0)
    quarter_past = one_row(scheduled_arrival_hour_local=6, scheduled_arrival_minute_local=15)
    assert numeric_of(six, spec, "arrival_time_sin") == pytest.approx(1.0)  # 06:00 is a quarter of the day
    expected = math.sin(2 * math.pi * 6.25 / 24)
    assert numeric_of(quarter_past, spec, "arrival_time_sin") == pytest.approx(expected, abs=1e-6)


def test_month_and_weekday_cycles_use_the_right_periods() -> None:
    spec = di.fit_static_spec(frame(300))
    january, december = one_row(scheduled_departure_month=1), one_row(scheduled_departure_month=12)
    july = one_row(scheduled_departure_month=7)
    assert numeric_of(january, spec, "month_cos") == pytest.approx(1.0)  # month 1 is the start of the cycle
    assert numeric_of(july, spec, "month_cos") == pytest.approx(-1.0)  # 6 months later is half a cycle
    assert abs(numeric_of(december, spec, "month_sin")) < abs(numeric_of(july, spec, "month_sin")) + 0.6
    monday, next_monday = one_row(scheduled_departure_day_of_week=0), one_row(scheduled_departure_day_of_week=7)
    assert numeric_of(monday, spec, "day_of_week_cos") == pytest.approx(numeric_of(next_monday, spec, "day_of_week_cos"), abs=1e-6)


def test_missing_numbers_get_the_fit_median() -> None:
    fit = frame(300, seed=5)
    spec = di.fit_static_spec(fit)
    X = frame(3, seed=6)
    X.loc[X.index[1], "scheduled_elapsed_time_minutes"] = np.nan
    X.loc[X.index[2], "distance_miles"] = np.nan
    _, numeric = di.transform_static(X, spec)
    assert np.isfinite(numeric).all()
    filled = X.copy()
    filled.loc[filled.index[1], "scheduled_elapsed_time_minutes"] = spec.medians["scheduled_elapsed_time_minutes"]
    filled.loc[filled.index[2], "distance_miles"] = spec.medians["distance_miles"]
    assert np.allclose(numeric, di.transform_static(filled, spec)[1])


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        di.fit_static_spec(frame().drop(columns=["distance_miles"]))


def test_spec_round_trips_through_json() -> None:
    spec = di.fit_static_spec(frame(400))
    again = di.StaticSpec.from_json(spec.to_json())
    assert again.to_json() == spec.to_json()


# ---------------------------------------------------------------------------
# Sequence scaler
# ---------------------------------------------------------------------------
def test_scaler_uses_real_positions_only() -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(0, 40, size=(50, 6, 3)).astype("float32")
    mask = np.ones((50, 6), bool)
    mask[:, :2] = False
    values[:, :2, :] = 10_000.0  # padded region holds garbage that must not enter the statistics
    scaler = di.fit_sequence_scaler(values, mask)
    real = np.log1p(values[mask].astype("float64"))
    assert np.allclose(scaler.mean, real.mean(axis=0)) and np.allclose(scaler.std, real.std(axis=0))


def test_scaler_zeroes_padding_and_standardises_real_positions() -> None:
    rng = np.random.default_rng(1)
    values = rng.integers(0, 40, size=(200, 5, 3)).astype("float32")
    mask = np.ones((200, 5), bool)
    mask[:100, :3] = False
    scaler = di.fit_sequence_scaler(values, mask)
    scaled = scaler.apply(values, mask)
    assert scaled.dtype == np.float32
    assert (scaled[:100, :3] == 0).all()
    real = scaled[mask]
    assert np.allclose(real.mean(axis=0), 0, atol=1e-4) and np.allclose(real.std(axis=0), 1, atol=1e-3)


def test_scaler_needs_some_real_positions_and_survives_constant_channels() -> None:
    with pytest.raises(ValueError, match="No real positions"):
        di.fit_sequence_scaler(np.zeros((3, 4, 3), "float32"), np.zeros((3, 4), bool))
    constant = di.fit_sequence_scaler(np.full((3, 4, 3), 5.0, "float32"), np.ones((3, 4), bool))
    assert min(constant.std) > 0
    assert np.isfinite(constant.apply(np.full((3, 4, 3), 5.0, "float32"), np.ones((3, 4), bool))).all()


def test_scaler_round_trips_through_json() -> None:
    scaler = di.SequenceScaler([1.0, 2.0, 3.0], [0.5, 0.6, 0.7])
    assert di.SequenceScaler.from_json(scaler.to_json()) == scaler

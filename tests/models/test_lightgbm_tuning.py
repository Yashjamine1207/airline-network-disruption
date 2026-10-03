"""Tests for the pre-registered LightGBM tuning protocol."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from airline_disruption.models import lightgbm_tuning as tune
from airline_disruption.models.lightgbm_benchmark import LIGHTGBM_PARAMS


# ---------------------------------------------------------------------------
# Search space and sampler
# ---------------------------------------------------------------------------
def test_the_default_is_reachable_in_every_dimension() -> None:
    lightgbm_defaults = {"cat_smooth": 10, "min_data_per_group": 100, "cat_l2": 10, "max_cat_threshold": 32,
                         "colsample_bytree": 1.0, "subsample": 1.0}
    reference = {**lightgbm_defaults, **{k: v for k, v in LIGHTGBM_PARAMS.items() if k in tune.SEARCH_SPACE}}
    for key, values in tune.SEARCH_SPACE.items():
        assert reference[key] in values, key


def test_search_space_never_touches_reserved_or_fixed_settings() -> None:
    fixed = {"objective", "metric", "n_estimators", "verbosity", "random_state", "n_jobs"}
    assert not fixed & set(tune.SEARCH_SPACE)
    assert all(len(values) >= 2 for values in tune.SEARCH_SPACE.values())


def test_first_config_is_the_default_and_the_rest_are_distinct_draws_from_the_space() -> None:
    configs = tune.sample_configs(25, seed=3)
    assert len(configs) == 26 and configs[0] == {}
    draws = [tuple(sorted(c.items())) for c in configs[1:]]
    assert len(set(draws)) == 25
    for config in configs[1:]:
        assert set(config) == set(tune.SEARCH_SPACE)
        for key, value in config.items():
            assert value in tune.SEARCH_SPACE[key]


def test_draws_are_distinct_even_when_the_space_is_tiny(monkeypatch) -> None:
    monkeypatch.setattr(tune, "SEARCH_SPACE", {"learning_rate": [0.02, 0.05], "num_leaves": [15, 31]})  # 4 combinations
    configs = tune.sample_configs(4, seed=0)
    combinations = {(c["learning_rate"], c["num_leaves"]) for c in configs[1:]}
    assert len(combinations) == 4  # every combination exactly once
    with pytest.raises(ValueError, match="larger than the search space"):
        tune.sample_configs(5)


def test_random_state_and_n_jobs_can_never_be_set_from_outside() -> None:
    assert not {"random_state", "n_jobs"} & tune.ALLOWED_KEYS


def test_sampling_is_reproducible_and_seed_dependent() -> None:
    assert tune.sample_configs(10, seed=5) == tune.sample_configs(10, seed=5)
    assert tune.sample_configs(10, seed=5) != tune.sample_configs(10, seed=6)


def test_zero_draws_gives_only_the_default_and_bad_sizes_are_rejected() -> None:
    assert tune.sample_configs(0) == [{}]
    with pytest.raises(ValueError, match="negative"):
        tune.sample_configs(-1)
    with pytest.raises(ValueError, match="larger than the search space"):
        tune.sample_configs(10**9)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_params_for_the_default_equals_the_phase4_settings_and_leaves_them_untouched() -> None:
    before = dict(LIGHTGBM_PARAMS)
    assert tune.params_for({}) == before
    tune.params_for({"num_leaves": 63})
    assert LIGHTGBM_PARAMS == before


def test_overrides_are_applied_on_top_of_the_defaults() -> None:
    params = tune.params_for({"num_leaves": 63, "cat_smooth": 200})
    assert params["num_leaves"] == 63 and params["cat_smooth"] == 200
    assert params["learning_rate"] == LIGHTGBM_PARAMS["learning_rate"]


@pytest.mark.parametrize("subsample, expect_frequency", [(0.7, True), (1.0, False)])
def test_bagging_frequency_is_set_only_when_subsampling(subsample: float, expect_frequency: bool) -> None:
    params = tune.params_for({"subsample": subsample})
    assert ("subsample_freq" in params) is expect_frequency
    if expect_frequency:
        assert params["subsample_freq"] == 1


@pytest.mark.parametrize("bad_key", ["num_leavs", "random_state", "n_jobs", "objective_typo"])
def test_unknown_or_reserved_settings_are_rejected(bad_key: str) -> None:
    with pytest.raises(ValueError, match="Unknown or reserved"):
        tune.params_for({bad_key: 1})


# ---------------------------------------------------------------------------
# Validation halves
# ---------------------------------------------------------------------------
def stamps(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values), utc=True, format="ISO8601"))


def test_halves_split_exactly_at_the_first_of_july_utc() -> None:
    selection, confirmation = tune.validation_halves(
        stamps("2022-01-01 00:00", "2022-06-30 23:59:59", "2022-07-01 00:00", "2022-12-31 23:59")
    )
    assert selection.tolist() == [True, True, False, False]
    assert confirmation.tolist() == [False, False, True, True]


def test_halves_are_disjoint_and_cover_every_row() -> None:
    rng = np.random.default_rng(0)
    seconds = rng.integers(0, 365 * 86400, 5000)
    series = pd.Series(pd.to_datetime(pd.Timestamp("2022-01-01", tz="UTC").timestamp() + seconds, unit="s", utc=True))
    selection, confirmation = tune.validation_halves(series)
    assert not (selection & confirmation).any() and (selection | confirmation).all()


@pytest.mark.parametrize("outside", ["2021-12-31 23:59:59", "2023-01-01 00:00"])
def test_rows_outside_2022_are_rejected(outside: str) -> None:
    with pytest.raises(ValueError, match="outside 2022"):
        tune.validation_halves(stamps("2022-05-01", outside))


# ---------------------------------------------------------------------------
# Selection and adoption rules
# ---------------------------------------------------------------------------
def test_best_config_is_the_highest_selection_score_and_never_the_default() -> None:
    assert tune.best_config_index([0.90, 0.41, 0.44, 0.43]) == 2  # default is highest but is not a candidate
    assert tune.best_config_index([0.10, 0.41, 0.44, 0.43], exclude_default=False) == 2
    assert tune.best_config_index([0.90, 0.10], exclude_default=False) == 0


def test_ties_go_to_the_lower_index_and_nan_is_ignored() -> None:
    assert tune.best_config_index([0.0, 0.5, 0.5, 0.4]) == 1
    assert tune.best_config_index([0.0, np.nan, 0.3, 0.2]) == 2
    with pytest.raises(ValueError, match="No candidate"):
        tune.best_config_index([0.5, np.nan, np.nan])
    with pytest.raises(ValueError, match="No candidate"):
        tune.best_config_index([0.5])  # only the default exists


@pytest.mark.parametrize(
    "ci_low, expected",
    [(0.0004, True), (1e-9, True), (0.0, False), (-0.0001, False), (np.nan, False), (np.inf, False)],
)
def test_challenger_is_adopted_only_when_the_interval_is_wholly_above_zero(ci_low: float, expected: bool) -> None:
    assert tune.decide_adoption(ci_low) is expected


# ---------------------------------------------------------------------------
# Settings file
# ---------------------------------------------------------------------------
def test_params_file_round_trip(tmp_path) -> None:
    path = tmp_path / "selected.json"
    params = tune.params_for({"num_leaves": 63, "subsample": 0.7})
    path.write_text(json.dumps({"params": params, "other": "ignored"}), encoding="utf-8")
    assert tune.load_params_file(path) == params


@pytest.mark.parametrize(
    "content, message",
    [
        ({"nothing": 1}, "no 'params' block"),
        ({"params": [1, 2]}, "no 'params' block"),
        ({"params": {"num_leaves": 31, "random_state": 7}}, "not allowed"),
        ({"params": {"num_leavs": 31}}, "not allowed"),
    ],
)
def test_params_file_rejects_bad_content(tmp_path, content, message) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        tune.load_params_file(path)

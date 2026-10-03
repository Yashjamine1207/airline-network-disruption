"""Tests for calibration split safety, the decision rules and the config (Phase 7A, Step 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from airline_disruption.calibration import decision, guards
from airline_disruption.calibration.config import load_calibration_config
from airline_disruption.calibration.experiment import fit_and_apply, metrics_table

DAY = 86_400
JAN_2022 = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp())
JAN_2023 = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_the_final_test_start_constant_is_2023_01_01_utc() -> None:
    assert guards.FINAL_TEST_START_SECONDS == JAN_2023


def test_a_final_test_row_stops_the_run() -> None:
    guards.reject_final_test_rows([JAN_2023 - 1], "x")  # the last second of 2022 is fine
    with pytest.raises(ValueError, match="final-test"):
        guards.reject_final_test_rows([JAN_2022, JAN_2023], "validation")


def test_calibration_must_end_strictly_before_evaluation_starts() -> None:
    guards.require_calibration_before_evaluation([1, 5], [6, 9], "ok")
    for cal, ev in (([1, 6], [6, 9]), ([1, 7], [6, 9])):  # touching and overlapping both fail
        with pytest.raises(ValueError, match="Calibration must come first"):
            guards.require_calibration_before_evaluation(cal, ev, "bad")
    with pytest.raises(ValueError, match="no calibration rows"):
        guards.require_calibration_before_evaluation([], [1], "empty")


def test_shared_rows_are_rejected() -> None:
    guards.require_disjoint([1, 2, 3], [4, 5], "ok")
    with pytest.raises(ValueError, match="1 rows are used both"):
        guards.require_disjoint([1, 2, 3], [3, 4], "bad")


# ---------------------------------------------------------------------------
# The experiment function
# ---------------------------------------------------------------------------
def two_periods(n_fit: int = 30_000, n_eval: int = 30_000, seed: int = 0):
    """Calibration rows in 2021 and evaluation rows in 2022. The model under-predicts by a fixed amount."""
    rng = np.random.default_rng(seed)
    n = n_fit + n_eval
    score = np.clip(expit(rng.normal(-3.6, 0.9, n)), 1e-4, 0.9)
    y = (rng.random(n) < expit(0.4 + logit(score))).astype("int8")
    seconds = np.concatenate([JAN_2022 - 100 * DAY + np.arange(n_fit) * 60, JAN_2022 + np.arange(n_eval) * 60])
    return score, y, seconds, n_fit


def test_the_experiment_function_has_no_way_to_receive_evaluation_labels() -> None:
    """Labels of the rows being scored are not an argument, so they cannot reach a calibrator by construction."""
    import inspect

    names = list(inspect.signature(fit_and_apply).parameters)
    assert "y_fit" in names
    assert [n for n in names if n.startswith(("y_", "labels")) and n != "y_fit"] == []


def test_outputs_depend_only_on_the_fit_rows_and_the_scores_being_applied() -> None:
    score, y, seconds, n_fit = two_periods()
    args = (("uncalibrated", "platt", "isotonic"), score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t")
    _, first = fit_and_apply(*args)
    _, second = fit_and_apply(*args)
    for method in first:
        assert np.array_equal(first[method], second[method])  # deterministic


def test_calibration_labels_do_change_the_calibrators() -> None:
    """The opposite check: the fit rows' labels are what the calibrator learns from."""
    score, y, seconds, n_fit = two_periods()
    methods = ("platt", "isotonic")
    _, base = fit_and_apply(methods, score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t")
    flipped = y[:n_fit].copy()
    np.random.default_rng(2).shuffle(flipped)
    _, other = fit_and_apply(methods, score[:n_fit], flipped, seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t")
    for method in methods:
        assert not np.allclose(base[method], other[method])


def test_the_experiment_refuses_final_test_rows_and_wrong_order_and_overlap() -> None:
    score, y, seconds, n_fit = two_periods(2_000, 2_000)
    methods = ("platt",)
    late = seconds.copy()
    late[-1] = JAN_2023 + 5
    with pytest.raises(ValueError, match="final-test"):
        fit_and_apply(methods, score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], late[n_fit:], "t")
    with pytest.raises(ValueError, match="Calibration must come first"):  # roles swapped
        fit_and_apply(methods, score[n_fit:], y[n_fit:], seconds[n_fit:], score[:n_fit], seconds[:n_fit], "t")
    ids = np.arange(len(score))
    with pytest.raises(ValueError, match="both to fit"):
        fit_and_apply(methods, score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t",
                      ids_fit=ids[:n_fit], ids_apply=ids[n_fit - 1 : -1])


def test_platt_repairs_the_level_shift_on_later_rows_and_does_not_change_pr_auc() -> None:
    score, y, seconds, n_fit = two_periods(60_000, 60_000)
    methods = ("uncalibrated", "platt", "isotonic")
    _, probs = fit_and_apply(methods, score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t")
    everything = {"all": np.ones(len(score) - n_fit, dtype=bool)}
    table = metrics_table("A", "old rows", y[n_fit:], probs, everything, float(y[:n_fit].mean()), 10).set_index("method")
    assert table.loc["platt", "brier"] < table.loc["uncalibrated", "brier"]
    assert abs(table.loc["platt", "calibration_in_the_large"]) < abs(table.loc["uncalibrated", "calibration_in_the_large"])
    assert table.loc["uncalibrated", "predicted_to_observed"] < 0.85
    assert 0.95 < table.loc["platt", "predicted_to_observed"] < 1.05
    assert table.loc["platt", "pr_auc"] == pytest.approx(table.loc["uncalibrated", "pr_auc"], abs=1e-12)  # same ranking
    assert table.loc["isotonic", "distinct_scores"] < table.loc["uncalibrated", "distinct_scores"]


def test_metrics_table_skips_empty_periods_and_labels_every_row() -> None:
    score, y, seconds, n_fit = two_periods(5_000, 5_000)
    _, probs = fit_and_apply(("platt",), score[:n_fit], y[:n_fit], seconds[:n_fit], score[n_fit:], seconds[n_fit:], "t")
    n = len(score) - n_fit
    periods = {"first half": np.arange(n) < n // 2, "second half": np.arange(n) >= n // 2, "empty": np.zeros(n, dtype=bool)}
    table = metrics_table("A", "old rows", y[n_fit:], probs, periods, 0.03, 10)
    assert table["period"].tolist() == ["first half", "second half"]
    assert (table["experiment"] == "A").all() and (table["calibrator_fitted_on"] == "old rows").all()


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
BRIER = {"uncalibrated": 0.0250, "platt": 0.0248, "isotonic": 0.0247}


def test_rule_1_keeps_the_scores_when_no_calibrator_helps_on_the_selection_half() -> None:
    worse = {**BRIER, "platt": 0.0251, "isotonic": 0.0252}
    method, reason = decision.choose_method(worse, {"platt": -1.0, "isotonic": -1.0}, -1.0)
    assert method == "uncalibrated" and "selection half" in reason


def test_rule_1_needs_the_confirmation_interval_to_be_wholly_below_zero() -> None:
    assert decision.choose_method(BRIER, {"platt": 1e-5, "isotonic": 2e-5}, -1e-5)[0] == "uncalibrated"
    assert decision.choose_method(BRIER, {"platt": 0.0, "isotonic": 0.0}, -1e-5)[0] == "uncalibrated"  # zero is not below zero
    assert decision.choose_method(BRIER, {"platt": np.nan, "isotonic": np.nan}, -1e-5)[0] == "uncalibrated"


def test_rule_1_picks_the_only_confirmed_method() -> None:
    assert decision.choose_method(BRIER, {"platt": -1e-5, "isotonic": 1e-5}, -1.0)[0] == "platt"
    assert decision.choose_method(BRIER, {"platt": 1e-5, "isotonic": -1e-5}, -1.0)[0] == "isotonic"


def test_rule_1_prefers_platt_unless_isotonic_clearly_beats_it() -> None:
    both = {"platt": -1e-5, "isotonic": -2e-5}
    assert decision.choose_method(BRIER, both, 1e-6)[0] == "platt"
    assert decision.choose_method(BRIER, both, np.nan)[0] == "platt"
    assert decision.choose_method(BRIER, both, -1e-6)[0] == "isotonic"


def test_rule_1_only_confirms_methods_that_improved_the_selection_half() -> None:
    only_platt_helped = {**BRIER, "isotonic": 0.0251}
    method, _ = decision.choose_method(only_platt_helped, {"platt": -1e-5, "isotonic": -1e-5}, -1e-5)
    assert method == "platt"  # isotonic passed the confirmation test but was not a candidate


def test_rule_1_rejects_incomplete_input() -> None:
    with pytest.raises(ValueError, match="missing"):
        decision.choose_method({"uncalibrated": 0.02, "platt": 0.02}, {}, 0.0)


def test_rule_2_final_fit_source() -> None:
    assert decision.choose_final_fit_source("uncalibrated", -1.0)[0] == "none"
    assert decision.choose_final_fit_source("platt", -1e-5)[0] == "recent_year"
    assert decision.choose_final_fit_source("platt", 1e-5)[0] == "same_as_here"
    assert decision.choose_final_fit_source("isotonic", np.nan)[0] == "same_as_here"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOOD = """
model: {feature_set: base_no_year, params_file: configs/x.json, seed: 42, reference_predictions: p.parquet, reference_column: c}
methods: [uncalibrated, platt, isotonic]
calibration: {primary_fit_rows: early_stop, diagnostic_fit_rows: validation_first_half}
evaluation: {selection: validation_first_half, confirmation: validation_second_half}
metrics: {reliability_bins: 10}
bootstrap: {resamples: 100, seed: 42, confidence: 0.95}
final_test: {allowed: false}
"""


def write(tmp_path, text: str):
    path = tmp_path / "calibration.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_valid_config_loads(tmp_path) -> None:
    config = load_calibration_config(write(tmp_path, GOOD))
    assert config["methods"] == ["uncalibrated", "platt", "isotonic"] and config["bootstrap"]["resamples"] == 100


def test_a_config_that_allows_the_final_test_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="final test is locked"):
        load_calibration_config(write(tmp_path, GOOD.replace("allowed: false", "allowed: true")))


def test_bad_configs_are_rejected_with_a_clear_message(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown methods"):
        load_calibration_config(write(tmp_path, GOOD.replace("isotonic]", "isotonic, beta]")))
    with pytest.raises(ValueError, match="'uncalibrated' must be listed"):
        load_calibration_config(write(tmp_path, GOOD.replace("[uncalibrated, platt, isotonic]", "[platt]")))
    with pytest.raises(ValueError, match="must list exactly"):
        load_calibration_config(write(tmp_path, GOOD.replace("[uncalibrated, platt, isotonic]", "[uncalibrated, platt]")))
    with pytest.raises(ValueError, match="missing bootstrap.seed"):
        load_calibration_config(write(tmp_path, GOOD.replace("seed: 42, confidence", "confidence")))
    with pytest.raises(ValueError, match="missing section 'metrics'"):
        load_calibration_config(write(tmp_path, GOOD.replace("metrics: {reliability_bins: 10}\n", "")))
    with pytest.raises(ValueError, match="Cannot read"):
        load_calibration_config(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="mapping"):
        load_calibration_config(write(tmp_path, "- just\n- a list\n"))

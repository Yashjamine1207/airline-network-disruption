"""What the three generated reports must say, given tables with known numbers (Phase 7, Step 13)."""

import json

import pandas as pd
import pytest

from airline_disruption.reporting import calibration_report, error_report, explain_report
from airline_disruption.reporting.common import load_evidence
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.text import check_clean


def build(root, module):
    inp = Inputs(root=root)
    return module.build(inp, load_evidence(inp)), inp


# ============================================================================================ all reports
@pytest.mark.parametrize("module", [calibration_report, explain_report, error_report])
def test_every_report_passes_the_writing_rules_and_names_the_lock(report_root, module, lock_hash):
    text, _ = build(report_root, module)
    check_clean(text)                                    # no long dashes, no nan%, no placeholders
    assert lock_hash[:16] in text
    assert "2026-10-01T09:00:00+00:00" in text           # first run time from the consumed marker
    assert "Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience" in text
    assert "not an airline operating system" in text
    assert "What the evidence does not support" in text and "aircraft or tail-number propagation" in text and "weather" in text


@pytest.mark.parametrize("module", [calibration_report, explain_report, error_report])
def test_a_smoke_build_carries_a_loud_banner(make_root, module):
    root = make_root("smoke", smoke=True)
    inp = Inputs(root=root, lock_path="reports/final/final_protocol_lock_smoke.json")
    text = module.build(inp, load_evidence(inp))
    assert "TEST BUILD from smoke tables" in text


@pytest.mark.parametrize("module", [calibration_report, explain_report, error_report])
def test_a_real_build_has_no_smoke_banner(report_root, module):
    text, _ = build(report_root, module)
    assert "TEST BUILD" not in text


def test_the_inputs_section_lists_what_was_read_and_what_was_not_found(report_root):
    (report_root / "reports/tables/phase7a_case_studies.csv").unlink()
    text, inp = build(report_root, explain_report)
    assert "reports/tables/phase7a_case_studies.csv" in inp.missing
    assert "Files that were looked for and not found" in text
    assert "The case-study table was not found" in text


# ============================================================================================ calibration and alert policy
def test_the_protocol_section_states_the_locked_calibrator_from_the_lock(report_root):
    text, _ = build(report_root, calibration_report)
    assert "337,156 early-stopping rows with 8,154 events (event rate 2.40%)" in text
    assert "a = 0.509, b = 1.078" in text
    assert "Rule 1 chose **platt**" in text and "the same rows as in the validation experiment" in text
    assert "No calibrator was fitted or chosen with 2023 labels." in text
    assert "The model was not refitted on 2022" in text


def test_the_scope_describes_only_the_features_the_locked_model_has(report_root):
    text, _ = build(report_root, calibration_report)
    assert "prior calendar month" not in text and "previous calendar month" not in text
    assert "carrier, origin, destination and route" in text


def test_level_drift_is_stated_when_the_event_rate_rose_and_the_ratio_fell(report_root):
    text, _ = build(report_root, calibration_report)
    assert "2.40% in the rows the calibrator was fitted on, 2.67% in 2022 and 3.19% in 2023" in text
    assert "0.930 on 2022 and 0.830 on 2023" in text
    assert "under-predicts by more" in text


def test_level_drift_is_not_claimed_when_the_ratio_did_not_fall(make_root):
    root = make_root("flat", ratio23=0.95, prev23=0.0267)
    text, _ = build(root, calibration_report)
    assert "under-predicts by more" not in text
    assert "0.930 on 2022 and 0.950 on 2023" in text


def test_calibration_with_different_pr_auc_between_variants_is_refused(report_root):
    path = report_root / "reports/tables/phase7b_final_severe_delay_120_calibration.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["variant"] == "uncalibrated", "pr_auc"] += 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cannot change a ranking"):
        build(report_root, calibration_report)


def test_the_review_tables_carry_the_counts_from_the_capacity_file(report_root):
    text, _ = build(report_root, calibration_report)
    events = int(0.0319 * 454413)                                      # how the fixture builds its event count
    assert f"(454,413 flights, {events:,} severe delays)" in text
    assert "| 1% | 4,544 |" in text                                    # ceil-free round of 1% of 454,413
    assert "within each day" in text.lower() and "within each month" in text.lower()
    assert "Rankings use the uncalibrated score" in text


def test_the_season_sentence_names_the_busiest_month_and_prefers_the_within_day_figure(report_root):
    text, _ = build(report_root, calibration_report)
    assert "Season: the three months with the most alerts" in text and "2023-03" in text
    assert "Quote the within-day figure for ranking skill." in text


def test_fixed_cutoffs_that_flag_more_flights_are_reported_with_the_factor(report_root):
    text, _ = build(report_root, calibration_report)
    assert "flagged 12.8% of 2023 flights (1.28 times as many)" in text
    assert "Scores rose between the two years" in text


def test_fixed_cutoffs_that_hold_their_workload_are_not_called_drift(make_root):
    root = make_root("steady", ten_pct_alert_2023=0.102)
    text, _ = build(root, calibration_report)
    assert "held within 5%" in text and "Scores rose between the two years" not in text


def test_the_penalty_section_says_the_weights_are_assumptions_and_gives_a_break_even(report_root):
    text, _ = build(report_root, calibration_report)
    assert "not airline costs" in text and "nothing here estimates a saving" in text
    assert "Break-even (false alerts per event caught)" in text and "ratio 20" in text
    assert "Whether R is anywhere near these values is not something this data can say." in text


def test_cancellation_failure_is_reported_as_a_negative_result(report_root):
    text, _ = build(report_root, calibration_report)
    assert "reported as a negative result" in text
    assert "The baseline is better, with the whole interval below zero." in text
    assert "Ranking inside a single day does beat the event rate" in text
    assert "No calibration, threshold or alert policy was built for it." in text
    assert "0.0176" in text and "0.0208" in text                         # model and route-history PR-AUC


def test_a_cancellation_model_that_passes_is_not_called_a_negative_result(make_root):
    root = make_root("good", cancel_claims=(True, True, True))
    text, _ = build(root, calibration_report)
    assert "reported as a negative result" not in text


def test_the_limits_count_the_stray_rows_of_both_targets(report_root):
    text, _ = build(report_root, calibration_report)
    assert "353 rows across both targets" in text                        # 175 + 178


def test_claims_are_listed_with_their_outcome(report_root):
    text, _ = build(report_root, calibration_report)
    assert "**C1_skill_over_chance**: supported." in text
    assert "**C2_beats_history**: not supported." in text                # the cancellation block


# ============================================================================================ explainability
def test_the_global_table_lists_features_in_the_order_given_with_their_shares(report_root):
    text, _ = build(report_root, explain_report)
    assert "| carrier_identifier | carrier | 0.1500 | 25.0% |" in text
    assert "The largest family is calendar and clock at 47.0%." in text


def test_the_text_does_not_mention_features_the_locked_model_does_not_have(report_root):
    text, _ = build(report_root, explain_report)
    assert "prior-month volume" not in text


def test_category_effects_show_five_highest_and_five_lowest_levels(report_root):
    text, _ = build(report_root, explain_report)
    assert text.count("| highest |") == 5 and text.count("| lowest |") == 5


def test_cases_show_the_typical_case_of_each_category_and_both_extremes(report_root):
    text, _ = build(report_root, explain_report)
    assert "Correct high-risk alert (top 1%, severe)" in text and "Missed severe delay (outside the top 10%)" in text
    assert text.count("Most confident false alarm") >= 1 and text.count("Most confident miss") >= 1
    assert "never as a model input" in text


def test_without_perturbation_tables_the_section_says_so_and_still_lists_rejected_challengers(report_root):
    text, _ = build(report_root, explain_report)
    assert "No sequence perturbation tables were found." in text
    assert "lstm: did not beat the MLP control (DEC-020)" in text
    assert "none was scored on 2023" in text


def test_perturbation_results_are_summarised_by_how_many_moved_pr_auc_clearly(report_root):
    table = pd.DataFrame([
        {"arch": "lstm", "perturbation": "baseline", "description": "unchanged", "rows": 100, "events": 5, "pr_auc": 0.05, "delta_pr_auc": None, "ci_low": None, "ci_high": None, "verdict": None},
        {"arch": "lstm", "perturbation": "no_sequence", "description": "lookback removed", "rows": 100, "events": 5, "pr_auc": 0.04, "delta_pr_auc": -0.01, "ci_low": -0.02, "ci_high": -0.005,
         "verdict": "clear drop: the whole interval is below zero"},
        {"arch": "lstm", "perturbation": "shuffled", "description": "shuffled", "rows": 100, "events": 5, "pr_auc": 0.05, "delta_pr_auc": 0.0001, "ci_low": -0.001, "ci_high": 0.001,
         "verdict": "no clear change: the interval includes zero"}])
    table.to_csv(report_root / "reports/tables/phase7a_sequence_perturbation_lstm.csv", index=False)
    (report_root / "reports/tables/phase7a_sequence_perturbation_lstm_decision.json").write_text(json.dumps(
        {"rows_scored": 100, "period": "2022 second half", "lookback_bins": 16, "reproduction_check": {"pr_auc_difference": 0.0}}))
    text, _ = build(report_root, explain_report)
    assert "1 of 2 input changes moved PR-AUC clearly" in text and "no_sequence" in text.split("moved PR-AUC clearly")[1][:80]
    assert "**LSTM**" in text


def test_the_link_to_the_errors_quotes_hour_and_month_shares_from_the_table(report_root):
    text, _ = build(report_root, explain_report)
    assert "departure hour 30.0%" in text and "month 17.0%" in text


def test_explanations_are_described_as_associations_and_shap_is_said_to_be_validation_only(report_root):
    text, _ = build(report_root, explain_report)
    assert "It is an association inside the model. It is not a cause of delay" in text
    assert "SHAP was computed on validation data" in text and "scoring only" in text


# ============================================================================================ error analysis
def test_regimes_include_each_backtest_year_and_the_locked_model_with_its_fit_rows(report_root):
    text, _ = build(report_root, error_report)
    assert "| 2020 shock | 12 |" in text and "1.27" in text
    assert "2023 final test (locked model) | 53 | 1,454,352 |" in text
    assert "2020 is kept as a distribution-shift regime, not removed." in text


def test_the_time_tables_are_labelled(report_root):
    text, _ = build(report_root, error_report)
    assert "2022, by quarter:" in text and "2023, by month:" in text


def test_the_morning_gap_is_quantified_from_the_hour_table(report_root):
    text, _ = build(report_root, error_report)
    assert "Departures from 06:00 to 09:59 are 23% of the 2023 flights and 14% of the severe events. A top 10% review catches 1.0% of them." in text


def test_carriers_without_alerts_are_counted_with_correct_grammar(report_root):
    text, _ = build(report_root, error_report)
    assert "1 carrier received no top 1% alert. The largest of them, C1," in text


def test_the_decile_check_reads_the_table_not_an_assumption(report_root):
    text, _ = build(report_root, error_report)
    assert "The event rate rises with every decile" in text


def test_recall_that_does_not_rise_with_delay_size_is_said_plainly(report_root):
    text, _ = build(report_root, error_report)
    assert "It does not rise steadily with the size of the delay" in text


def test_cancellation_section_names_the_month_that_holds_the_alerts(report_root):
    text, _ = build(report_root, error_report)
    assert "Both pre-registered tests of skill failed on 2023." in text
    assert "of the top 1% alerts fall in 2023-03" in text
    assert "Ranking inside a single day does beat the event rate (claim C3 holds)" in text


def test_open_items_come_from_the_explanation_decision_file(report_root):
    text, _ = build(report_root, error_report)
    assert "2019 and 2020 regimes" in text and "held-out carrier or airport experiments" in text


@pytest.mark.parametrize("claims", [(False, True, True), (True, False, True)])
def test_a_cancellation_model_that_fails_only_one_claim_is_not_called_a_negative_result(make_root, claims):
    root = make_root("one_failed", cancel_claims=claims)
    text, _ = build(root, calibration_report)
    assert "reported as a negative result" not in text
    root_errors = make_root("one_failed_errors", cancel_claims=claims)
    errors_text, _ = build(root_errors, error_report)
    assert "Both pre-registered tests of skill failed on 2023." not in errors_text


def test_a_within_day_ranking_that_fails_is_not_described_as_beating_the_event_rate(make_root):
    root = make_root("no_c3", cancel_claims=(False, False, False))
    text, _ = build(root, calibration_report)
    assert "Ranking inside a single day does not beat the event rate either." in text
    errors_text, _ = build(make_root("no_c3_errors", cancel_claims=(False, False, False)), error_report)
    assert "Ranking inside a single day does not beat the event rate either." in errors_text


def test_the_season_sentence_changes_when_the_whole_period_is_not_better_than_within_day(report_root):
    path = report_root / "reports/tables/phase7b_final_severe_delay_120_capacity.csv"
    frame = pd.read_csv(path)
    mask = (frame["period"] == "final_2023") & (frame["scheme"] == "whole_year") & (frame["capacity"] == "1%")
    frame.loc[mask, "precision"] = 0.01
    frame.to_csv(path, index=False)
    text, _ = build(report_root, calibration_report)
    assert "season and weekday add nothing to the ranking here" in text
    assert "comes from knowing which months and days are busy" not in text


def test_a_penalty_table_where_small_reviews_are_not_cheaper_says_so(report_root):
    path = report_root / "reports/tables/phase7b_final_severe_delay_120_utility.csv"
    frame = pd.read_csv(path)
    mask = (frame["period"] == "final_2023") & (frame["scheme"] == "by_day")
    frame.loc[mask & (frame["capacity"] == "0.5%"), "break_even_ratio"] = 50.0
    frame.loc[mask & (frame["capacity"] == "10%"), "break_even_ratio"] = 10.0
    frame.to_csv(path, index=False)
    text, _ = build(report_root, calibration_report)
    assert "does not fall as the review gets smaller" in text


def test_a_delay_size_table_where_recall_rises_is_said_to_rise(report_root):
    path = report_root / "reports/tables/phase7b_final_severe_delay_120_severity.csv"
    frame = pd.read_csv(path)
    frame["recall_top10pct"] = [0.10, 0.20, 0.30]
    frame.to_csv(path, index=False)
    text, _ = build(report_root, error_report)
    assert "It rises with every band, so longer delays are found more often." in text


def test_deciles_that_do_not_rise_are_not_called_rising(report_root):
    path = report_root / "reports/tables/phase7b_final_severe_delay_120_by_group.csv"
    frame = pd.read_csv(path)
    mask = (frame["dimension"] == "score_decile") & (frame["group"] == "decile 05")
    frame.loc[mask, "prevalence"] = 0.001
    frame.to_csv(path, index=False)
    text, _ = build(report_root, error_report)
    assert "does not rise with every decile" in text


def table_lines(text: str, marker: str) -> list[str]:
    """The data rows of the first markdown table after ``marker``."""
    after = text.split(marker, 1)[1].split("\n")
    rows, started = [], False
    for line in after:
        if line.startswith("|"):
            started = True
            rows.append(line)
        elif started:
            break
    return rows[2:]


def test_each_review_table_holds_only_its_own_scheme_and_four_capacities(report_root):
    text, _ = build(report_root, calibration_report)
    whole = table_lines(text, "**Whole period (one ranking over every flight), 2023**")
    month = table_lines(text, "**Within each month, 2023**")
    day = table_lines(text, "**Within each day, 2023**")
    assert [len(whole), len(month), len(day)] == [4, 4, 4]
    assert all("| 2.00 |" in row for row in whole)          # the fixture gives whole-period lift 2.0
    assert all("| 1.80 |" in row for row in month + day)     # and 1.8 for the month and day schemes


def test_the_season_sentence_uses_the_three_busiest_months(report_root):
    text, _ = build(report_root, calibration_report)
    assert "hold 98% of those alerts while holding 75% of the flights and 78% of the severe events" in text


def test_drift_needs_a_higher_event_rate_as_well_as_a_lower_ratio(make_root):
    root = make_root("same_rate", ratio23=0.80, prev23=0.0267)
    text, _ = build(root, calibration_report)
    assert "under-predicts by more" not in text


def test_the_global_importance_table_shows_ten_features_at_most(report_root):
    path = report_root / "reports/tables/phase7a_shap_global.csv"
    frame = pd.read_csv(path)
    extra = pd.DataFrame([{"feature": f"f{i}", "mean_abs_shap": 0.001, "share_of_total": 0.001, "family": "route"} for i in range(12)])
    pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)
    text, _ = build(report_root, explain_report)
    rows = table_lines(text, "## Global importance")
    assert len(rows) == 10 and rows[0].startswith("| carrier_identifier |")

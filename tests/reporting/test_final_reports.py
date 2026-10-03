"""What the final model-selection report and the final technical report must say, given tables with known numbers (Phase 7B, Step 14)."""

import json

import pandas as pd
import pytest

from airline_disruption.reporting import error_report, selection_report, technical_report
from airline_disruption.reporting.common import load_evidence
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.text import check_clean

NEW = [selection_report, technical_report]
SEQUENCE = {"entity": "origin airport schedule (DEC-017)", "lookback_bins": 16, "bin_hours": 3, "lookback_hours": 48,
            "channels": ["scheduled_departures", "scheduled_arrivals"]}


def build(root, module):
    inp = Inputs(root=root)
    return module.build(inp, load_evidence(inp)), inp


def add_deep(root, arch, tag="final", smoke=False, created="2026-09-30T10:00:00+00:00", pr=0.031, with_validation=True, architecture=None, sequence=SEQUENCE):
    folder = root / "reports/tables"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"phase6_deep_{arch}_{tag}"
    decision = {"architecture": architecture or arch, "tag": tag, "smoke_run": smoke, "created_utc": created, "sequence": sequence}
    (folder / f"{stem}_decision.json").write_text(json.dumps(decision), encoding="utf-8")
    if with_validation:
        rows = [{"model": m, "period": "2022 full year", "rows": 600000, "events": 12000, "prevalence": 0.02, "pr_auc": p, "pr_auc_lift": p / 0.02, "roc_auc": 0.6}
                for m, p in ((arch, pr), ("lightgbm_base_no_year", 0.034))]
        pd.DataFrame(rows).to_csv(folder / f"{stem}_validation.csv", index=False)


def edit_csv(root, relative, change):
    path = root / relative
    frame = pd.read_csv(path)
    change(frame)
    frame.to_csv(path, index=False)


# ============================================================================================ both reports
@pytest.mark.parametrize("module", NEW)
def test_both_reports_pass_the_writing_rules_and_carry_the_standard_blocks(report_root, module, lock_hash):
    text, _ = build(report_root, module)
    check_clean(text)
    assert lock_hash[:16] in text and "not an airline operating system" in text
    assert "What the evidence does not support" in text and "aircraft or tail-number propagation" in text
    assert "Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience" in text


@pytest.mark.parametrize("module", NEW)
def test_a_smoke_build_carries_a_loud_banner(make_root, module):
    root = make_root("smoke_final", smoke=True)
    inp = Inputs(root=root, lock_path="reports/final/final_protocol_lock_smoke.json")
    assert "TEST BUILD from smoke tables" in module.build(inp, load_evidence(inp))


@pytest.mark.parametrize("module", NEW)
def test_a_real_build_has_no_smoke_banner(report_root, module):
    assert "TEST BUILD" not in build(report_root, module)[0]


# ============================================================================================ decision paragraph
def test_all_claims_supported_is_stated_with_the_numbers_from_the_tables(report_root):
    text, _ = build(report_root, selection_report)
    assert "All three pre-registered claims were supported" in text
    assert "PR-AUC was 0.0590 against an event rate of 3.19% (lift 1.85" in text
    assert "baseline_carrier_rate" in text and "+0.0150" in text


def test_a_failed_claim_is_named_and_the_all_three_sentence_is_gone(make_root):
    text, _ = build(make_root("c2", severe_claims=(True, False, True)), selection_report)
    assert "2 of 3 pre-registered claims were supported" in text and "Not supported: C2_beats_history" in text
    assert "All three pre-registered claims" not in text


def test_the_remark_about_missed_events_depends_on_recall(report_root, make_root):
    assert "Most severe delays are not in a top-10% list" in build(report_root, selection_report)[0]
    root = make_root("highrecall")
    def raise_recall(frame):
        frame.loc[(frame["period"] == "final_2023") & (frame["scheme"] == "by_day") & (frame["capacity"] == "10%"), "recall"] = 0.6
    edit_csv(root, "reports/tables/phase7b_final_severe_delay_120_capacity.csv", raise_recall)
    assert "Most severe delays are not in a top-10% list" not in build(root, selection_report)[0]


def test_cancellation_with_no_skill_is_a_negative_result_in_both_reports(report_root):
    selection, _ = build(report_root, selection_report)
    technical, _ = build(report_root, technical_report)
    assert "Cancellation: no model is selected" in selection
    assert "Cancellation: no model was selected" in technical and "cancellation: no demonstrated skill" in technical


def test_cancellation_that_shows_skill_is_not_called_a_negative_result(make_root):
    root = make_root("cancel_ok", cancel_claims=(True, False, True))
    selection, _ = build(root, selection_report)
    technical, _ = build(root, technical_report)
    assert "no model is selected" not in selection and "passed 2 of 3 claims" in selection
    assert "no model was selected" not in technical and "cancellation: the baseline passed at least one" in technical


# ============================================================================================ record and evidence
def test_the_record_holds_every_field_the_decision_must_state(report_root):
    text, _ = build(report_root, selection_report)
    for needle in ("| Target |", "| Eligible cohort |", "| Prediction timestamp | scheduled departure UTC minus 2 hours |", "| Feature version |", "| Sequence definition |",
                   "| Training period |", "| Calibration period |", "| Validation period |", "| Final-test period |", "| Selected model |", "| Rejected challengers | lstm, transformer |"):
        assert needle in text, needle
    assert "12 features" in text and "1,454,352 rows" in text and "337,156 rows" in text and "53 trees" in text and "num_leaves 31" in text
    assert "`eligible_severe_delay`" in text and "phase6_cohort_v1.parquet" in text


def test_mae_is_marked_not_applicable_and_the_phase4_tables_are_pointed_to(report_root):
    text, _ = build(report_root, selection_report)
    assert "MAE: not applicable to this classifier" in text and "phase4_*_regression_validation.csv" in text


def test_both_periods_are_tabulated_with_all_scorers(report_root):
    text, _ = build(report_root, selection_report)
    assert "**2022 validation" in text and "**2023 final test" in text
    assert text.count("| baseline_carrier_rate |") == 2 and text.count("| lightgbm_locked |") == 2


def test_the_calibration_drift_sentence_needs_a_lower_ratio_and_a_higher_rate(report_root, make_root):
    assert "too low by more than on 2022" in build(report_root, selection_report)[0]
    assert "too low by more than on 2022" not in build(make_root("nodrift", ratio23=0.95), selection_report)[0]


def test_the_calibrator_fit_period_comes_from_the_lock(report_root):
    assert "fitted on 2021-07 to 2021-12" in build(report_root, selection_report)[0]


def test_the_ranking_section_gives_the_within_day_table_first_and_says_which_score_is_used(report_root):
    text, _ = build(report_root, selection_report)
    assert text.index("Inside each day") < text.index("Over the whole period")
    assert "uncalibrated scores" in text and "Quote the within-day figure" in text


def test_a_complexity_verdict_follows_the_history_claim(report_root, make_root):
    assert "was kept because it beat the best history baseline" in build(report_root, selection_report)[0]
    failed, _ = build(make_root("c2fail", severe_claims=(True, False, True)), selection_report)
    assert "did not clearly beat the best history baseline" in failed and "was kept because" not in failed


def test_unseen_category_rows_are_tabulated(report_root):
    text, _ = build(report_root, selection_report)
    assert "| unseen category | 2,000 |" in text and "| complete | 28,000 |" in text


# ============================================================================================ challengers from Phase 6 files
def test_without_phase6_files_no_challenger_pr_auc_is_invented(report_root):
    text, _ = build(report_root, selection_report)
    assert "were not found by this builder, so no PR-AUC is restated" in text
    assert "origin-airport schedule sequence (DEC-017); the run files" in text


def test_a_found_run_adds_its_pr_auc_and_the_lightgbm_reference_row(report_root):
    add_deep(report_root, "lstm", pr=0.031)
    text, inp = build(report_root, selection_report)
    assert "| lstm | lstm | 2022 full year | 0.0310 |" in text and "| lstm | LightGBM in the same file | 2022 full year | 0.0340 |" in text
    assert "reports/tables/phase6_deep_lstm_final_decision.json" in inp.used and "reports/tables/phase6_deep_lstm_final_validation.csv" in inp.used


def test_the_sequence_definition_is_read_from_the_run(report_root):
    add_deep(report_root, "gru")
    text, _ = build(report_root, selection_report)
    assert "origin airport schedule (DEC-017): 16 bins of 3 hours (48 hours of lookback)" in text and "scheduled_departures, scheduled_arrivals" in text


@pytest.mark.parametrize("options", [
    {"smoke": True}, {"with_validation": False}, {"architecture": "gru"},
], ids=["smoke run", "no validation table", "decision names another architecture"])
def test_a_run_that_is_not_a_real_result_is_ignored(report_root, options):
    add_deep(report_root, "lstm", **options)
    text, inp = build(report_root, selection_report)
    assert "were not found by this builder" in text
    assert not any("phase6_deep_lstm" in path for path in inp.used)


def test_the_latest_non_smoke_run_wins(report_root):
    add_deep(report_root, "lstm", tag="old", created="2026-09-01T00:00:00+00:00", pr=0.020)
    add_deep(report_root, "lstm", tag="new", created="2026-09-20T00:00:00+00:00", pr=0.031)
    add_deep(report_root, "lstm", tag="newest_smoke", created="2026-09-30T00:00:00+00:00", smoke=True, pr=0.099)
    text, _ = build(report_root, selection_report)
    assert "0.0310" in text and "0.0200" not in text and "0.0990" not in text


def test_paired_verdicts_from_a_run_are_listed(report_root):
    add_deep(report_root, "transformer")
    pd.DataFrame([{"challenger": "transformer", "reference": "lightgbm_base_no_year", "period": "2022 full year", "pr_auc_reference": 0.034, "pr_auc_challenger": 0.02,
                   "difference": -0.014, "ci_low": -0.02, "ci_high": -0.01, "verdict": "lightgbm_base_no_year is better: the whole interval is below zero."}]
                 ).to_csv(report_root / "reports/tables/phase6_deep_transformer_final_comparisons.csv", index=False)
    text, _ = build(report_root, selection_report)
    assert "transformer against lightgbm_base_no_year, 2022 full year: lightgbm_base_no_year is better" in text


def test_the_2020_ablation_table_appears_only_when_the_file_exists(report_root):
    assert "Training with the 2020 shock year" not in build(report_root, selection_report)[0]
    pd.DataFrame([{"challenger": "no2020", "reference": "full", "period": "2022 full year", "pr_auc_reference": 0.0345, "pr_auc_challenger": 0.0340, "difference": -0.0005,
                   "ci_low": -0.0008, "ci_high": -0.0003, "verdict": "full is better than no2020: the whole 95% interval is below zero."}]
                 ).to_csv(report_root / selection_report.ABLATION_FILE, index=False)
    text, _ = build(report_root, selection_report)
    assert "Training with the 2020 shock year" in text and "full is better than no2020" in text and "(-0.00080 to -0.00030)" in text


# ============================================================================================ technical report
def test_the_question_table_has_ten_rows_and_takes_answers_from_tables(report_root):
    text, _ = build(report_root, technical_report)
    table = [l for l in text.splitlines() if l.startswith("| ") and "---" not in l and "Question" not in l and "Answer from the tables" not in l][:10]
    assert len(table) == 10
    assert "calendar and clock (47%)" in text
    assert "Predicted-to-observed ratio 0.930 on 2022 and 0.830 on 2023" in text
    assert "the top 1% caught 1.8% of severe delays at 5.7% precision" in text


def test_files_written_by_the_same_run_are_not_called_missing_but_absent_earlier_reports_are(report_root):
    text, _ = build(report_root, technical_report)
    assert "`reports/evaluation/calibration_and_alert_policy_report.md`" in text and "calibration_and_alert_policy_report.md` (not found)" not in text
    assert "`reports/evaluation/network_resilience_report.md` (not found)" in text
    (report_root / "reports/evaluation").mkdir(parents=True, exist_ok=True)
    (report_root / "reports/evaluation/network_resilience_report.md").write_text("x", encoding="utf-8")
    assert "network_resilience_report.md` (not found)" not in build(report_root, technical_report)[0]


def test_the_not_done_list_separates_not_scored_from_not_done(report_root):
    text, _ = build(report_root, technical_report)
    scored, done = text.index("Not scored on the 2023 final test"), text.index("Not done at all")
    assert scored < done
    assert "airport recovery episodes" in text[scored:done] and "airport episodes with fast and slow recovery" in text[done:]
    assert "airport recovery episodes" not in text[done:text.index("## Reproduction")].replace("airport episodes with fast and slow recovery", "")


def test_case_studies_leave_the_not_done_lists_once_their_tables_exist(report_root):
    for name in ("regression_large_error_cases.csv", "airport_recovery_episode_cases.csv"):
        (report_root / "reports/tables" / name).write_text("a\n1\n", encoding="utf-8")
    technical, _ = build(report_root, technical_report)
    errors, _ = build(report_root, error_report)
    assert "fast and slow recovery" not in technical and "large regression errors" not in technical
    assert "were NOT built" not in errors


def test_the_error_report_says_plainly_which_required_case_studies_were_not_built(report_root):
    text, _ = build(report_root, error_report)
    assert "were NOT built" in text and "incomplete against that plan" in text
    assert "large regression errors" in text and "airport episodes with fast and slow recovery" in text


def test_one_missing_case_study_is_listed_alone(report_root):
    (report_root / "reports/tables/regression_large_error_cases.csv").write_text("a\n1\n", encoding="utf-8")
    text, _ = build(report_root, error_report)
    assert "airport episodes with fast and slow recovery" in text and "large regression errors" not in text


def test_the_reproduction_section_lists_packages_and_hardware_from_the_decision(report_root):
    text, _ = build(report_root, technical_report)
    assert "lightgbm 4.7.0, pandas 2.2.3" in text and "cpu_logical_cores 16" in text and "--confirm USE-2023-ONCE" in text


def test_the_technical_summary_counts_rejected_challengers_from_the_lock(report_root):
    assert "2 sequence and neural challengers were rejected on 2022 evidence" in build(report_root, technical_report)[0]


def test_the_technical_summary_follows_the_claims(report_root, make_root):
    assert "Severe delay: all three pre-registered claims were supported" in build(report_root, technical_report)[0]
    failed, _ = build(make_root("tech_c2", severe_claims=(True, False, True)), technical_report)
    assert "Severe delay: 2 of 3 pre-registered claims were supported (not supported: C2_beats_history)" in failed
    assert "all three pre-registered claims were supported" not in failed

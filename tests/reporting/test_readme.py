"""The generated README.md (Phase 7B, Step 15) and the way the build script writes it."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from airline_disruption.reporting import readme_report
from airline_disruption.reporting.common import load_evidence
from airline_disruption.reporting.inputs import Inputs
from airline_disruption.reporting.text import check_clean

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_phase7_reports.py"


def build(root):
    inp = Inputs(root=root)
    return readme_report.build(inp, load_evidence(inp))


def run(root, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *extra], capture_output=True, text=True, cwd=ROOT)


def touch(root, relative, text="x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ============================================================================================ content
def edit_csv(root, relative, change):
    path = root / relative
    frame = pd.read_csv(path)
    change(frame)
    frame.to_csv(path, index=False)


def test_the_readme_starts_with_the_marker_and_the_title_and_passes_the_writing_rules(report_root):
    text = build(report_root)
    check_clean(text)
    assert text.startswith(readme_report.MARKER + "\n# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience")
    assert "advanced portfolio study for Data Science" in text and "not an airline operating system" in text
    assert "It has no API, no deployment, no streaming and no live airline data." in text


def test_the_sections_come_in_the_planned_order(report_root):
    text = build(report_root)
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## The project in plain words", "## Project outcome", "## Words used in this README", "## Questions this project answers", "## Dataset",
                        "## Methodology", "## Advanced experiments", "## Explainability and error analysis", "## Results in more detail", "## Repository structure",
                        "## Key reports", "## Figures", "## Reproducibility", "## Limitations", "## Portfolio claim", "## Licence"]


def test_the_plain_words_result_follows_the_claims(report_root, make_root):
    assert "The model ranks severe delays better than chance and better than the strongest simple rule." in build(report_root)
    failed = build(make_root("r_c2", severe_claims=(True, False, True)))
    assert "The model passed 2 of 3 checks that were fixed in advance. It did not pass: beats the strongest history rule." in failed
    assert "ranks severe delays better than chance" not in failed


def test_the_numbers_come_from_the_tables(report_root):
    text = build(report_root)
    assert "Its PR-AUC is 0.0590, which is 1.85 times what a random ranking would score." in text
    assert "(3.19% of the 2023 flights in this data)" in text
    for row in ("| Flights scored | 100,000 |", "| Share of flights severely delayed | 3.19% |", "| PR-AUC of the model | 0.0590 (0.0531 to 0.0649) |",
                "| PR-AUC of the strongest simple rule (carrier history) | 0.0430 |", "| Lift over chance | 1.85 |",
                "| Model | LightGBM with 12 schedule, carrier, airport and route features, locked before 2023 was read |"):
        assert row in text, row


def test_the_review_table_matches_the_capacity_table(report_root):
    capacity = pd.read_csv(report_root / "reports/tables/phase7b_final_severe_delay_120_capacity.csv")
    row = capacity[(capacity["period"] == "final_2023") & (capacity["scheme"] == "by_day") & (capacity["capacity"] == "1%")].iloc[0]
    text = build(report_root)
    expected = f"| 1% | {int(row.reviewed):,} | {int(row.events_captured):,} | {row.recall * 100:.1f}% | {row.precision * 100:.1f}% | {int(row.false_alerts):,} |"
    assert expected in text
    assert f"The top 1% list catches {int(row.events_captured):,} severe delays and holds {int(row.false_alerts):,} false alerts." in text
    ten = capacity[(capacity["period"] == "final_2023") & (capacity["scheme"] == "by_day") & (capacity["capacity"] == "10%")].iloc[0]
    assert f"the top 10% list leaves {int(ten.events_missed):,} severe delays outside it and holds {int(ten.false_alerts):,} flights" in text


def test_most_severe_delays_missed_is_said_only_while_recall_is_below_half(report_root, make_root):
    assert "most severe delays (82%) are not in its top 10% of flights each day" in build(report_root)
    high = make_root("r_recall")

    def raise_recall(frame):
        frame.loc[(frame["period"] == "final_2023") & (frame["scheme"] == "by_day") & (frame["capacity"] == "10%"), "recall"] = 0.6

    edit_csv(high, "reports/tables/phase7b_final_severe_delay_120_capacity.csv", raise_recall)
    assert "are not in its top 10% of flights each day" not in build(high)


def test_cancellation_sentence_depends_on_the_claims(report_root, make_root):
    assert "Cancellations: no model was selected" in build(report_root)
    positive = build(make_root("r_cancel", cancel_claims=(True, False, True)))
    assert "Cancellations: no model was selected" not in positive and "stays a ranking-only baseline" in positive


def test_the_readme_states_the_scope_change_the_data_the_time_rule_and_the_limits(report_root):
    text = build(report_root)
    assert "The final pipeline uses flight data only" in text and "data/raw/flights_kaggle/flights_sample_3m.csv" in text
    link = "https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023"
    assert f"[Flight Delay and Cancellation Dataset 2019 to 2023]({link})" in text and f"Download the flight file from [Kaggle]({link})" in text
    assert "The full data are not in this repository" in text
    assert "Prediction timestamp: scheduled departure UTC minus 2 hours" in text
    assert "no usable aircraft identifier" in text and "The evidence does not support:" in text and "Aircraft or tail-number propagation" in text
    assert "Real airline costs, savings or operating decisions" in text


def test_the_ten_research_questions_are_listed_and_numbered(report_root):
    section = build(report_root).split("## Questions this project answers")[1].split("## Dataset")[0]
    numbered = [line for line in section.splitlines() if line[:1].isdigit()]
    assert len(numbered) == 10 and numbered[0].startswith("1. ") and numbered[-1].startswith("10. ")
    assert "How reliable are the probabilities across operating regimes?" in section


def test_the_periods_come_from_the_lock(report_root):
    text = build(report_root)
    assert "Fit: 2019-01 to 2021-06. Early stopping: 2021-07 to 2021-12. Validation: 2022. Final test: 2023-01-01 to 2023-08-31." in text
    assert "We trained on flights from 2019-01 to 2021-06, stopped training early using 2021-07 to 2021-12, chose between models on 2022" in text


def test_the_calibration_text_follows_the_lock_and_the_selection_file(report_root):
    text = build(report_root)
    assert "no calibration, Platt scaling and isotonic regression" in text
    assert "Platt scaling (a sigmoid curve) was chosen. It was fitted on 2021-07 to 2021-12 data and never on 2023." in text


def test_the_results_section_has_the_scorer_table_and_the_calibration_ratios(report_root):
    text = build(report_root)
    assert "| LightGBM (locked model) | 0.0590 |" in text and "| carrier history | 0.0430 |" in text and "| overall rate | 0.0319 |" in text
    assert "was 0.930 on 2022 and 0.830 on 2023" in text
    assert "No regression model was locked, so none was scored on 2023." in text


def test_the_drift_sentence_appears_only_when_the_ratio_fell_and_the_rate_rose(report_root, make_root):
    assert "The severe-delay rate rose from 2.67% to 3.19%, so the probabilities were too low in 2023." in build(report_root)
    steady = build(make_root("r_steady", ratio23=0.95))
    assert "so the probabilities were too low in 2023" not in steady and "was 0.930 on 2022 and 0.950 on 2023" in steady


def test_the_rejected_challengers_are_listed_with_their_reasons(report_root):
    text = build(report_root)
    assert "2 neural-network models were tested and rejected." in text
    assert "  - `lstm`: did not beat the MLP control (DEC-020)." in text and "  - `transformer`: overfits (DEC-022)." in text


def test_the_claim_and_challenger_sentences_drop_out_when_the_lock_lists_no_challenger(report_root):
    lock_path = report_root / "reports/final/final_protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["rejected_challengers"] = {}
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    text = build(report_root)
    assert "neural-network models were tested and rejected" not in text and "The simpler tabular model won" not in text
    assert "The lock file lists no rejected challenger." in text


def edit_lock(root, change):
    path = root / "reports/final/final_protocol_lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    change(lock)
    path.write_text(json.dumps(lock), encoding="utf-8")


def test_the_calibrator_period_and_feature_count_are_left_out_when_the_lock_lacks_them(report_root):
    def strip(lock):
        del lock["targets"]["severe_delay_120"]["calibrator_primary"]["fit_period"]
        del lock["targets"]["severe_delay_120"]["n_features"]

    edit_lock(report_root, strip)
    text = build(report_root)
    check_clean(text)
    assert "Platt scaling (a sigmoid curve) was chosen." in text and "It was fitted on" not in text
    assert "| Model | LightGBM, locked before 2023 was read |" in text


def test_the_shap_sentence_needs_the_shap_tables(report_root):
    assert "SHAP on 2022 data: the features that moved the model most were" in build(report_root)
    (report_root / "reports/tables/phase7a_shap_global.csv").unlink()
    text = build(report_root)
    assert "SHAP on 2022 data" not in text and "SHAP shows what the model used. It does not show what caused a delay." in text


def test_the_glossary_defines_the_terms_the_readme_uses(report_root):
    section = build(report_root).split("## Words used in this README")[1].split("## Questions")[0]
    for term in ("Severe delay", "Prediction time", "Leakage", "Baseline", "PR-AUC", "Lift", "Precision and recall", "Calibration", "SHAP"):
        assert f"**{term}**" in section


# ============================================================================================ things linked only if they exist
def test_figures_are_embedded_only_when_the_file_exists(report_root):
    text = build(report_root)
    assert "![Reliability of the probabilities on the 2023 final test](reports/figures/final_calibration_curve_severe_delay_120.png)" in text
    assert "shap_summary.png" in text and "local_flight_explanation.png" in text
    assert "final_pr_curve_severe_delay_120.png" not in text and "kaplan_meier_recovery.png" not in text and "airport_network.png" not in text


def test_no_figure_section_when_there_are_no_figures(report_root):
    for png in (report_root / "reports/figures").glob("*.png"):
        png.unlink()
    assert "## Figures" not in build(report_root)


def test_reports_written_by_the_same_run_are_listed_and_the_others_only_when_they_exist(report_root):
    text = build(report_root)
    for name in ("final_technical_report.md", "final_model_selection_report.md", "calibration_and_alert_policy_report.md", "explainability_report.md", "error_analysis_report.md"):
        assert f"(reports/evaluation/{name})" in text
    assert "network_resilience_report.md" not in text and "survival_recovery_report.md" not in text
    touch(report_root, "reports/evaluation/survival_recovery_report.md")
    after = build(report_root)
    assert "(reports/evaluation/survival_recovery_report.md)" in after and "network_resilience_report.md" not in after


def test_the_licence_statement_follows_the_files(report_root):
    before = build(report_root)
    assert "No licence file has been chosen yet" in before and "CITATION.cff" not in before
    touch(report_root, "LICENSE")
    touch(report_root, "CITATION.cff")
    text = build(report_root)
    assert "released under the licence in `LICENSE`" in text and "does not relicense the third-party flight data" in text and "`CITATION.cff`" in text
    assert "No licence file" not in text


def test_documentation_links_appear_only_for_files_that_exist(report_root):
    names = ("docs/data/source_metadata.md", "docs/data/data_card.md", "docs/data/leakage_audit.md", "docs/decisions/limitations_and_appropriate_use.md")
    before = build(report_root)
    for name in names:
        assert name not in before
    for name in names:
        touch(report_root, name)
    after = build(report_root)
    for name in names:
        assert name in after


def test_the_structure_lists_only_folders_that_exist_and_always_ends_with_data(report_root):
    section = build(report_root).split("## Repository structure")[1].split("## Key reports")[0]
    tee, corner = "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 "
    assert "airline-network-disruption-intelligence/" in section
    assert f"{tee}configs/" in section and f"{tee}reports/evaluation/" in section
    assert f"{corner}data/" in section and "Not in Git" in section
    assert "scripts/" not in section and "tests/" not in section
    (report_root / "scripts").mkdir()
    after = build(report_root).split("## Repository structure")[1].split("## Key reports")[0]
    assert f"{tee}scripts/" in after and f"{corner}data/" in after


def test_the_reproduction_steps_are_in_order_and_use_the_python_version_from_the_final_test_record(report_root):
    text = build(report_root)
    assert "The final test ran on Python 3.13.5." in text
    steps = [text.index(marker) for marker in ("### 1. Environment", "### 2. Get the data", "### 3. Run the tests", "### 4. Rebuild the final steps")]
    assert steps == sorted(steps)
    assert "--confirm USE-2023-ONCE" in text and "python scripts/check_repo_hygiene.py" in text


# ============================================================================================ the build script and an existing README
def test_a_run_writes_a_generated_readme_and_lists_it_in_the_manifest(report_root):
    result = run(report_root)
    assert result.returncode == 0, result.stdout + result.stderr
    text = (report_root / "README.md").read_text(encoding="utf-8")
    assert text.startswith(readme_report.MARKER)
    manifest = json.loads((report_root / "reports/tables/phase7_reports_manifest.json").read_text())
    assert "README.md" in manifest["reports"]


def test_a_generated_readme_is_replaced_on_the_next_run(report_root):
    run(report_root)
    (report_root / "README.md").write_text(readme_report.MARKER + "\nold text\n", encoding="utf-8")
    run(report_root)
    assert "old text" not in (report_root / "README.md").read_text(encoding="utf-8")
    assert not (report_root / "README.generated.md").exists()


def test_a_hand_written_readme_is_never_overwritten(report_root):
    mine = "# My own README\n\nWritten by hand.\n"
    (report_root / "README.md").write_text(mine, encoding="utf-8")
    result = run(report_root)
    assert result.returncode == 0
    assert (report_root / "README.md").read_text(encoding="utf-8") == mine
    assert (report_root / "README.generated.md").read_text(encoding="utf-8").startswith(readme_report.MARKER)
    assert "left alone" in result.stdout


def test_a_smoke_build_writes_only_a_smoke_readme(make_root):
    root = make_root("smoke_readme", smoke=True)
    (root / "README.md").write_text("# mine\n", encoding="utf-8")
    result = run(root, "--lock", "reports/final/final_protocol_lock_smoke.json")
    assert result.returncode == 0, result.stdout
    assert (root / "README.md").read_text(encoding="utf-8") == "# mine\n"
    assert not (root / "README.generated.md").exists()
    assert "TEST BUILD from smoke tables" in (root / "README_smoke.md").read_text(encoding="utf-8")


def test_a_failure_while_building_the_readme_writes_nothing(report_root):
    (report_root / "reports/tables/phase7b_final_severe_delay_120_summary.csv").unlink()
    result = run(report_root)
    assert result.returncode == 1 and "CHECK FAILED" in result.stdout
    assert not (report_root / "README.md").exists()

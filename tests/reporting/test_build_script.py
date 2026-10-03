"""scripts/build_phase7_reports.py as a user runs it."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_phase7_reports.py"
REPORT_NAMES = ("calibration_and_alert_policy_report", "explainability_report", "error_analysis_report", "final_model_selection_report", "final_technical_report")


def run(root, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *extra], capture_output=True, text=True, cwd=ROOT)


def test_a_run_writes_five_reports_and_a_manifest_with_the_input_hashes(report_root, lock_hash):
    result = run(report_root)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in REPORT_NAMES:
        assert (report_root / f"reports/evaluation/{name}.md").stat().st_size > 1000
    manifest = json.loads((report_root / "reports/tables/phase7_reports_manifest.json").read_text())
    assert manifest["lock_sha256"] == lock_hash and manifest["smoke"] is False
    assert manifest["reports"] and "reports/tables/phase7b_final_decision.json" in manifest["inputs"]
    assert all(len(h) == 64 for h in manifest["inputs"].values())
    assert "No flight data was loaded" in result.stdout


def test_two_runs_give_byte_identical_reports(report_root):
    run(report_root)
    first = {n: (report_root / f"reports/evaluation/{n}.md").read_bytes() for n in REPORT_NAMES}
    run(report_root)
    second = {n: (report_root / f"reports/evaluation/{n}.md").read_bytes() for n in REPORT_NAMES}
    assert first == second


def test_without_the_consumed_marker_nothing_is_written(report_root):
    (report_root / "reports/final/final_test_consumed.json").unlink()
    result = run(report_root)
    assert result.returncode == 1 and "CHECK FAILED" in result.stdout
    assert not (report_root / "reports/evaluation").exists()


def test_a_lock_hash_mismatch_stops_the_build(report_root):
    path = report_root / "reports/tables/phase7b_final_decision.json"
    decision = json.loads(path.read_text())
    decision["lock_sha256"] = "00" * 32
    path.write_text(json.dumps(decision))
    result = run(report_root)
    assert result.returncode == 1 and "do not agree" in result.stdout
    assert not (report_root / "reports/evaluation").exists()


def test_a_long_dash_in_any_table_text_stops_the_build_and_leaves_no_half_written_set(report_root):
    import pandas as pd
    path = report_root / "reports/tables/phase7a_case_studies.csv"
    frame = pd.read_csv(path)
    frame["top_features"] = "carrier \u2014 high"
    frame.to_csv(path, index=False, encoding="utf-8")
    result = run(report_root)
    assert result.returncode == 1 and "long dash" in result.stdout and "No report was written." in result.stdout
    evaluation = report_root / "reports/evaluation"
    assert not evaluation.exists() or not list(evaluation.glob("*.md"))


def test_a_smoke_build_writes_tagged_files_and_never_the_real_names(make_root):
    root = make_root("smoke_script", smoke=True)
    result = run(root, "--lock", "reports/final/final_protocol_lock_smoke.json")
    assert result.returncode == 0, result.stdout
    for name in REPORT_NAMES:
        assert not (root / f"reports/evaluation/{name}.md").exists()          # the real names stay untouched
        assert (root / f"reports/evaluation/{name}_smoke.md").exists()        # forced tag, although --tag was not given
    assert not (root / "reports/tables/phase7_reports_manifest.json").exists()
    manifest = json.loads((root / "reports/tables/phase7_reports_manifest_smoke.json").read_text())
    assert manifest["smoke"] is True


def test_a_missing_optional_table_is_reported_on_screen_and_in_the_manifest(report_root):
    (report_root / "reports/tables/phase7a_shap_global.csv").unlink()
    result = run(report_root)
    assert result.returncode == 0
    manifest = json.loads((report_root / "reports/tables/phase7_reports_manifest.json").read_text())
    assert "reports/tables/phase7a_shap_global.csv" in manifest["missing_inputs"]
    assert "reports/tables/phase7a_shap_global.csv" in result.stdout

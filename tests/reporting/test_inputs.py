"""Reading tables, recording what was read, and the lock consistency check."""

import json

import pytest

from airline_disruption.reporting.common import load_evidence
from airline_disruption.reporting.inputs import Inputs, MissingInput, sha256_of


def test_a_table_that_was_read_is_recorded_with_its_hash(report_root):
    inp = Inputs(root=report_root)
    frame = inp.csv("reports/tables/phase7b_final_cancelled_summary.csv")
    assert len(frame) > 0
    path = "reports/tables/phase7b_final_cancelled_summary.csv"
    assert inp.used[path] == sha256_of(report_root / path)
    assert inp.missing == []


def test_a_missing_optional_table_is_listed_and_returns_none(report_root):
    inp = Inputs(root=report_root)
    assert inp.csv("reports/tables/not_there.csv") is None
    assert inp.missing == ["reports/tables/not_there.csv"]


def test_a_quiet_miss_is_not_listed_but_a_required_miss_still_raises(report_root):
    inp = Inputs(root=report_root)
    assert inp.csv("reports/tables/not_there.csv", quiet=True) is None
    assert inp.missing == []
    with pytest.raises(MissingInput):
        inp.csv("reports/tables/not_there.csv", required=True, quiet=True)
    assert inp.missing == ["reports/tables/not_there.csv"]   # a required file is never quiet


def test_the_same_missing_file_is_listed_once(report_root):
    inp = Inputs(root=report_root)
    inp.json("a.json")
    inp.json("a.json")
    assert inp.missing == ["a.json"]


def test_figures_are_linked_only_when_the_file_exists(report_root):
    inp = Inputs(root=report_root)
    assert inp.figure("shap_summary.png") == "../figures/shap_summary.png"
    assert inp.figure("nope.png") is None


def test_evidence_loads_when_lock_marker_and_decision_agree(report_root, lock_hash):
    evidence = load_evidence(Inputs(root=report_root))
    assert evidence.lock_sha == lock_hash and not evidence.smoke
    assert evidence.target("cancelled")["trees"] == 59


def test_evidence_refuses_when_the_hashes_disagree(report_root, write_json_file):
    write_json_file(report_root, "reports/final/final_test_consumed.json", {"lock_sha256": "cd" * 32, "first_run_utc": "x"})
    with pytest.raises(MissingInput, match="do not agree"):
        load_evidence(Inputs(root=report_root))


def test_evidence_refuses_when_the_decision_says_the_test_was_not_used(report_root):
    path = report_root / "reports/tables/phase7b_final_decision.json"
    decision = json.loads(path.read_text())
    decision["final_test_used"] = False
    path.write_text(json.dumps(decision))
    with pytest.raises(MissingInput, match="not used"):
        load_evidence(Inputs(root=report_root))


def test_evidence_refuses_without_the_consumed_marker(report_root):
    (report_root / "reports/final/final_test_consumed.json").unlink()
    with pytest.raises(MissingInput, match="consumed"):
        load_evidence(Inputs(root=report_root))


def test_a_smoke_lock_marks_the_evidence_as_smoke(make_root):
    root = make_root("p", smoke=True)
    evidence = load_evidence(Inputs(root=root, lock_path="reports/final/final_protocol_lock_smoke.json"))
    assert evidence.smoke

"""The protocol lock and the run-once rules (Phase 7B, Step 12)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from airline_disruption.final import lock as lk

ROOT = Path(__file__).resolve().parents[2]


def sample_lock(**changes):
    base = {"lock_version": 1, "created_utc": "2026-10-01T00:00:00+00:00", "smoke": False, "feature_set": "base_no_year", "seed": 42,
            "n_jobs": 8, "params": {"a": 1}, "calibration": {"method": "platt"}, "targets": {"severe_delay_120": {"best_iteration": 53}},
            "preregistered_claims": {}, "files": {}, "final_test_used": False}
    base.update(changes)
    return base


def test_a_sealed_lock_reads_back_and_the_key_order_does_not_matter(tmp_path):
    path = tmp_path / "lock.json"
    sealed = lk.write_lock(path, sample_lock())
    assert lk.read_lock(path) == sealed
    reordered = dict(reversed(list(sample_lock().items())))
    assert lk.lock_hash(reordered) == lk.lock_hash(sample_lock())


def test_editing_any_field_after_sealing_is_detected(tmp_path):
    path = tmp_path / "lock.json"
    lk.write_lock(path, sample_lock())
    text = json.loads(path.read_text())
    text["seed"] = 43
    path.write_text(json.dumps(text))
    with pytest.raises(ValueError, match="edited"):
        lk.read_lock(path)


def test_missing_damaged_and_incomplete_locks_are_refused(tmp_path):
    with pytest.raises(ValueError, match="No protocol lock"):
        lk.read_lock(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        lk.read_lock(tmp_path / "bad.json")
    (tmp_path / "short.json").write_text(json.dumps({"lock_version": 1}))
    with pytest.raises(ValueError, match="missing"):
        lk.read_lock(tmp_path / "short.json")


def test_fingerprints_detect_a_changed_and_a_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    recorded = lk.file_fingerprints(tmp_path, ["a.txt"])
    lk.check_fingerprints(tmp_path, recorded)  # unchanged: fine
    (tmp_path / "a.txt").write_text("two")
    with pytest.raises(ValueError, match="a.txt: changed"):
        lk.check_fingerprints(tmp_path, recorded)
    (tmp_path / "a.txt").unlink()
    with pytest.raises(ValueError, match="a.txt: missing"):
        lk.check_fingerprints(tmp_path, recorded)
    with pytest.raises(ValueError, match="not found"):
        lk.file_fingerprints(tmp_path, ["absent.txt"])


def test_same_size_different_content_is_still_detected(tmp_path):
    (tmp_path / "a.txt").write_text("abc")
    recorded = lk.file_fingerprints(tmp_path, ["a.txt"])
    (tmp_path / "a.txt").write_text("abd")
    with pytest.raises(ValueError, match="changed"):
        lk.check_fingerprints(tmp_path, recorded)


def test_smoke_locks_and_used_locks_cannot_run_the_final_test(monkeypatch):
    monkeypatch.delenv(lk.SMOKE_ENVIRONMENT_VARIABLE, raising=False)
    lk.require_final_lock(sample_lock())
    with pytest.raises(ValueError, match="smoke"):
        lk.require_final_lock(sample_lock(smoke=True))
    with pytest.raises(ValueError, match="already used"):
        lk.require_final_lock(sample_lock(final_test_used=True))
    monkeypatch.setenv(lk.SMOKE_ENVIRONMENT_VARIABLE, "1")
    lk.require_final_lock(sample_lock(smoke=True))  # only the test suite sets this


def test_the_confirmation_phrase_must_be_exact():
    lk.require_confirmation(lk.CONFIRMATION_PHRASE)
    for wrong in (None, "", "yes", lk.CONFIRMATION_PHRASE.lower()):
        with pytest.raises(ValueError, match="--confirm"):
            lk.require_confirmation(wrong)


def test_a_refit_that_differs_from_the_lock_is_refused():
    recorded = {"best_iteration": 53, "validation_pr_auc": 0.0479}
    lk.check_reproduces(recorded, 53, 0.0479 + 5e-7)
    with pytest.raises(ValueError, match="trees"):
        lk.check_reproduces(recorded, 54, 0.0479)
    with pytest.raises(ValueError, match="PR-AUC"):
        lk.check_reproduces(recorded, 53, 0.0481)


def test_stored_and_new_scores_are_compared_row_by_row():
    assert lk.max_score_difference(np.array([0.1, 0.2]), np.array([0.1, 0.2 + 1e-12])) == pytest.approx(1e-12)
    with pytest.raises(ValueError, match="rows"):
        lk.max_score_difference(np.zeros(3), np.zeros(4))


def test_the_final_script_stops_without_a_lock_before_touching_any_data(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_phase7b_final_test.py"), "--root", str(tmp_path), "--confirm", lk.CONFIRMATION_PHRASE],
                            capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode != 0
    assert "No protocol lock" in result.stdout + result.stderr


def test_the_final_script_stops_without_the_confirmation_phrase(tmp_path):
    lk.write_lock(tmp_path / lk.LOCK_PATH, sample_lock())
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_phase7b_final_test.py"), "--root", str(tmp_path)],
                            capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode != 0
    assert "--confirm" in result.stdout + result.stderr


def test_the_seal_covers_every_field_including_nested_ones(tmp_path):
    path = tmp_path / "lock.json"
    lk.write_lock(path, sample_lock())
    text = json.loads(path.read_text())
    text["targets"]["severe_delay_120"]["best_iteration"] = 54  # nested edit
    path.write_text(json.dumps(text))
    with pytest.raises(ValueError, match="edited"):
        lk.read_lock(path)


def test_reading_a_lock_does_not_need_the_data_folder(tmp_path):
    """The lock is plain JSON that can be committed and read on any machine."""
    lk.write_lock(tmp_path / "x.json", sample_lock())
    assert lk.read_lock(tmp_path / "x.json")["feature_set"] == "base_no_year"

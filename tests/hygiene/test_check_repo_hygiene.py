"""Tests for scripts/check_repo_hygiene.py.

The pure rules are tested on paths and sizes. The Git behaviour is tested in a real temporary repository, because
the part that matters (what ``git add -A`` would stage, and what is already in the index) cannot be faked.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_repo_hygiene.py"


@pytest.fixture(scope="module")
def hygiene():
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> None:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def make_repo(root: Path, files: dict[str, str | bytes], gitignore: str = "") -> None:
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    if gitignore:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


# ------------------------------------------------------------------ pure rules
@pytest.mark.parametrize("path", [
    "data/raw/flights_kaggle/flights_sample_3m.csv", "data/processed/phase6/x.csv", "models/lightgbm.txt", "mlruns/0/meta.yaml",
    "logs/run.txt", "data/duckdb/flights.txt",
])
def test_forbidden_prefix(hygiene, path):
    assert any("stay out of Git" in p for p in hygiene.classify(path, 10))


@pytest.mark.parametrize("path", [
    "x.parquet", "a/b.duckdb", "m.keras", "w.ckpt", "m.joblib", "a.pkl", "run.log", "bundle.zip", "pyproject.toml.bak", "a.npy", "m.h5", "m.onnx", "x.pyc",
])
def test_forbidden_suffix(hygiene, path):
    assert hygiene.classify(path, 10), path


@pytest.mark.parametrize("path", [".env", ".env.local", "kaggle.json", "configs/.env", "sub/Kaggle.json", ".DS_Store"])
def test_forbidden_names(hygiene, path):
    assert "an environment or credential file" in hygiene.classify(path, 10)


def test_env_example_is_allowed(hygiene):
    assert hygiene.classify(".env.example", 10) == []


@pytest.mark.parametrize("path", ["src/__pycache__/a.txt", ".vscode/settings.json", "a/.ipynb_checkpoints/n.txt", ".venv/lib/x.txt", ".idea/x.xml"])
def test_forbidden_folders(hygiene, path):
    assert hygiene.classify(path, 10), path


@pytest.mark.parametrize("path", [
    "src/airline_disruption/models/lightgbm_benchmark.py", "src/airline_disruption/final/lock.py", "scripts/run_phase7b_final_test.py",
    "reports/tables/phase7b_final_decision.json", "reports/figures/pr_curve.png", "README.md", "configs/calibration.yaml", "docs/decisions/decision_log.md",
    "tests/reporting/test_reports.py", ".gitignore", "pyproject.toml",
])
def test_normal_project_files_pass(hygiene, path):
    # src/airline_disruption/models/ is code, not the top-level models/ folder.
    assert hygiene.classify(path, 5_000) == []


def test_smoke_outputs_are_flagged_in_reports_but_not_in_tests_or_scripts(hygiene):
    assert hygiene.classify("reports/final/final_protocol_lock_smoke.json", 10)
    assert hygiene.classify("reports/tables/phase7a_smoke_shap_global.csv", 10)
    assert hygiene.classify("configs/something_smoke.yaml", 10)
    assert hygiene.classify("tests/final/test_smoke_lock.py", 10) == []
    assert hygiene.classify("scripts/run_smoke_check.py", 10) == []


def test_exempt_prefixes_allow_small_samples_and_fixtures(hygiene):
    assert hygiene.classify("data/sample/flights_tiny.csv", 1_000) == []
    assert hygiene.classify("tests/data/fixture.parquet", 1_000) == []


def test_exempt_prefixes_do_not_exempt_size(hygiene):
    assert hygiene.classify("data/sample/flights_big.csv", hygiene.CSV_LIMIT + 1)


def test_size_limits(hygiene):
    assert hygiene.classify("reports/tables/a.csv", hygiene.CSV_LIMIT) == []
    assert hygiene.classify("reports/tables/a.csv", hygiene.CSV_LIMIT + 1)
    assert hygiene.classify("reports/figures/a.png", hygiene.FILE_LIMIT) == []
    assert hygiene.classify("reports/figures/a.png", hygiene.FILE_LIMIT + 1)
    assert hygiene.classify("reports/figures/a.png", None) == []


def test_backslash_paths_are_treated_like_slashes(hygiene):
    assert hygiene.classify("data\\raw\\flights.csv", 10)


def test_secret_patterns(hygiene):
    key = "-----BEGIN " + "RSA PRIVATE KEY-----"
    kaggle = '{"username": "u", "key": "' + "0123456789abcdef" * 2 + '"}'
    setting = "KAGGLE_" + "KEY=abcdef0123456789"
    generic = 'api_key = "' + "A" * 24 + '"'
    assert hygiene.scan_text(key) == ["a private key block"]
    assert hygiene.scan_text(kaggle) == ["a Kaggle key"]
    assert hygiene.scan_text(setting) == ["a Kaggle key setting"]
    assert hygiene.scan_text(generic) == ["a hard-coded secret"]


def test_secret_scan_leaves_normal_text_alone(hygiene):
    assert hygiene.scan_text("token = None\npassword = get_password()\nkey = 'short'\nself.api_key_name = 'x'") == []


# ------------------------------------------------------------------ git behaviour
def test_clean_repo_returns_zero(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"README.md": "hello", "src/a.py": "x = 1\n"})
    assert hygiene.main(["--root", str(tmp_path)]) == 0
    assert "No file breaks a project rule" in capsys.readouterr().out


def test_ignored_data_is_not_a_problem(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"README.md": "x", "data/processed/big.parquet": b"0" * 100}, gitignore="data/processed/\n")
    assert hygiene.main(["--root", str(tmp_path)]) == 0


def test_data_that_git_add_would_take_is_reported(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"README.md": "x", "data/processed/big.parquet": b"0" * 100})
    assert hygiene.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "data/processed/big.parquet   [would be added]" in out and "[IN THE INDEX]" not in out


def test_data_already_in_the_index_is_reported_as_such(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"README.md": "x", "reports/tables/phase7a_smoke_shap_global.csv": "a,b\n1,2\n"})
    git(tmp_path, "add", "-A")
    assert hygiene.main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "phase7a_smoke_shap_global.csv   [IN THE INDEX]" in out and "git rm --cached" in out


def test_credentials_in_a_text_file_are_reported(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"configs/a.yaml": "KAGGLE_" + "KEY=abcdef0123456789\n"})
    assert hygiene.main(["--root", str(tmp_path)]) == 1
    assert "Kaggle key setting" in capsys.readouterr().out


def test_secret_scan_can_be_skipped(hygiene, tmp_path):
    make_repo(tmp_path, {"configs/a.yaml": "KAGGLE_" + "KEY=abcdef0123456789\n"})
    assert hygiene.main(["--root", str(tmp_path), "--no-secret-scan"]) == 0


def test_file_names_with_spaces_and_non_ascii_are_handled(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"my notes/café plan.md": "ok", "my notes/run 1.log": "x"})
    assert hygiene.main(["--root", str(tmp_path)]) == 1
    problems_part = capsys.readouterr().out.split("Evidence files")[0]
    assert "my notes/run 1.log" in problems_part
    assert "plan.md" not in problems_part


def test_not_a_git_repository_returns_two(hygiene, tmp_path, capsys):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert hygiene.main(["--root", str(tmp_path)]) == 2


def test_running_from_a_subfolder_is_refused(hygiene, tmp_path, capsys):
    make_repo(tmp_path, {"sub/a.txt": "x"})
    assert hygiene.main(["--root", str(tmp_path / "sub")]) == 2
    assert "repository root" in capsys.readouterr().out


def test_deleted_tracked_file_does_not_crash(hygiene, tmp_path):
    make_repo(tmp_path, {"a.txt": "x", "b.txt": "y"})
    git(tmp_path, "add", "-A")
    (tmp_path / "b.txt").unlink()
    assert hygiene.main(["--root", str(tmp_path)]) == 0


def test_evidence_status_values(hygiene, tmp_path):
    make_repo(tmp_path, {"reports/final/final_protocol_lock.json": "{}", "reports/final/final_test_consumed.json": "{}",
                         "reports/tables/phase7b_final_decision.json": "{}"}, gitignore="reports/final/final_test_consumed.json\n")
    git(tmp_path, "add", "reports/final/final_protocol_lock.json")
    root = tmp_path.resolve()
    index = set(hygiene.index_files(root))
    to_add = set(hygiene.would_add_files(root))
    status = dict(hygiene.evidence_status(root, index, to_add))
    assert status["reports/final/final_protocol_lock.json"] == "in the index"
    assert status["reports/final/final_test_consumed.json"] == "present but ignored by Git"
    assert status["reports/tables/phase7b_final_decision.json"].startswith("not yet added")
    assert status["reports/tables/phase7_reports_manifest.json"] == "not found"


def test_data_external_holds_source_documentation_and_is_allowed_but_still_size_and_suffix_checked(hygiene):
    """The project's .gitignore pattern keeps data/external/ for field mappings, timezone mapping and audit tables."""
    assert hygiene.classify("data/external/airport_timezone_mapping.csv", 50_000) == []
    assert hygiene.classify("data/external/flight_data_audit.json", 10_000) == []
    assert hygiene.classify("data/external/big_extract.csv", hygiene.CSV_LIMIT + 1)
    assert hygiene.classify("data/external/flights.parquet", 100)


def test_data_features_and_the_other_data_folders_are_still_forbidden(hygiene):
    for folder in ("data/raw", "data/processed", "data/interim", "data/features", "data/duckdb"):
        assert hygiene.classify(f"{folder}/x.csv", 100), folder


def test_the_real_project_result_tables_of_a_few_megabytes_pass(hygiene):
    assert hygiene.classify("reports/tables/route_disruption_by_year.csv", 3_900_000) == []
    assert hygiene.classify("reports/tables/entity_disruption_summary.csv", 2_400_000) == []

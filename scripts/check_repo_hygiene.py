"""Check what is about to go into Git (Phase 7B, Step 14a).

Run it from the project root BEFORE ``git add`` and again before ``git commit``:

    python scripts/check_repo_hygiene.py

It looks at two sets of files and judges the union:

* files already in the Git index (tracked or staged), and
* files that ``git add -A`` would stage right now (nothing is staged by the check itself).

It stops with exit code 1 when a file breaks a project rule: raw or processed data, Parquet and DuckDB files,
model and checkpoint files, logs, environment files, credentials, smoke-run outputs, local editor settings, or
anything unusually large. It never changes the repository.

Exit codes: 0 clean, 1 at least one problem, 2 not a Git repository or Git not found.

The rules are lists at the top of this file. Edit the lists, not the logic, if the project rules change.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- rules (edit here)
# A path that starts with one of these is data or output that must stay out of Git.
FORBIDDEN_PREFIXES = (
    "data/raw/", "data/processed/", "data/interim/", "data/features/", "data/duckdb/",
    "models/", "artifacts/", "checkpoints/", "mlruns/", "mlflow/", "logs/",
)
# A folder with one of these names, anywhere in the path.
FORBIDDEN_FOLDERS = (
    "__pycache__", ".ipynb_checkpoints", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode", "mlruns", "node_modules",
)
# File suffixes that hold data, model weights, archives or logs.
FORBIDDEN_SUFFIXES = (
    ".parquet", ".feather", ".duckdb", ".db", ".sqlite", ".sqlite3", ".h5", ".hdf5", ".keras", ".ckpt", ".pt", ".pth",
    ".joblib", ".pkl", ".pickle", ".npy", ".npz", ".pb", ".onnx", ".log", ".zip", ".gz", ".tar", ".7z", ".pyc",
    ".bak", ".orig", ".rej", ".swp", ".tmp",
)
# Exact file names that must never be committed (``.env.example`` is allowed on purpose).
FORBIDDEN_NAMES = ("kaggle.json", ".env", "thumbs.db", ".ds_store")
# Small permitted samples and test fixtures are exempt from the prefix, folder and suffix rules (not from size).
EXEMPT_PREFIXES = ("data/sample/", "tests/")
# Where a file with "smoke" in its name is a test build of a result (scripts and tests may use the word freely).
SMOKE_FOLDERS = ("reports/", "configs/", "data/")
# Files that contain the credential patterns on purpose.
SCAN_SKIP = ("scripts/check_repo_hygiene.py", "tests/hygiene/")

CSV_LIMIT = 5_000_000       # bytes. Aggregated result tables and audit tables can be a few MB; a bigger CSV is probably data.
FILE_LIMIT = 5_000_000      # bytes, any other file
SCAN_LIMIT = 1_000_000      # text files above this size are not scanned for credentials
SCAN_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".cfg", ".ini", ".ps1", ".sh", ".env", ".ipynb")

# Parts are joined at import time so this file does not match its own patterns.
SECRET_PATTERNS = (
    ("a private key block", re.compile("-----BEGIN (?:RSA |EC |OPENSSH |DSA )?" + "PRIVATE KEY-----")),
    ("a Kaggle key", re.compile(r"""["']key["']\s*:\s*["'][0-9a-f]{32}["']""")),
    ("a Kaggle key setting", re.compile(r"KAGGLE_" + r"KEY\s*[=:]\s*\S{8,}")),
    ("a hard-coded secret", re.compile(r"""(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|token)\b\s*[:=]\s*["'][A-Za-z0-9_\-/+=]{16,}["']""")),
)

# Evidence files that should normally be in the repository after the final test. Informational only.
EVIDENCE_PATHS = (
    "reports/final/final_protocol_lock.json",
    "reports/final/final_test_consumed.json",
    "reports/tables/phase7b_final_decision.json",
    "reports/evaluation/calibration_and_alert_policy_report.md",
    "reports/evaluation/explainability_report.md",
    "reports/evaluation/error_analysis_report.md",
    "reports/tables/phase7_reports_manifest.json",
)


# --------------------------------------------------------------------------- pure checks
def classify(path: str, size: int | None) -> list[str]:
    """Return the problems with one repository-relative path (forward slashes). An empty list means it is fine."""
    problems: list[str] = []
    norm = path.replace("\\", "/")
    lower = norm.lower()
    name = lower.rsplit("/", 1)[-1]
    parts = lower.split("/")
    exempt = norm.startswith(EXEMPT_PREFIXES)

    if not exempt:
        for prefix in FORBIDDEN_PREFIXES:
            if lower.startswith(prefix):
                problems.append(f"inside {prefix}, which holds data or outputs that stay out of Git")
                break
        for folder in FORBIDDEN_FOLDERS:
            if folder in parts[:-1]:
                problems.append(f"inside a {folder} folder")
                break
        for suffix in FORBIDDEN_SUFFIXES:
            if name.endswith(suffix):
                problems.append(f"a {suffix} file (data, model, archive, log or backup)")
                break
    if name in FORBIDDEN_NAMES or (name.startswith(".env.") and name != ".env.example"):
        problems.append("an environment or credential file")
    if "smoke" in name and norm.startswith(SMOKE_FOLDERS):
        problems.append("a smoke-run output, which is a test build and not evidence")
    if size is not None:
        limit = CSV_LIMIT if name.endswith(".csv") else FILE_LIMIT
        if size > limit:
            problems.append(f"{size / 1e6:.1f} MB, above the {limit / 1e6:.0f} MB limit for this file type")
    return problems


def scan_text(text: str) -> list[str]:
    """Names of the credential patterns found in a text."""
    return [label for label, pattern in SECRET_PATTERNS if pattern.search(text)]


# --------------------------------------------------------------------------- git helpers
def run_git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=root, capture_output=True, text=True)


def index_files(root: Path) -> list[str]:
    """Files in the Git index (tracked or staged)."""
    done = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip())
    return [p for p in done.stdout.decode("utf-8").split("\0") if p]


def would_add_files(root: Path) -> list[str]:
    """Files that ``git add -A`` would stage, from a dry run. Deleted files are left out."""
    done = run_git(root, "add", "--dry-run", "-A")
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip())
    found = []
    for line in done.stdout.splitlines():
        match = re.match(r"^add '(.*)'$", line.strip())
        if match:
            found.append(match.group(1))
    return found


def find_problems(root: Path, paths: list[str], scan: bool = True) -> dict[str, list[str]]:
    """Map each problem path to its list of problems."""
    found: dict[str, list[str]] = {}
    for rel in sorted(set(paths)):
        file = root / rel
        if not file.is_file():
            continue
        size = file.stat().st_size
        problems = classify(rel, size)
        norm = rel.replace("\\", "/")
        if scan and size <= SCAN_LIMIT and norm.lower().endswith(SCAN_SUFFIXES) and not norm.startswith(SCAN_SKIP):
            try:
                hits = scan_text(file.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                hits = []
            problems += [f"contains {label}" for label in hits]
        if problems:
            found[rel] = problems
    return found


def evidence_status(root: Path, in_index: set[str], to_add: set[str]) -> list[tuple[str, str]]:
    rows = []
    for rel in EVIDENCE_PATHS:
        if not (root / rel).exists():
            status = "not found"
        elif rel in in_index:
            status = "in the index"
        elif rel in to_add:
            status = "not yet added (git add would take it)"
        else:
            status = "present but ignored by Git"
        rows.append((rel, status))
    return rows


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="project root (default: current folder)")
    parser.add_argument("--no-secret-scan", action="store_true", help="skip the credential text scan")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    top = run_git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        print(f"{root} is not inside a Git repository (or Git is not installed).")
        return 2
    if Path(top.stdout.strip()).resolve() != root:
        print(f"Run this from the repository root: {top.stdout.strip()}")
        return 2

    try:
        in_index = index_files(root)
        to_add = would_add_files(root)
    except RuntimeError as error:
        print(f"git failed: {error}")
        return 2

    problems = find_problems(root, in_index + to_add, scan=not args.no_secret_scan)
    print(f"Checked {len(set(in_index) | set(to_add))} files "
          f"({len(in_index)} in the index, {len(set(to_add) - set(in_index))} more that 'git add -A' would stage).")

    if problems:
        print(f"\n{len(problems)} file(s) break a project rule:\n")
        in_index_set = set(in_index)
        for rel, items in problems.items():
            where = "IN THE INDEX" if rel in in_index_set else "would be added"
            print(f"  {rel}   [{where}]")
            for item in items:
                print(f"      - {item}")
        print("\nFor a file that is only 'would be added': put its pattern in .gitignore, or do not add it.")
        print("For a file IN THE INDEX: run  git rm --cached <path>  (keeps the file on disk), then commit that removal.")
    else:
        print("No file breaks a project rule.")

    print("\nEvidence files:")
    for rel, status in evidence_status(root, set(in_index), set(to_add)):
        print(f"  {rel}: {status}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

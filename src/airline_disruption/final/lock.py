"""The protocol lock and the run-once rules for the final test.

Phase 7B, Step 12. The lock is a small JSON file written BEFORE any 2023 row is read. It records the model
settings, the calibration choice, the thresholds learned on 2022, the claims that will be judged, and the
fingerprints of the files the result depends on. The final script refuses to run if the lock was edited,
if one of those files changed, if the lock came from a smoke run, or if the confirmation phrase is missing.

This is a discipline tool, not security. Someone who deletes the stored 2023 scores and edits the lock can
still cheat. The point is to make an accidental second look impossible and a deliberate one visible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

LOCK_PATH = "reports/final/final_protocol_lock.json"
CONSUMED_PATH = "reports/final/final_test_consumed.json"
SCORES_DIR = "data/processed/phase7/final"
CONFIRMATION_PHRASE = "USE-2023-ONCE"
SMOKE_ENVIRONMENT_VARIABLE = "AIRLINE_FINAL_SMOKE_OK"  # only the test suite and fixture runs set this
REQUIRED_KEYS = ("lock_version", "created_utc", "smoke", "feature_set", "seed", "n_jobs", "params", "calibration",
                 "targets", "preregistered_claims", "files", "final_test_used")


def canonical_json(value) -> str:
    """Stable text for hashing: sorted keys, no spaces."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def lock_hash(lock: dict) -> str:
    """SHA-256 of the lock without its own ``lock_sha256`` field."""
    body = {k: v for k, v in lock.items() if k != "lock_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal(lock: dict) -> dict:
    """A copy of ``lock`` with ``lock_sha256`` set."""
    sealed = dict(lock)
    sealed.pop("lock_sha256", None)
    sealed["lock_sha256"] = lock_hash(sealed)
    return sealed


def write_lock(path: Path, lock: dict) -> dict:
    sealed = seal(lock)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    return sealed


def read_lock(path: Path) -> dict:
    """Read the lock and check its seal. Raises ValueError if it is missing, damaged or edited."""
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as problem:
        raise ValueError(f"No protocol lock at {path}. Run scripts/run_phase7b_final_lock.py first.") from problem
    except json.JSONDecodeError as problem:
        raise ValueError(f"{path} is not valid JSON: {problem}") from problem
    missing = [k for k in REQUIRED_KEYS if k not in lock]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if lock.get("lock_sha256") != lock_hash(lock):
        raise ValueError(f"{path} was edited after it was written (its seal does not match). Run the lock script again.")
    return lock


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprints(root: Path, relative_paths) -> dict[str, dict]:
    """``{relative path: {sha256, bytes}}``. Raises ValueError for a missing file."""
    result = {}
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file():
            raise ValueError(f"file not found: {path}")
        result[str(relative)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def check_fingerprints(root: Path, recorded: dict) -> None:
    """Raise ValueError naming every recorded file that is missing or has changed since the lock."""
    problems = []
    for relative, expected in recorded.items():
        path = Path(root) / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
        elif path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            problems.append(f"{relative}: changed since the lock was written")
    if problems:
        raise ValueError("Files behind the lock are not the ones that were locked:\n  " + "\n  ".join(problems))


def require_final_lock(lock: dict) -> None:
    """The lock must come from a real run on the full data and must not already claim a final test."""
    if lock.get("smoke") and os.environ.get(SMOKE_ENVIRONMENT_VARIABLE) != "1":
        raise ValueError("This lock came from a smoke run (reduced fit rows). It cannot be used for the final test.")
    if lock.get("final_test_used"):
        raise ValueError("The lock says the final test was already used.")


def require_confirmation(given: str | None) -> None:
    if given != CONFIRMATION_PHRASE:
        raise ValueError(f"The final test is not run without --confirm {CONFIRMATION_PHRASE}")


def check_reproduces(recorded: dict, best_iteration: int, validation_pr_auc: float, tolerance: float = 1e-6) -> None:
    """The refitted model must be the one that was locked: same tree count, same 2022 PR-AUC."""
    if int(best_iteration) != int(recorded["best_iteration"]):
        raise ValueError(f"The refit chose {best_iteration} trees but the lock recorded {recorded['best_iteration']}. "
                         "Use the same --n-jobs as the lock (it is stored there) and the same package versions.")
    difference = abs(float(validation_pr_auc) - float(recorded["validation_pr_auc"]))
    if difference > tolerance:
        raise ValueError(f"The refit gives 2022 PR-AUC {validation_pr_auc:.9f}, the lock recorded "
                         f"{recorded['validation_pr_auc']:.9f} (difference {difference:.2e}).")


def max_score_difference(stored, new) -> float:
    """Largest absolute difference between two score vectors of the same length."""
    stored, new = np.asarray(stored, dtype="float64"), np.asarray(new, dtype="float64")
    if stored.shape != new.shape:
        raise ValueError(f"The stored 2023 scores have {len(stored):,} rows, this run has {len(new):,}")
    return float(np.max(np.abs(stored - new))) if len(new) else 0.0

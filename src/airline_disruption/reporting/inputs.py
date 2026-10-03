"""Where the report builder reads its tables from, and a record of what it read.

``Inputs`` is the only place that knows file names. A table that a section needs but that does not exist is
recorded as missing, and the report says so in its last section. Nothing is skipped silently. The few files
that every report depends on (the final decision, the lock and the consumed marker) are required.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TARGETS = ("severe_delay_120", "cancelled")
LOCK_PATH = "reports/final/final_protocol_lock.json"
CONSUMED_PATH = "reports/final/final_test_consumed.json"
FINAL_DECISION_PATH = "reports/tables/phase7b_final_decision.json"
CALIBRATION_SELECTED_PATH = "configs/calibration_selected.json"
ARCHITECTURES = ("lstm", "gru", "transformer", "mlp")


class MissingInput(Exception):
    """A required file is not there."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Inputs:
    root: Path
    tag: str = ""                      # file-name tag of the Step 10 and Step 11 tables ("" for the real run)
    lock_path: str = LOCK_PATH
    used: dict[str, str] = field(default_factory=dict)   # relative path -> sha256
    missing: list[str] = field(default_factory=list)

    @property
    def infix(self) -> str:
        return f"_{self.tag}" if self.tag else ""

    # ------------------------------------------------------------------ names
    def path(self, relative: str) -> Path:
        return self.root / relative

    def table(self, name: str) -> str:
        return f"reports/tables/{name}"

    def final_name(self, target: str, kind: str) -> str:
        return self.table(f"phase7b_final_{target}_{kind}.csv")

    def step11_name(self, target: str, kind: str, extension: str = "csv") -> str:
        return self.table(f"phase7b_{target}{self.infix}_{kind}.{extension}")

    def shap_name(self, kind: str) -> str:
        return self.table(f"phase7a{self.infix}_shap_{kind}.csv")

    def error_breakdown_name(self) -> str:
        return self.table("error_breakdown.csv" if not self.tag else f"phase7a{self.infix}_error_breakdown.csv")

    # ------------------------------------------------------------------ reads
    def _note(self, relative: str, quiet: bool = False) -> Path | None:
        path = self.path(relative)
        if not path.exists():
            if not quiet and relative not in self.missing:
                self.missing.append(relative)
            return None
        self.used.setdefault(relative, sha256_of(path))
        return path

    def csv(self, relative: str, required: bool = False, quiet: bool = False) -> pd.DataFrame | None:
        """Read a table. ``quiet`` means the file is optional by design (for example one architecture of several): its absence is not listed."""
        path = self._note(relative, quiet and not required)
        if path is None:
            if required:
                raise MissingInput(f"required table not found: {relative}")
            return None
        return pd.read_csv(path)

    def json(self, relative: str, required: bool = False, quiet: bool = False) -> dict | None:
        path = self._note(relative, quiet and not required)
        if path is None:
            if required:
                raise MissingInput(f"required file not found: {relative}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def figure(self, name: str) -> str | None:
        """Markdown-relative path (from reports/evaluation) of a figure, or None if the file does not exist."""
        if (self.root / "reports/figures" / name).exists():
            return f"../figures/{name}"
        return None

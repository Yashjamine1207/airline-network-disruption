#!/usr/bin/env python
"""Make pytest use importlib import mode, once, so two test files with the same name in different folders do not collide.

Save as:   scripts/ensure_pytest_importlib.py
Run from the project root:

    python scripts/ensure_pytest_importlib.py

Without this setting pytest stops with "import file mismatch" when tests/test_ingestion.py and tests/data/test_ingestion.py
both exist. The script edits pyproject.toml in place (a copy is kept as pyproject.toml.bak) and is safe to run twice.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER = "[tool.pytest.ini_options]"
OPTION = "--import-mode=importlib"


def add_importlib_mode(text: str) -> tuple[str, str]:
    """Return ``(new_text, message)``. ``new_text`` equals ``text`` when nothing needed to change."""
    if OPTION in text:
        return text, "pyproject.toml already sets --import-mode=importlib. Nothing changed."
    lines = text.splitlines()
    if HEADER not in [line.strip() for line in lines]:
        addition = f'\n{HEADER}\naddopts = "{OPTION}"\n'
        return text + ("" if text.endswith("\n") or not text else "\n") + addition, f"Added a new {HEADER} section."
    start = [line.strip() for line in lines].index(HEADER)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    for i in range(start + 1, end):
        match = re.match(r'^(\s*addopts\s*=\s*)"(.*)"\s*$', lines[i])
        if match:
            lines[i] = f'{match.group(1)}"{match.group(2).strip()} {OPTION}"'
            return "\n".join(lines) + "\n", "Added the option to the existing addopts string."
        if re.match(r"^\s*addopts\s*=", lines[i]):
            return text, f'addopts exists in a form this script will not edit. Add "{OPTION}" to it by hand.'
    lines.insert(start + 1, f'addopts = "{OPTION}"')
    return "\n".join(lines) + "\n", "Added addopts to the existing pytest section."


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not path.exists():
        print(f"pyproject.toml not found at {path}")
        return 1
    text = path.read_text(encoding="utf-8")
    new_text, message = add_importlib_mode(text)
    if new_text != text:
        path.with_name("pyproject.toml.bak").write_text(text, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for scripts/inspect_tables.py (read-only schema printer)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_tables.py"


@pytest.fixture(scope="module")
def inspect_tables():
    spec = importlib.util.spec_from_file_location("inspect_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_csv_is_printed_whole(inspect_tables, tmp_path, capsys):
    path = tmp_path / "small.csv"
    pd.DataFrame({"model": ["a", "b"], "mae": [10.5, 11.25]}).to_csv(path, index=False)
    inspect_tables.main([str(path)])
    out = capsys.readouterr().out
    assert "rows: 2" in out and "whole table" in out and "11.25" in out and "model: object" in out


def test_large_csv_prints_count_and_head_only(inspect_tables, tmp_path, capsys):
    path = tmp_path / "big.csv"
    pd.DataFrame({"x": range(100), "y": ["v"] * 100}).to_csv(path, index=False)
    inspect_tables.main([str(path)])
    out = capsys.readouterr().out
    assert "rows: 100" in out and "first 3 rows" in out and "whole table" not in out
    assert "99" not in out.split("first 3 rows")[1]


def test_row_count_streams_in_chunks(inspect_tables, tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame({"x": range(2500)}).to_csv(path, index=False)
    assert inspect_tables.csv_row_count(path, chunk=1000) == 2500


def test_parquet(inspect_tables, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": range(50), "b": [1.5] * 50}).to_parquet(path)
    inspect_tables.main([str(path)])
    out = capsys.readouterr().out
    assert "rows: 50" in out and "a: int64" in out and "b: float64" in out


def test_json_and_missing_and_unknown(inspect_tables, tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps({"k1": 1, "k2": 2}), encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    inspect_tables.main([str(tmp_path / "a.json"), str(tmp_path / "nope.csv"), str(tmp_path / "a.txt")])
    out = capsys.readouterr().out
    assert "['k1', 'k2']" in out and "NOT FOUND" in out and "not a csv, parquet or json file" in out


def test_no_arguments_prints_usage_and_returns_two(inspect_tables, capsys):
    assert inspect_tables.main([]) == 2
    assert "inspect_tables.py" in capsys.readouterr().out

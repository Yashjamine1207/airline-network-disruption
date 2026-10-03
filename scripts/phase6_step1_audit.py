#!/usr/bin/env python
"""
Phase 6, Step 1: read-only preflight audit of the project.

Save this file as:   scripts/phase6_step1_audit.py
Run from VS Code terminal (project root, virtual environment active):

    python scripts/phase6_step1_audit.py

What it collects (all facts Phase 6 depends on):
  1. Environment: Python, packages, TensorFlow devices, CPU, RAM, disk
  2. Repository layout: configs, docs, src modules, tests, scripts
  3. Data files: size, row count, columns, tail-identifier check
  4. Chronological split manifest: counts, date ranges, ordering
  5. Label prevalence per split
  6. Feature-column review: outcome-like and history-like names
  7. Phase 4 result tables exactly as recorded on disk
  8. Text search: final-test use, quarantine, historical outcome rates
  9. Key decision documents and configs (first lines)
 10. Git state and tracked-data check

Safety:
  * It never modifies existing project files.
  * The only file it writes is a text copy of the report, saved to
    reports/generated/phase6_step1_audit.txt (ignored by the project
    .gitignore).
  * No network access. No model training. No final-test evaluation.

Options:
    --root PATH       project root (default: parent of the scripts folder)
    --out PATH        where to save the text report
    --skip-tf         skip the TensorFlow import (saves ~10 seconds)
    --max-rows N      max rows to print per result table (default 40)
    --doc-lines N     max lines to print per document (default 40)
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths taken from the Phase 3 and Phase 4 summaries. If a file lives
# somewhere else on your disk, the audit says so instead of failing.
# ---------------------------------------------------------------------------
DATA_FILES = {
    "raw_flights": "data/raw/flights_kaggle/flights_sample_3m.csv",
    "base_predictors": "data/features/base/base_predictors.csv",
    "model_labels": "data/processed/targets/model_labels.csv",
    "prior_month_counts": "data/features/historical/prior_month_schedule_counts.csv",
    "prior_month_network": "data/features/network/prior_month_airport_network.csv",
    "phase3_features": "data/features/tabular/phase3_schedule_network_features.csv",
    "split_manifest": "data/processed/model_ready/temporal_split_manifest.csv",
}

# Files whose column lists are printed in full (raw file included, so the
# absence of any tail-identifier column is visible).
PRINT_COLUMNS_FOR = ["raw_flights", "base_predictors", "model_labels", "phase3_features"]

# Split sizes reported in the Phase 3 summary.
EXPECTED_SPLIT_COUNTS = {
    "development": 1_848_655,
    "validation": 687_860,
    "final_test_locked": 463_484,
}

PACKAGES = [
    ("pandas", ["pandas"]),
    ("numpy", ["numpy"]),
    ("polars", ["polars"]),
    ("pyarrow", ["pyarrow"]),
    ("scikit-learn", ["scikit-learn"]),
    ("lightgbm", ["lightgbm"]),
    ("xgboost", ["xgboost"]),
    ("tensorflow", ["tensorflow", "tensorflow-cpu", "tensorflow-intel", "tensorflow-macos"]),
    ("keras", ["keras"]),
    ("shap", ["shap"]),
    ("networkx", ["networkx"]),
    ("lifelines", ["lifelines"]),
    ("PyYAML", ["pyyaml", "PyYAML"]),
    ("pytest", ["pytest"]),
    ("psutil", ["psutil"]),
]

EXPECTED_PATHS = [
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "configs/lightgbm.yaml",
    "configs/lstm.yaml",
    "configs/gru.yaml",
    "configs/transformer.yaml",
    "configs/validation.yaml",
    "configs/features.yaml",
    "configs/calibration.yaml",
    "configs/targets.yaml",
    "configs/decision_policy.yaml",
    "configs/network_features.yaml",
    "configs/survival.yaml",
    "docs/decisions/decision_log.md",
    "docs/decisions/historical_outcome_availability.md",
    "docs/decisions/prediction_timestamp_policy.md",
    "docs/decisions/validation_strategy.md",
    "docs/data/leakage_audit.md",
    "docs/data/feature_catalog.md",
    "models/registry/model_registry.md",
    "src/airline_disruption/models/boosting.py",
    "src/airline_disruption/models/calibration.py",
    "src/airline_disruption/models/training.py",
    "src/airline_disruption/models/lstm.py",
    "src/airline_disruption/models/gru.py",
    "src/airline_disruption/models/transformer.py",
    "src/airline_disruption/features/sequences.py",
    "src/airline_disruption/features/point_in_time.py",
    "src/airline_disruption/validation/temporal_splits.py",
    "src/airline_disruption/evaluation/classification.py",
    "src/airline_disruption/evaluation/regression.py",
    "src/airline_disruption/evaluation/operational_value.py",
    "tests/validation",
    "tests/features",
]

# Subpackages whose file names are listed, to see what code already exists.
CODE_DIRS_TO_LIST = [
    "src/airline_disruption/models",
    "src/airline_disruption/features",
    "src/airline_disruption/validation",
    "src/airline_disruption/evaluation",
    "src/airline_disruption/network",
    "src/airline_disruption/explainability",
    "src/airline_disruption/utils",
]

# Phase 4 result tables, plus anything else found in models/metrics.
PHASE4_TABLES = [
    "models/metrics/classification_model_comparison.csv",
    "models/metrics/regression_model_comparison.csv",
    "models/metrics/calibration_metrics.csv",
    "reports/tables/classification_model_comparison.csv",
    "reports/tables/regression_model_comparison.csv",
    "reports/tables/calibration_summary.csv",
    "reports/tables/error_breakdown.csv",
]

# Documents printed at the end (head N lines, or tail for the decision log).
KEY_DOCS = [
    ("configs/lightgbm.yaml", "head"),
    ("configs/validation.yaml", "head"),
    ("configs/calibration.yaml", "head"),
    ("docs/decisions/historical_outcome_availability.md", "head"),
    ("docs/decisions/decision_log.md", "tail"),
    ("models/registry/model_registry.md", "head"),
]

# Column-name heuristics. They only raise flags for manual review.
OUTCOME_LIKE = re.compile(
    r"(arr|dep|arrival|departure)[_ ]?delay|actual|cancel|divert|delay[_ ]?(cause|due)"
    r"|carrier_delay|weather_delay|nas_delay|security_delay|late_aircraft|taxi|wheels"
    r"|air_time|severe|target|label|outcome|exclusion",
    re.I,
)
SCHEDULE_SAFE = re.compile(r"crs|sched", re.I)
HISTORY_LIKE = re.compile(
    r"rate|rolling|roll_|hist|lag|prior|past|prev|trailing|ewm|pressure|_(7|30|90)d|last_",
    re.I,
)
SPLIT_LIKE = re.compile(r"split|period|regime|dataset|eval|fold|partition", re.I)
FINAL_TEST_VALUE = re.compile(r"test|final|2023", re.I)
TAIL_LIKE = re.compile(r"tail|aircraft|n_number|registration", re.I)
PRIORITY_COLS = re.compile(
    r"model|target|split|period|regime|feature|cohort|pr_?auc|roc|mae|rmse|brier"
    r"|precision|recall|f1|n_|rows|events|prevalence|version",
    re.I,
)

# Text-search patterns for the contradictions that matter for Phase 6.
SEARCH_PATTERNS = {
    "quarantine": r"quarantin",
    "final-test references": r"final[_ ]?test|test_final|\bx_test\b|evaluate_on_test",
    "outcome-availability policy": r"outcome[_ ]availab|historical_outcome",
    "historical outcome rates used as features": (
        r"(hist\w*|rolling\w*|prior\w*|past\w*)[_ ]?(severe|cancel\w*|delay)\w*[_ ]?"
        r"(rate|mean|median|count)s?"
    ),
    "feature-list definitions": r"feature_(cols|columns|list|names|set)\b|FEATURE_(COLS|COLUMNS|LIST)",
}
SCAN_DIRS = ["src", "scripts", "configs", "docs", "notebooks", "models/registry", "reports/evaluation", "tests"]
SCAN_EXT = {".py", ".yaml", ".yml", ".md", ".ipynb", ".txt", ".toml"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints",
    "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache",
}

PROTECTED_DIRS = [
    "data/raw", "data/interim", "data/processed", "data/features",
    "models/artifacts", "models/checkpoints",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class Report:
    """Prints every line and keeps a copy so it can be saved to a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed_sections: list[str] = []

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.add("")
        self.add("=" * 78)
        self.add(title)
        self.add("=" * 78)


def count_lines(path: Path, block: int = 1 << 24) -> int:
    """Count newline-terminated lines by streaming bytes (no full load)."""
    total = 0
    last = b""
    with path.open("rb") as handle:
        while True:
            buf = handle.read(block)
            if not buf:
                break
            total += buf.count(b"\n")
            last = buf[-1:]
    if last and last != b"\n":
        total += 1  # final line without a trailing newline
    return total


def table_info(path: Path) -> tuple[list[str], int | None]:
    """Return (column names, row count) without loading the whole table."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return list(parquet_file.schema_arrow.names), parquet_file.metadata.num_rows
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle), [])
    return header, max(count_lines(path) - 1, 0)


def load_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def find_col(columns, *patterns: str) -> str | None:
    """First column matching the first pattern that matches anything."""
    for pattern in patterns:
        for column in columns:
            if re.search(pattern, str(column), flags=re.I):
                return column
    return None


def mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:,.1f} MB"


def mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def is_binary(series: pd.Series) -> bool:
    """True for 0/1 (or boolean) columns."""
    if not (pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)):
        return False
    values = pd.unique(series.dropna())
    if len(values) > 2:
        return False
    return set(pd.Series(values).astype(float)) <= {0.0, 1.0}


def display_frame(df: pd.DataFrame, limit: int = 16) -> tuple[pd.DataFrame, list[str]]:
    """Pick the most informative columns so wide tables stay readable."""
    columns = list(df.columns)
    keep = [c for c in columns if PRIORITY_COLS.search(str(c))][:limit]
    for column in columns:
        if len(keep) >= limit:
            break
        if column not in keep:
            keep.append(column)
    keep = [c for c in columns if c in keep]  # restore original order
    hidden = [c for c in columns if c not in keep]
    return df[keep], hidden


def memory_gb() -> tuple[float, float] | None:
    """(total, available) RAM in GB, using only the standard library if needed."""
    try:
        import psutil

        info = psutil.virtual_memory()
        return info.total / 1e9, info.available / 1e9
    except ImportError:
        pass
    if sys.platform.startswith("win"):
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return status.ullTotalPhys / 1e9, status.ullAvailPhys / 1e9
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = float(rest.split()[0]) * 1024  # kB to bytes
        if "MemTotal" in values and "MemAvailable" in values:
            return values["MemTotal"] / 1e9, values["MemAvailable"] / 1e9
    return None


# ---------------------------------------------------------------------------
# Section 1: environment
# ---------------------------------------------------------------------------
def section_environment(rep: Report, root: Path, args, ctx: dict) -> None:
    rep.add(f"Report time (local): {datetime.now():%Y-%m-%d %H:%M:%S}")
    rep.add(f"Project root: {root}")
    rep.add(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    rep.add(f"Platform: {platform.platform()}")
    rep.add(f"CPU logical cores: {os.cpu_count()}")

    ram = memory_gb()
    if ram:
        rep.add(f"RAM: {ram[0]:.1f} GB total, {ram[1]:.1f} GB available right now")
    else:
        rep.add("RAM: could not be determined")

    disk = shutil.disk_usage(root)
    rep.add(f"Disk at project root: {disk.free / 1e9:,.0f} GB free of {disk.total / 1e9:,.0f} GB")

    rep.add("")
    rep.add("Installed packages:")
    for label, dist_names in PACKAGES:
        version = None
        for dist_name in dist_names:
            try:
                version = f"{md.version(dist_name)} (dist: {dist_name})"
                break
            except md.PackageNotFoundError:
                continue
        rep.add(f"  {label:<14} {version or 'NOT INSTALLED'}")

    rep.add("")
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        rep.add(f"nvidia-smi GPUs: {result.stdout.strip() or result.stderr.strip()}")
    else:
        rep.add("nvidia-smi: not found (no NVIDIA driver on PATH)")

    if args.skip_tf:
        rep.add("TensorFlow device check: skipped (--skip-tf)")
        return
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF start-up logs
        import tensorflow as tf

        devices = [f"{d.name} ({d.device_type})" for d in tf.config.list_physical_devices()]
        rep.add(f"TensorFlow {tf.__version__} imported OK. Physical devices: {devices}")
    except ImportError as exc:
        rep.add(f"TensorFlow cannot be imported in this interpreter: {exc}")
    except Exception as exc:  # noqa: BLE001 - report any start-up failure
        rep.add(f"TensorFlow import failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Section 2: repository layout
# ---------------------------------------------------------------------------
def section_layout(rep: Report, root: Path, args, ctx: dict) -> None:
    missing = []
    for rel in EXPECTED_PATHS:
        exists = (root / rel).exists()
        if not exists:
            missing.append(rel)
        rep.add(f"  [{'OK' if exists else 'MISSING'}] {rel}")
    rep.add(f"Missing: {len(missing)} of {len(EXPECTED_PATHS)} expected paths")

    rep.add("")
    for rel in CODE_DIRS_TO_LIST:
        folder = root / rel
        if folder.exists():
            names = sorted(p.name for p in folder.glob("*.py") if p.name != "__init__.py")
            rep.add(f"{rel}/: {', '.join(names) if names else '(no modules)'}")
        else:
            rep.add(f"{rel}/: folder does not exist")

    scripts = root / "scripts"
    if scripts.exists():
        names = sorted(p.name for p in scripts.glob("*.py"))
        rep.add(f"scripts/ ({len(names)} files): {', '.join(names)}")

    tests = root / "tests"
    if tests.exists():
        test_files = sorted(tests.rglob("test_*.py"))
        rep.add(f"tests/: {len(test_files)} test files")

    notebooks = root / "notebooks"
    if notebooks.exists():
        rep.add(f"notebooks/: {', '.join(sorted(p.name for p in notebooks.glob('*.ipynb')))}")


# ---------------------------------------------------------------------------
# Section 3: data files
# ---------------------------------------------------------------------------
def section_data_files(rep: Report, root: Path, args, ctx: dict) -> None:
    for key, rel in DATA_FILES.items():
        path = root / rel
        if not path.exists():
            rep.add(f"[MISSING] {key}: {rel}")
            continue
        columns, rows = table_info(path)
        rep.add(f"[OK] {key}: {rel}")
        rep.add(f"     size {mb(path)}, rows {rows:,}, columns {len(columns)}, modified {mtime(path)}")
        ctx.setdefault("columns", {})[key] = columns
        if key in PRINT_COLUMNS_FOR:
            rep.add(f"     columns: {columns}")
        if key == "raw_flights":
            tail_cols = [c for c in columns if TAIL_LIKE.search(c)]
            rep.add(f"     tail/aircraft-like columns in raw file: {tail_cols or 'none'}")

    # Anything else in the feature/processed folders (e.g. the Phase 4 full-split table).
    rep.add("")
    rep.add("Other tables under data/features and data/processed (header only):")
    known = {(root / rel).resolve() for rel in DATA_FILES.values()}
    found = 0
    for folder in ("data/features", "data/processed"):
        base = root / folder
        if not base.exists():
            continue
        for path in sorted(list(base.rglob("*.csv")) + list(base.rglob("*.parquet"))):
            if path.resolve() in known:
                continue
            found += 1
            if found > 60:
                break
            try:
                columns, rows = table_info(path) if path.stat().st_size < 3e9 else ([], None)
                rows_text = f"{rows:,}" if rows is not None else "n/a"
                history = [c for c in columns if HISTORY_LIKE.search(c)]
                rep.add(f"  {path.relative_to(root)}  ({mb(path)}, rows {rows_text}, cols {len(columns)})")
                if history:
                    rep.add(f"      history-like columns: {history[:12]}")
            except Exception as exc:  # noqa: BLE001
                rep.add(f"  {path.relative_to(root)}: could not read header ({type(exc).__name__})")
    if found == 0:
        rep.add("  (none found)")


# ---------------------------------------------------------------------------
# Section 4: split manifest
# ---------------------------------------------------------------------------
def section_manifest(rep: Report, root: Path, args, ctx: dict) -> None:
    path = root / DATA_FILES["split_manifest"]
    if not path.exists():
        folder = root / "data/processed/model_ready"
        candidates = sorted(folder.glob("*manifest*")) if folder.exists() else []
        if not candidates:
            rep.add("Split manifest not found. Cannot check chronology.")
            return
        path = candidates[0]
        rep.add(f"Using {path.relative_to(root)} instead of the expected path.")

    df = load_table(path)
    columns = list(df.columns)
    id_col = find_col(columns, r"^source_row_number$", r"source_row")
    split_col = find_col(columns, r"^split", r"split")
    ts_col = find_col(columns, r"prediction.*utc", r"utc.*prediction", r"prediction")
    rep.add(f"Columns: {columns}")
    rep.add(f"Detected: id={id_col}, split={split_col}, prediction timestamp={ts_col}")
    ctx.update(manifest=df, id_col=id_col, split_col=split_col, ts_col=ts_col)

    rep.add(f"Rows: {len(df):,}")
    if id_col:
        rep.add(f"Duplicate {id_col} values: {int(df[id_col].duplicated().sum()):,}")
    if not split_col:
        rep.add("No split column detected. Cannot continue this section.")
        return

    counts = df[split_col].value_counts(dropna=False)
    rep.add("")
    rep.add("Rows per split (expected value from the Phase 3 summary in brackets):")
    for label, value in counts.items():
        expected = EXPECTED_SPLIT_COUNTS.get(str(label))
        note = "" if expected is None else f"  [expected {expected:,}, diff {value - expected:+,}]"
        rep.add(f"  {label}: {value:,}{note}")

    if not ts_col:
        rep.add("No prediction-timestamp column detected. Cannot check date ranges.")
        return
    stamps = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    rep.add("")
    rep.add(f"Unparseable {ts_col} values: {int(stamps.isna().sum()):,}")
    grouped = stamps.groupby(df[split_col]).agg(["min", "max"])
    rep.add("Prediction-timestamp range per split (UTC):")
    rep.add(grouped.to_string())

    # Chronology: each split (ordered by its first timestamp) must start after the previous one ends.
    ordered = grouped.sort_values("min")
    passed = True
    for (prev_name, prev_row), (name, row) in zip(ordered.iterrows(), list(ordered.iterrows())[1:]):
        ok = prev_row["max"] < row["min"]
        passed &= bool(ok)
        rep.add(f"  {prev_name} ends {prev_row['max']:%Y-%m-%d %H:%M}, {name} starts {row['min']:%Y-%m-%d %H:%M}: {'OK' if ok else 'OVERLAP'}")
    rep.add(f"Chronological ordering of splits: {'PASS' if passed else 'FAIL'}")

    rep.add("")
    rep.add("Rows per split by prediction-timestamp year (UTC):")
    rep.add(pd.crosstab(df[split_col], stamps.dt.year).to_string())


# ---------------------------------------------------------------------------
# Section 5: label prevalence per split
# ---------------------------------------------------------------------------
def section_labels(rep: Report, root: Path, args, ctx: dict) -> None:
    path = root / DATA_FILES["model_labels"]
    if not path.exists() or "manifest" not in ctx or not ctx.get("split_col") or not ctx.get("id_col"):
        rep.add("Labels file or usable manifest missing. Skipped.")
        return
    id_col, split_col = ctx["id_col"], ctx["split_col"]
    labels = load_table(path)
    if id_col not in labels.columns:
        rep.add(f"{id_col} not in labels file. Columns: {list(labels.columns)}")
        return
    merged = labels.merge(ctx["manifest"][[id_col, split_col]], on=id_col, how="left")
    rep.add(f"Label rows: {len(labels):,}. Rows without a manifest split: {int(merged[split_col].isna().sum()):,}")

    binary_cols = [c for c in labels.columns if c != id_col and is_binary(labels[c])]
    rep.add(f"Binary label-like columns: {binary_cols}")
    for column in binary_cols[:12]:
        values = merged[column].astype("float64")
        table = values.groupby(merged[split_col]).agg(non_null="count", events="sum", prevalence="mean")
        rep.add("")
        rep.add(f"{column} by split:")
        rep.add(table.round(5).to_string())

    continuous = [
        c for c in labels.columns
        if c != id_col and pd.api.types.is_numeric_dtype(labels[c]) and not is_binary(labels[c])
        and re.search(r"delay|regress|target", c, re.I)
    ]
    for column in continuous[:4]:
        table = merged[column].groupby(merged[split_col]).agg(["count", "mean", "median", "max"])
        rep.add("")
        rep.add(f"{column} (continuous) by split:")
        rep.add(table.round(3).to_string())


# ---------------------------------------------------------------------------
# Section 6: feature-column review
# ---------------------------------------------------------------------------
def section_feature_columns(rep: Report, root: Path, args, ctx: dict) -> None:
    rep.add("Name-based review only. A clean result here is not proof of no leakage;")
    rep.add("it only catches obvious outcome-like or history-like column names.")
    for key in ("base_predictors", "phase3_features"):
        columns = ctx.get("columns", {}).get(key)
        if columns is None:
            rep.add(f"{key}: file not found, skipped")
            continue
        outcome = [c for c in columns if OUTCOME_LIKE.search(c) and not SCHEDULE_SAFE.search(c)]
        history = [c for c in columns if HISTORY_LIKE.search(c)]
        rep.add("")
        rep.add(f"{key} ({len(columns)} columns)")
        rep.add(f"  outcome-like names (REVIEW if any): {outcome or 'none'}")
        rep.add(f"  history-like names: {history or 'none'}")


# ---------------------------------------------------------------------------
# Section 7: Phase 4 result tables
# ---------------------------------------------------------------------------
def section_phase4_tables(rep: Report, root: Path, args, ctx: dict) -> None:
    paths = [root / rel for rel in PHASE4_TABLES]
    metrics_dir = root / "models/metrics"
    if metrics_dir.exists():
        paths += sorted(metrics_dir.glob("*.csv"))
    seen: set[Path] = set()
    any_found = False
    for path in paths:
        if path.resolve() in seen or not path.exists():
            continue
        seen.add(path.resolve())
        any_found = True
        df = pd.read_csv(path, low_memory=False)
        rep.add("")
        rep.add(f"--- {path.relative_to(root)}  (modified {mtime(path)}, {len(df)} rows x {df.shape[1]} columns)")
        rep.add(f"All columns: {list(df.columns)}")

        # Rows that mention the final/test period. Raw signal only.
        split_cols = [c for c in df.columns if SPLIT_LIKE.search(str(c))]
        if split_cols:
            mask = pd.Series(False, index=df.index)
            for column in split_cols:
                rep.add(f"  {column} values: {sorted(map(str, df[column].dropna().unique()))[:15]}")
                mask |= df[column].astype(str).str.contains(FINAL_TEST_VALUE)
            rep.add(f"  Rows whose split/period value mentions test/final/2023: {int(mask.sum())} of {len(df)}")
            rep.add("  (This shows results were recorded for that period. It does not say whether the run was valid.)")

        shown, hidden = display_frame(df)
        numeric = shown.select_dtypes("number").columns
        shown = shown.copy()
        shown[numeric] = shown[numeric].round(4)
        rep.add(shown.head(args.max_rows).to_string(max_colwidth=30))
        if len(df) > args.max_rows:
            rep.add(f"  ... {len(df) - args.max_rows} more rows not shown")
        if hidden:
            rep.add(f"  Columns not shown: {hidden}")
    if not any_found:
        rep.add("No Phase 4 result tables found in the expected locations.")


# ---------------------------------------------------------------------------
# Section 8: text search
# ---------------------------------------------------------------------------
def iter_text_files(root: Path, skip_paths: set[Path]):
    for rel in SCAN_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in SCAN_EXT and path.resolve() not in skip_paths:
                    yield path


def section_text_search(rep: Report, root: Path, args, ctx: dict) -> None:
    compiled = {label: re.compile(pattern, re.I) for label, pattern in SEARCH_PATTERNS.items()}
    hits: dict[str, list[tuple[str, int, str]]] = {label: [] for label in compiled}
    skip_paths = {Path(__file__).resolve()}
    scanned = skipped = 0
    for path in iter_text_files(root, skip_paths):
        if path.stat().st_size > 8_000_000:
            skipped += 1
            continue
        scanned += 1
        rel = str(path.relative_to(root))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                line = line[:3000]  # notebook JSON can contain very long lines
                for label, pattern in compiled.items():
                    if pattern.search(line):
                        hits[label].append((rel, number, line.strip()[:130]))
    rep.add(f"Scanned {scanned} text files ({skipped} skipped as too large).")
    for label, items in hits.items():
        rep.add("")
        rep.add(f"[{label}] {len(items)} hits in {len({i[0] for i in items})} files")
        per_file: dict[str, int] = {}
        for rel, _, _ in items:
            per_file[rel] = per_file.get(rel, 0) + 1
        for rel, count in sorted(per_file.items(), key=lambda kv: -kv[1])[:10]:
            rep.add(f"    {count:>4}  {rel}")
        for rel, number, snippet in items[:6]:
            rep.add(f"      {rel}:{number}: {snippet}")


# ---------------------------------------------------------------------------
# Section 9: key documents
# ---------------------------------------------------------------------------
def section_key_docs(rep: Report, root: Path, args, ctx: dict) -> None:
    for rel, mode in KEY_DOCS:
        path = root / rel
        rep.add("")
        if not path.exists():
            rep.add(f"--- {rel}: MISSING")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        chosen = lines[: args.doc_lines] if mode == "head" else lines[-args.doc_lines:]
        rep.add(f"--- {rel} (modified {mtime(path)}, {len(lines)} lines, showing {mode} {len(chosen)})")
        for line in chosen:
            rep.add(f"    {line[:160]}")


# ---------------------------------------------------------------------------
# Section 10: git
# ---------------------------------------------------------------------------
def run_git(root: Path, *git_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *git_args], cwd=root, capture_output=True, text=True, timeout=120)


def section_git(rep: Report, root: Path, args, ctx: dict) -> None:
    if shutil.which("git") is None:
        rep.add("git not found on PATH.")
        return
    if run_git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        rep.add("Project root is not a git repository.")
        return
    rep.add(f"Branch: {run_git(root, 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()}")
    rep.add("Last 8 commits:")
    for line in run_git(root, "log", "--oneline", "-8").stdout.splitlines():
        rep.add(f"    {line}")
    status = run_git(root, "status", "--short").stdout.splitlines()
    rep.add(f"Uncommitted or untracked entries: {len(status)}")
    for line in status[:40]:
        rep.add(f"    {line}")
    if len(status) > 40:
        rep.add(f"    ... {len(status) - 40} more")
    rep.add("")
    rep.add("Tracked files inside protected data/model folders (should be none):")
    total = 0
    for rel in PROTECTED_DIRS:
        tracked = [
            f for f in run_git(root, "ls-files", rel).stdout.splitlines()
            if not f.endswith((".gitkeep", ".md"))
        ]
        total += len(tracked)
        if tracked:
            rep.add(f"    {rel}: {len(tracked)} tracked, e.g. {tracked[:3]}")
    if total == 0:
        rep.add("    none")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 Step 1 read-only project audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-tf", action="store_true")
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--doc-lines", type=int, default=40)
    args = parser.parse_args()

    # Keep the Windows console from choking on any unexpected character.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    root = args.root.resolve()
    out_path = args.out or (root / "reports" / "generated" / "phase6_step1_audit.txt")
    rep = Report()
    ctx: dict = {}

    sections = [
        ("1. ENVIRONMENT", section_environment),
        ("2. REPOSITORY LAYOUT", section_layout),
        ("3. DATA FILES", section_data_files),
        ("4. CHRONOLOGICAL SPLIT MANIFEST", section_manifest),
        ("5. LABEL PREVALENCE PER SPLIT", section_labels),
        ("6. FEATURE-COLUMN REVIEW", section_feature_columns),
        ("7. PHASE 4 RESULT TABLES ON DISK", section_phase4_tables),
        ("8. TEXT SEARCH (final test, quarantine, historical rates)", section_text_search),
        ("9. KEY DOCUMENTS AND CONFIGS", section_key_docs),
        ("10. GIT STATE", section_git),
    ]
    for title, func in sections:
        rep.section(title)
        try:
            func(rep, root, args, ctx)
        except Exception as exc:  # noqa: BLE001 - one broken section must not stop the rest
            rep.failed_sections.append(title)
            rep.add(f"[SECTION FAILED] {type(exc).__name__}: {exc}")
            rep.add(traceback.format_exc(limit=3))

    rep.section("DONE")
    if rep.failed_sections:
        rep.add(f"Sections that failed: {rep.failed_sections}")
    else:
        rep.add("All sections ran.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rep.lines), encoding="utf-8")
    rep.add(f"Report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
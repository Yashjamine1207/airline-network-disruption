#!/usr/bin/env python
"""Phase 6, Step 7a: build the model inputs for the neural networks.

Save as:   scripts/build_phase6_deep_inputs.py
Run from the project root, after the Step 2, 3 and 5 build scripts:

    python scripts/build_phase6_deep_inputs.py

What it does
------------
1. Loads the ``base_no_year`` schedule features and the severe-delay label for the fit,
   early-stopping and 2022 validation rows (the same loader and roles as the LightGBM
   benchmark, so every model sees the same rows). The final-test rows are never loaded.
2. Learns category vocabularies, medians and scaling from the FIT rows only.
3. Turns every row into integer category codes and ten numeric inputs.
4. Attaches each row's airport-table row and newest closed bin (from Step 5) and checks
   them three ways (own origin airport, recomputed end bin, no bin after the prediction time).
5. Fits the sequence scaler on a sample of FIT rows.
6. Reports how many rows in each period have a category the model has no embedding for.

Writes (large files stay out of Git; the two small files can be committed):
    data/processed/phase6/deep_inputs_v1/*.npy, spec.json, sequence_scaler.json, manifest.json
    reports/tables/phase6_deep_inputs_unseen_levels.csv
    reports/tables/phase6_deep_inputs_v1_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.deep import build as deep_build
    from airline_disruption.deep import inputs as di
    from airline_disruption.features.feature_sets import FEATURE_VERSION
    from airline_disruption.models import lightgbm_benchmark as lb
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.deep import build as deep_build  # noqa: E402
    from airline_disruption.deep import inputs as di  # noqa: E402
    from airline_disruption.features.feature_sets import FEATURE_VERSION  # noqa: E402
    from airline_disruption.models import lightgbm_benchmark as lb  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

TABLE_PATH = "data/processed/phase6/airport_bins_v1.npz"
INDEX_PATH = "data/processed/phase6/phase6_sequence_index_v1.parquet"
OUTPUT_DIR = "data/processed/phase6/deep_inputs_v1"
FEATURE_SET = "base_no_year"  # DEC-018


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 6 neural-network inputs")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scaler-sample", type=int, default=200_000, help="fit rows used to fit the sequence scaler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=OUTPUT_DIR, help="folder under the project root")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    for relative in (lb.COHORT_PATH, lb.PREDICTORS_PATH, TABLE_PATH, INDEX_PATH):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2, 3 and 5 build scripts first)")

    # ------------------------------------------------------------------
    # Rows and roles: identical to the LightGBM benchmark
    # ------------------------------------------------------------------
    print(f"Loading the {FEATURE_SET} rows (development and 2022 validation; the final test is not loaded) ...")
    try:
        frame = lb.load_model_frame(root, FEATURE_SET)
        roles = lb.assign_roles(frame.meta, lb.PROTOCOL_PHASE6)
        lb.check_role_chronology(frame.meta, roles)
    except ValueError as problem:
        fail(str(problem))
    counts = roles.value_counts()
    print("  rows by role: " + ", ".join(f"{role} {int(counts.get(role, 0)):,}" for role in ("fit", "early_stop", "validation")))

    table, airports, grid = ab.load_bin_table(root / TABLE_PATH)
    index = pd.read_parquet(root / INDEX_PATH, columns=[lb.ID_COLUMN, "origin_index", "end_bin"])
    print(f"  airport table {table.shape}, sequence index {len(index):,} rows")

    # ------------------------------------------------------------------
    # Build and check
    # ------------------------------------------------------------------
    print("Learning vocabularies and scalers from the fit rows, transforming every row, checking the sequence index ...")
    try:
        inputs, unseen = deep_build.build_deep_inputs(
            frame.X, frame.y, frame.meta, roles, index, table, airports, grid, scaler_sample=args.scaler_sample, seed=args.seed
        )
    except ValueError as problem:
        fail(str(problem))
    print("  index checks passed: own origin airport, recomputed end bin, no bin closes after the prediction time")

    sizes = inputs.spec.vocabulary_sizes()
    print("  embedding rows per column (known levels + 1 unknown): " + ", ".join(f"{c} {n}" for c, n in sizes.items()))
    print("\nShare of rows whose level has no embedding (mapped to 'unknown'):")
    print(unseen.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    per_role = {}
    for role in di.ROLE_CODES:
        rows = inputs.rows(role)
        stamps = pd.to_datetime(inputs.prediction_seconds[rows], unit="s", utc=True)
        per_role[role] = {
            "rows": int(len(rows)),
            "event_rate": float(inputs.label[rows].mean()),
            "first_prediction_utc": stamps.min().isoformat(),
            "last_prediction_utc": stamps.max().isoformat(),
        }
    inputs.manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_version": di.INPUT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_set": FEATURE_SET,
        "sequence_version": ab.SEQUENCE_VERSION,
        "sequence_entity": "origin airport",
        "lookback_bins": ab.LOOKBACK_BINS,
        "bin_hours": ab.BIN_HOURS,
        "protocol": lb.PROTOCOL_PHASE6,
        "target": lb.TARGET,
        "vocabulary_sizes": sizes,
        "min_level_count": di.MIN_LEVEL_COUNT,
        "numeric_inputs": list(di.NUMERIC_NAMES),
        "sequence_scaler_sample_rows": min(args.scaler_sample, per_role["fit"]["rows"]),
        "seed": args.seed,
        "rows_by_role": per_role,
        "final_test_used": False,
        "build_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    out = root / args.output
    di.save_deep_inputs(out, inputs)
    tables = root / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    unseen.to_csv(tables / "phase6_deep_inputs_unseen_levels.csv", index=False)
    (tables / "phase6_deep_inputs_v1_manifest.json").write_text(json.dumps(inputs.manifest, indent=2, default=str), encoding="utf-8")

    megabytes = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e6
    print(f"\nWrote {out} ({megabytes:,.0f} MB) and two files in {tables}")
    print(f"Done in {time.perf_counter() - started:,.0f} s. The final-test rows were not loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 6, Step 3: build the locked predictor table for every Phase 6 model.

Save as:   scripts/build_phase6_predictors.py
Run from the project root, after scripts/build_phase6_cohort.py:

    python scripts/build_phase6_predictors.py --dry-run   # check everything, write nothing
    python scripts/build_phase6_predictors.py             # build and write

What it does
------------
1. Reads data/features/tabular/phase3_schedule_network_features.csv (read only) and
   keeps the columns of the approved feature sets plus the row id, the scheduled
   departure in UTC, and the two columns needed for alignment checks.
2. Refuses to continue if any selected column looks like an outcome.
3. Checks that the rows, UTC prediction timestamps and flight dates match the
   Phase 6 cohort table exactly.
4. Reports missing values (development and validation only), categories in
   validation that development never saw, and an independent recomputation of the
   seven prior-month features.
5. Writes:
     data/processed/phase6/phase6_predictors_v1.parquet      (ignored by Git)
     reports/tables/phase6_predictor_quality.csv             (small, safe to commit)
     reports/tables/phase6_prior_month_verification.csv      (small, safe to commit)
     reports/tables/phase6_predictors_v1_manifest.json       (small, safe to commit)

The table holds predictors only. Labels and splits live in the cohort table. A
model joins the two on source_row_number.

Nothing here reads a label. The final-test split is not used in any statistic
that could shape preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.features.feature_sets import (
        ALL_FEATURE_COLUMNS,
        BASE_CATEGORICAL_COLUMNS,
        BASE_NUMERIC_COLUMNS,
        FEATURE_SETS,
        FEATURE_VERSION,
        ID_COLUMN,
        TIMESTAMP_COLUMN,
        columns_for,
        validate_feature_columns,
    )
    from airline_disruption.features.predictor_checks import (
        check_alignment,
        null_rate_table,
        unseen_category_table,
        verify_prior_month_features,
    )
    from airline_disruption.features.predictor_table import read_predictor_csv
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.features.feature_sets import (  # noqa: E402
        ALL_FEATURE_COLUMNS,
        BASE_CATEGORICAL_COLUMNS,
        BASE_NUMERIC_COLUMNS,
        FEATURE_SETS,
        FEATURE_VERSION,
        ID_COLUMN,
        TIMESTAMP_COLUMN,
        columns_for,
        validate_feature_columns,
    )
    from airline_disruption.features.predictor_checks import (  # noqa: E402
        check_alignment,
        null_rate_table,
        unseen_category_table,
        verify_prior_month_features,
    )
    from airline_disruption.features.predictor_table import read_predictor_csv  # noqa: E402

SOURCE_CSV = "data/features/tabular/phase3_schedule_network_features.csv"
COHORT_PARQUET = "data/processed/phase6/phase6_cohort_v1.parquet"
OUT_PARQUET = "data/processed/phase6/phase6_predictors_v1.parquet"

# Columns read besides the features: alignment checks need the first two, ordering needs the third.
CHECK_COLUMNS = ["prediction_timestamp_utc", "FL_DATE"]

# A mismatch rate above this makes the verdict REVIEW. It is a prompt to read the
# Phase 3 build script, not a proof of leakage.
REVIEW_MISMATCH_RATE = 0.005


def fail(message: str) -> None:
    """Stop with a clear message. A predictor table that fails a check must never be written."""
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 6 predictor table")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="run all checks but write no files")
    args = parser.parse_args()
    root = args.root.resolve()

    source_path = root / SOURCE_CSV
    cohort_path = root / COHORT_PARQUET
    out_path = root / OUT_PARQUET
    quality_path = root / "reports/tables/phase6_predictor_quality.csv"
    verification_path = root / "reports/tables/phase6_prior_month_verification.csv"
    manifest_path = root / "reports/tables/phase6_predictors_v1_manifest.json"

    # ------------------------------------------------------------------
    # 1. Guard the feature registry, then load
    # ------------------------------------------------------------------
    for name in FEATURE_SETS:
        columns_for(name)  # raises if any set contains an outcome-like or duplicated column
    validate_feature_columns(ALL_FEATURE_COLUMNS)

    for path in (source_path, cohort_path):
        if not path.exists():
            fail(f"file not found: {path}" + ("  (run scripts/build_phase6_cohort.py first)" if path == cohort_path else ""))

    wanted = [ID_COLUMN, TIMESTAMP_COLUMN, *CHECK_COLUMNS, *ALL_FEATURE_COLUMNS]
    header = pd.read_csv(source_path, nrows=0).columns.tolist()
    missing = [c for c in wanted if c not in header]
    if missing:
        fail(f"{source_path.name} is missing columns {missing}. Columns found: {header}")

    print(f"Loading {source_path.name} ({len(wanted)} of {len(header)} columns) ...")
    # Chunked read with compact types. Reading the file in one go needs about 3 GB of RAM.
    predictors = read_predictor_csv(
        source_path,
        columns=wanted,
        categorical_columns=BASE_CATEGORICAL_COLUMNS,
        float32_columns=BASE_NUMERIC_COLUMNS,
    )
    cohort = pd.read_parquet(cohort_path, columns=[ID_COLUMN, "prediction_timestamp_utc", "flight_date", "phase6_split"])
    print(f"  predictor rows: {len(predictors):,}   cohort rows: {len(cohort):,}")

    # ------------------------------------------------------------------
    # 2. Alignment with the cohort table
    # ------------------------------------------------------------------
    try:
        check_alignment(cohort, predictors)
    except ValueError as error:
        fail(str(error))
    print("  alignment with cohort: same flights, same UTC prediction timestamps, same flight dates")

    # The scheduled departure must be exactly two hours after the prediction timestamp.
    lead = predictors[TIMESTAMP_COLUMN] - predictors["prediction_timestamp_utc"]
    off_policy = int((lead != pd.Timedelta(hours=2)).sum())
    if off_policy:
        fail(f"{off_policy:,} rows where scheduled departure minus prediction timestamp is not 2 hours")
    print("  prediction timestamp = scheduled departure UTC minus 2 hours on every row")

    # Sorting copies the table, so skip it when the file is already in id order (the usual case).
    if not predictors[ID_COLUMN].is_monotonic_increasing:
        predictors = predictors.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    split = (
        cohort.set_index(ID_COLUMN)["phase6_split"].astype(str).reindex(predictors[ID_COLUMN]).reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 3. Missing values and unseen categories (development and validation only)
    # ------------------------------------------------------------------
    nulls = null_rate_table(predictors, split, list(ALL_FEATURE_COLUMNS))
    unseen = unseen_category_table(predictors, split, list(BASE_CATEGORICAL_COLUMNS))

    with_nulls = nulls[(nulls["null_rate_development"] > 0) | (nulls["null_rate_validation"] > 0)]
    print("\nFeatures with missing values (development, validation):")
    print(with_nulls.to_string(index=False) if len(with_nulls) else "  none")

    print("\nValidation categories that development never saw:")
    print(unseen.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Independent recomputation of the prior-month features
    # ------------------------------------------------------------------
    print("\nRecomputing the seven prior-month features from the schedule columns ...")
    verification = verify_prior_month_features(predictors)
    print(verification.drop(columns="definition").to_string(index=False))
    worst = float(verification["mismatch_rate"].max())
    if worst <= REVIEW_MISMATCH_RATE:
        verdict = "OK"
    else:
        verdict = "REVIEW"
    print(f"\nPrior-month verification: {verdict}  (worst mismatch rate {worst:.5f}, review above {REVIEW_MISMATCH_RATE})")
    if verdict == "REVIEW":
        print("  The stored features are not exactly 'count or degree in the previous UTC month of this file'.")
        print("  This is not proof of leakage. Send scripts/build_prior_month_schedule_counts.py and")
        print("  scripts/build_prior_month_network_features.py so the definition can be compared.")

    # ------------------------------------------------------------------
    # 5. Assemble and write
    # ------------------------------------------------------------------
    output_columns = [ID_COLUMN, TIMESTAMP_COLUMN, *ALL_FEATURE_COLUMNS]
    table = predictors[output_columns]

    quality = nulls.merge(
        pd.DataFrame(
            {
                "feature": list(ALL_FEATURE_COLUMNS),
                "dtype": [str(table[c].dtype) for c in ALL_FEATURE_COLUMNS],
                "distinct_values": [int(table[c].nunique()) for c in ALL_FEATURE_COLUMNS],
            }
        ),
        on="feature",
    )
    quality.insert(0, "feature_version", FEATURE_VERSION)

    print(f"\nPredictor table: {len(table):,} rows, {len(output_columns)} columns")
    for name, columns in FEATURE_SETS.items():
        print(f"  feature set {name:<22} {len(columns):>2} columns")

    if args.dry_run:
        print("\nDry run: all checks passed. No files written.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    quality.to_csv(quality_path, index=False)
    verification.to_csv(verification_path, index=False)

    manifest = {
        "feature_version": FEATURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": SOURCE_CSV,
        "source_csv_bytes": source_path.stat().st_size,
        "source_csv_sha256": sha256_file(source_path),
        "output_parquet": OUT_PARQUET,
        "output_parquet_bytes": out_path.stat().st_size,
        "output_parquet_sha256": sha256_file(out_path),
        "rows": int(len(table)),
        "columns": output_columns,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "prediction_lead_hours": 2,
        "prior_month_verification_verdict": verdict,
        "prior_month_worst_mismatch_rate": worst,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:,.1f} MB)")
    print(f"Wrote {quality_path}")
    print(f"Wrote {verification_path}")
    print(f"Wrote {manifest_path}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

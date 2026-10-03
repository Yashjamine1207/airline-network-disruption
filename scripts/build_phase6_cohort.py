from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.validation.temporal_splits import (
        SPLIT_EMBARGOED,
        SPLIT_FINAL_TEST,
        assign_regime,
        apply_utc_embargo,
        boundaries_from_config,
        chronology_ok,
        chronology_report,
        to_boolean,
    )
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.validation.temporal_splits import (  # noqa: E402
        SPLIT_EMBARGOED,
        SPLIT_FINAL_TEST,
        assign_regime,
        apply_utc_embargo,
        boundaries_from_config,
        chronology_ok,
        chronology_report,
        to_boolean,
    )

COHORT_VERSION = "phase6_cohort_v1"

MANIFEST_COLUMNS = ["source_row_number", "flight_date", "prediction_timestamp_utc", "temporal_split"]
LABEL_COLUMNS = [
    "source_row_number",
    "FL_DATE",
    "cancelled_target",
    "completed_flight",
    "arrival_delay_minutes",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
]
SEVERE_COLUMNS = ["severe_delay_60", "severe_delay_90", "severe_delay_120", "severe_delay_180"]


def fail(message: str) -> None:
    """Stop with a clear message. A cohort that fails a check must never be written."""
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def read_csv_checked(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        fail(f"file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in columns if c not in header]
    if missing:
        fail(f"{path.name} is missing columns {missing}. Columns found: {header}")
    return pd.read_csv(path, usecols=columns, low_memory=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 6 cohort table")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="run all checks but write no files")
    args = parser.parse_args()
    root = args.root.resolve()

    manifest_path = root / "data/processed/model_ready/temporal_split_manifest.csv"
    labels_path = root / "data/processed/targets/model_labels.csv"
    config_path = root / "configs/validation.yaml"
    out_path = root / f"data/processed/phase6/{COHORT_VERSION}.parquet"
    summary_path = root / "reports/tables/phase6_cohort_summary.csv"

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("Loading manifest and labels ...")
    manifest = read_csv_checked(manifest_path, MANIFEST_COLUMNS)
    labels = read_csv_checked(labels_path, LABEL_COLUMNS)
    boundaries = boundaries_from_config(config_path)
    print(f"  manifest rows: {len(manifest):,}   label rows: {len(labels):,}")
    print(f"  UTC boundaries from {config_path.name}:")
    print(f"    validation starts  {boundaries.validation_start_utc}")
    print(f"    final test starts  {boundaries.final_test_start_utc}")

    # ------------------------------------------------------------------
    # 2. Integrity checks between the two tables
    # ------------------------------------------------------------------
    if manifest["source_row_number"].duplicated().any():
        fail("duplicate source_row_number in the manifest")
    if labels["source_row_number"].duplicated().any():
        fail("duplicate source_row_number in the label table")

    merged = manifest.merge(labels, on="source_row_number", how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        fail(f"manifest and labels do not describe the same rows: {counts}")
    merged = merged.drop(columns="_merge")

    flight_date = pd.to_datetime(merged["flight_date"], format="ISO8601")
    label_date = pd.to_datetime(merged["FL_DATE"], format="ISO8601")
    date_mismatch = int((flight_date != label_date).sum())
    if date_mismatch:
        fail(f"{date_mismatch:,} rows have different dates in the manifest and label table")

    prediction_time = pd.to_datetime(merged["prediction_timestamp_utc"], utc=True, format="ISO8601")
    if prediction_time.isna().any():
        fail(f"{int(prediction_time.isna().sum()):,} unparseable prediction timestamps")

    # ------------------------------------------------------------------
    # 3. Apply the UTC embargo
    # ------------------------------------------------------------------
    original_split = merged["temporal_split"].astype(str)
    unexpected = sorted(set(original_split.unique()) - {"development", "validation", "final_test_locked"})
    if unexpected:
        fail(f"unexpected labels in temporal_split: {unexpected}")
    print("\nChronology BEFORE the embargo (original manifest labels):")
    before = chronology_report(original_split, prediction_time)
    print(before.to_string())
    print(f"  strictly ordered in UTC: {chronology_ok(original_split, prediction_time)}")

    phase6_split = apply_utc_embargo(original_split, prediction_time, boundaries)

    print("\nChronology AFTER the embargo:")
    after = chronology_report(phase6_split, prediction_time)
    print(after.to_string())
    if not chronology_ok(phase6_split, prediction_time):
        fail("splits are still not strictly ordered in UTC after the embargo")
    print("  strictly ordered in UTC: True")

    print("\nOriginal split (rows) x Phase 6 split (columns):")
    print(pd.crosstab(original_split, phase6_split).to_string())

    # The final test must be untouched by the embargo.
    final_before = int((original_split == SPLIT_FINAL_TEST).sum())
    final_after = int((phase6_split == SPLIT_FINAL_TEST).sum())
    if final_before != final_after:
        fail(f"final-test rows changed from {final_before:,} to {final_after:,}")

    # ------------------------------------------------------------------
    # 4. Assemble the cohort table (splits and labels only, no predictors)
    # ------------------------------------------------------------------
    completed = to_boolean(merged["completed_flight"])
    cohort = pd.DataFrame(
        {
            "source_row_number": merged["source_row_number"].astype("int64"),
            "flight_date": flight_date,
            "flight_year": flight_date.dt.year.astype("int16"),
            "regime": assign_regime(merged["flight_date"]),
            "prediction_timestamp_utc": prediction_time,
            "temporal_split_original": pd.Categorical(original_split),
            "phase6_split": phase6_split,
            "embargoed": (phase6_split == SPLIT_EMBARGOED).to_numpy(),
            "cancelled": to_boolean(merged["cancelled_target"]).astype("int8"),
            "completed_flight": completed,
            "arrival_delay_minutes": merged["arrival_delay_minutes"].astype("float64"),
        }
    )
    for column in SEVERE_COLUMNS:
        cohort[column] = merged[column].astype("Int8")

    # Task eligibility. These flags describe label availability only. Select rows with
    # (phase6_split == "development") & eligible_severe_delay, and so on.
    cohort["eligible_cancellation"] = True  # cancellation is defined for every scheduled flight
    cohort["eligible_severe_delay"] = completed & cohort["severe_delay_120"].notna()
    cohort["eligible_regression"] = completed & cohort["arrival_delay_minutes"].notna()

    # Severe-delay labels must exist only for completed flights.
    if cohort.loc[~completed, "severe_delay_120"].notna().any():
        fail("severe_delay_120 is populated for flights that are not completed")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    grouped = cohort.groupby(["phase6_split", "regime"], observed=True)
    summary = grouped.agg(
        rows=("source_row_number", "size"),
        completed_flights=("completed_flight", "sum"),
        cancelled_events=("cancelled", "sum"),
        severe_delay_120_events=("severe_delay_120", "sum"),
    ).reset_index()
    for column in ("completed_flights", "cancelled_events", "severe_delay_120_events"):
        summary[column] = summary[column].astype("float64").astype("int64")
    summary["cancellation_rate"] = (summary["cancelled_events"] / summary["rows"]).round(5)
    summary["severe_delay_120_rate_among_completed"] = (
        summary["severe_delay_120_events"] / summary["completed_flights"].where(summary["completed_flights"] > 0)
    ).round(5)
    summary.insert(0, "cohort_version", COHORT_VERSION)

    print("\nCohort summary (Phase 6 split x regime):")
    print(summary.to_string(index=False))

    retained = cohort.loc[~cohort["embargoed"]]
    print("\nRetained rows per Phase 6 split, and eligible rows per task:")
    per_split = retained.groupby("phase6_split", observed=True).agg(
        rows=("source_row_number", "size"),
        eligible_severe_delay=("eligible_severe_delay", "sum"),
        eligible_regression=("eligible_regression", "sum"),
    )
    print(per_split.to_string())

    # ------------------------------------------------------------------
    # 6. Write
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\nDry run: all checks passed. No files written.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size / 1e6:,.1f} MB, {len(cohort):,} rows)")
    print(f"Wrote {summary_path}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
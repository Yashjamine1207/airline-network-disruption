"""
Phase 3 — Step 3
Validate the target table created in Phase 3 Step 2.

This script checks:
- Row count.
- Required columns.
- Unique source-row identifiers.
- Valid binary source fields.
- Logical consistency between target fields.
- Severe-delay threshold consistency.
- Completed-flight eligibility.
- Target-exclusion reasons.
- Target prevalence and missingness.

The script does not modify the target file.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
    / "flight_targets.csv"
)

EXPECTED_ROW_COUNT = 3_000_000

SEVERE_DELAY_THRESHOLDS = {
    "severe_delay_60": 60,
    "severe_delay_90": 90,
    "severe_delay_120": 120,
    "severe_delay_180": 180,
}

REQUIRED_COLUMNS = {
    "source_row_number",
    "FL_DATE",
    "AIRLINE",
    "AIRLINE_CODE",
    "DOT_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "cancelled_source",
    "diverted_source",
    "cancelled_source_valid",
    "diverted_source_valid",
    "arrival_delay_valid",
    "completed_flight",
    "cancelled_target",
    "arrival_delay_minutes",
    "severe_delay_60",
    "severe_delay_90",
    "severe_delay_120",
    "severe_delay_180",
    "target_exclusion_reason",
    "source_file",
    "target_code_version",
}


def fail(message: str) -> None:
    """Print a validation failure and stop execution."""

    print(f"VALIDATION FAILED: {message}")
    sys.exit(1)


def require(condition: bool, message: str) -> None:
    """Stop execution when a validation condition is false."""

    if not condition:
        fail(message)


def check_binary_or_missing(
    series: pd.Series,
    column_name: str,
) -> None:
    """Check that a nullable binary column contains only 0, 1, or missing."""

    numeric_series = pd.to_numeric(series, errors="coerce")
    invalid = numeric_series.notna() & ~numeric_series.isin([0, 1])

    require(
        not invalid.any(),
        f"{column_name} contains values other than 0, 1, or missing.",
    )


def main() -> None:
    """Run all target-table validation checks."""

    if not TARGET_FILE.exists():
        fail(f"Target file was not found: {TARGET_FILE}")

    print(f"Reading target file:\n{TARGET_FILE}\n")

    targets = pd.read_csv(
        TARGET_FILE,
        low_memory=False,
    )

    print(f"Rows loaded: {len(targets):,}")
    print(f"Columns loaded: {len(targets.columns)}")

    missing_columns = REQUIRED_COLUMNS - set(targets.columns)

    require(
        not missing_columns,
        f"Missing required columns: {sorted(missing_columns)}",
    )

    require(
        len(targets) == EXPECTED_ROW_COUNT,
        (
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"found {len(targets):,}."
        ),
    )

    require(
        targets["source_row_number"].notna().all(),
        "source_row_number contains missing values.",
    )

    require(
        targets["source_row_number"].is_unique,
        "source_row_number is not unique.",
    )

    expected_row_numbers = pd.Series(
        np.arange(EXPECTED_ROW_COUNT),
        name="source_row_number",
    )

    actual_row_numbers = (
        targets["source_row_number"]
        .astype("int64")
        .reset_index(drop=True)
    )

    require(
        actual_row_numbers.equals(expected_row_numbers),
        "source_row_number is not a continuous 0-based source-row sequence.",
    )

    for column in [
        "cancelled_source",
        "diverted_source",
        "cancelled_target",
        "severe_delay_60",
        "severe_delay_90",
        "severe_delay_120",
        "severe_delay_180",
    ]:
        check_binary_or_missing(targets[column], column)

    boolean_columns = [
        "cancelled_source_valid",
        "diverted_source_valid",
        "arrival_delay_valid",
        "completed_flight",
    ]

    for column in boolean_columns:
        values = targets[column].astype(str).str.lower()
        allowed_values = {"true", "false", "0", "1", "nan"}

        require(
            values.isin(allowed_values).all(),
            f"{column} contains unexpected values.",
        )

    cancelled_valid = (
        targets["cancelled_source_valid"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    diverted_valid = (
        targets["diverted_source_valid"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    arrival_delay_valid = (
        targets["arrival_delay_valid"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    completed_flight = (
        targets["completed_flight"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    cancelled_source = pd.to_numeric(
        targets["cancelled_source"],
        errors="coerce",
    )

    diverted_source = pd.to_numeric(
        targets["diverted_source"],
        errors="coerce",
    )

    arrival_delay = pd.to_numeric(
        targets["arrival_delay_minutes"],
        errors="coerce",
    )

    expected_completed = (
        cancelled_valid
        & diverted_valid
        & cancelled_source.eq(0)
        & diverted_source.eq(0)
        & arrival_delay_valid
        & arrival_delay.notna()
        & np.isfinite(arrival_delay)
    )

    require(
        completed_flight.equals(expected_completed),
        "completed_flight does not match its documented eligibility rule.",
    )

    cancelled_target = pd.to_numeric(
        targets["cancelled_target"],
        errors="coerce",
    )

    require(
        cancelled_target.equals(cancelled_source),
        "cancelled_target does not exactly match cancelled_source.",
    )

    for target_name, threshold in SEVERE_DELAY_THRESHOLDS.items():
        target = pd.to_numeric(
            targets[target_name],
            errors="coerce",
        )

        expected_target = pd.Series(
            np.nan,
            index=targets.index,
            dtype="float64",
        )

        eligible = completed_flight & arrival_delay.notna()

        expected_target.loc[eligible] = (
            arrival_delay.loc[eligible] >= threshold
        ).astype(float)

        actual_target = target.astype(float)

        same_values = (
            actual_target.fillna(-999)
            .eq(expected_target.fillna(-999))
        )

        require(
            same_values.all(),
            f"{target_name} does not match ARR_DELAY >= {threshold}.",
        )

    regression_values_present = targets[
        "arrival_delay_minutes"
    ].notna()

    require(
        (~regression_values_present | completed_flight).all(),
        "Non-completed flights contain regression target values.",
    )

    invalid_exclusion_rows = targets[
        "target_exclusion_reason"
    ].isna()

    require(
        not invalid_exclusion_rows.any(),
        "target_exclusion_reason contains missing values.",
    )

    print("\nTarget-table validation passed.")

    print("\nTarget counts and prevalence:")

    for target_name in [
        "cancelled_target",
        "severe_delay_60",
        "severe_delay_90",
        "severe_delay_120",
        "severe_delay_180",
    ]:
        values = pd.to_numeric(
            targets[target_name],
            errors="coerce",
        )

        observed = values.notna()
        positives = values.eq(1)

        observed_count = int(observed.sum())
        positive_count = int(positives.sum())

        prevalence = (
            positive_count / observed_count
            if observed_count > 0
            else float("nan")
        )

        print(
            f"- {target_name}: "
            f"observed={observed_count:,}, "
            f"positive={positive_count:,}, "
            f"prevalence={prevalence:.4%}"
        )

    print("\nCompleted-flight regression cohort:")
    print(f"- Completed flights: {completed_flight.sum():,}")
    print(
        "- Missing regression targets: "
        f"{targets['arrival_delay_minutes'].isna().sum():,}"
    )

    print("\nExclusion reasons:")
    print(
        targets["target_exclusion_reason"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nStep 3 completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
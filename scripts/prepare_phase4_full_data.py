"""Build a full development/validation/final-test Parquet for Phase 4.

Joins the Phase 3 predictor, label, and split tables for ALL rows.
The modelling scripts will filter by temporal_split; this file simply
makes the joined data available without repeated CSV parsing.
"""

import os
from pathlib import Path

import duckdb

from airline_disruption.models.tabular_preprocessing import (
    BASE_CATEGORICAL_COLUMNS,
    BASE_FEATURE_COLUMNS,
    BASE_NUMERIC_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = (
    ROOT / "data/features/tabular/phase3_schedule_network_features.csv"
)
LABELS = ROOT / "data/processed/targets/model_labels.csv"
SPLITS = ROOT / "data/processed/model_ready/temporal_split_manifest.csv"

OUTPUT = (
    ROOT / "data/processed/model_ready/phase4_base_full_splits.parquet"
)
TEMP_OUTPUT = OUTPUT.with_name("phase4_base_full_splits.tmp.parquet")

LABEL_COLUMNS = (
    "cancelled_target",
    "completed_flight",
    "arrival_delay_minutes",
    "severe_delay_120",
)


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def quoted_column(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> None:
    for path in (PREDICTORS, LABELS, SPLITS):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 3 table not found: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = ROOT / "data/interim/duckdb_phase4_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    predictor_columns: list[str] = []
    for name in BASE_NUMERIC_COLUMNS:
        predictor_columns.append(
            f"CAST(p.{quoted_column(name)} AS DOUBLE) AS {quoted_column(name)}"
        )
    for name in BASE_CATEGORICAL_COLUMNS:
        predictor_columns.append(
            f"p.{quoted_column(name)} AS {quoted_column(name)}"
        )

    binary_label_columns = (
        "cancelled_target",
        "completed_flight",
        "severe_delay_120",
    )

    label_columns = []
    for name in binary_label_columns:
        column = f"l.{quoted_column(name)}"
        cleaned = f"lower(trim({column}))"

        label_columns.append(
            f"""
            CASE
                WHEN {column} IS NULL OR trim({column}) = '' THEN NULL
                WHEN {cleaned} = 'true' THEN 1
                WHEN {cleaned} = 'false' THEN 0
                WHEN TRY_CAST({column} AS DOUBLE) = 1 THEN 1
                WHEN TRY_CAST({column} AS DOUBLE) = 0 THEN 0
                ELSE error('Invalid binary value in {name}')
            END AS {quoted_column(name)}
            """
        )

    label_columns.append(
        'CAST(l."arrival_delay_minutes" AS DOUBLE) '
        'AS "arrival_delay_minutes"'
    )

    select_columns = ",\n            ".join(
        [
            "CAST(m.source_row_number AS BIGINT) AS source_row_number",
            "m.temporal_split AS temporal_split",
            *predictor_columns,
            *label_columns,
        ]
    )

    query = f"""
        SELECT
            {select_columns}
        FROM read_csv({sql_path(SPLITS)}, header=true, all_varchar=true) AS m
        INNER JOIN read_csv(
            {sql_path(PREDICTORS)}, header=true, all_varchar=true
        ) AS p
            ON CAST(m.source_row_number AS BIGINT)
             = CAST(p.source_row_number AS BIGINT)
        INNER JOIN read_csv(
            {sql_path(LABELS)}, header=true, all_varchar=true
        ) AS l
            ON CAST(m.source_row_number AS BIGINT)
             = CAST(l.source_row_number AS BIGINT)
    """

    connection = duckdb.connect()
    try:
        connection.execute(
            f"SET temp_directory = {sql_path(temp_directory)}"
        )

        expected_total = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_csv({sql_path(SPLITS)}, header=true, all_varchar=true)
            WHERE temporal_split IN ('development', 'validation', 'final_test_locked')
            """
        ).fetchone()[0]

        connection.execute(
            f"""
            COPY ({query})
            TO {sql_path(TEMP_OUTPUT)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        split_counts = dict(
            connection.execute(
                f"""
                SELECT temporal_split, COUNT(*)
                FROM read_parquet({sql_path(TEMP_OUTPUT)})
                GROUP BY temporal_split
                """
            ).fetchall()
        )

        actual_rows, unique_ids = connection.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT source_row_number)
            FROM read_parquet({sql_path(TEMP_OUTPUT)})
            """
        ).fetchone()

        if actual_rows != unique_ids:
            raise ValueError("Duplicate source_row_number after join.")

        if actual_rows != expected_total:
            raise ValueError(
                f"Row count mismatch: expected {expected_total}, got {actual_rows}."
            )

        os.replace(TEMP_OUTPUT, OUTPUT)

        print(f"Saved: {OUTPUT}")
        print(f"Total rows: {actual_rows:,}")
        print("Split counts:")
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count:,}")
    finally:
        connection.close()
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()


if __name__ == "__main__":
    main()
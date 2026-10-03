"""Build the Phase 4 base-feature development/validation table.

The output contains both predictors and labels. It is NOT itself an X
matrix: training code must call select_base_features() to obtain X.

Only development and validation rows are written. The final-test split
is not included in the output.
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
OUTPUT = ROOT / "data/processed/model_ready/phase4_base_dev_validation.parquet"
TEMP_OUTPUT = OUTPUT.with_name("phase4_base_dev_validation.tmp.parquet")

LABEL_COLUMNS = (
    "cancelled_target",
    "completed_flight",
    "arrival_delay_minutes",
    "severe_delay_120",
)


def sql_path(path: Path) -> str:
    """Quote a local path safely as a SQL string literal."""
    return "'" + path.as_posix().replace("'", "''") + "'"


def quoted_column(name: str) -> str:
    """Quote one known project column as a SQL identifier."""
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

    # Keep the source CSVs as strings at ingestion, then explicitly cast
    # only columns needed for this modelling table.
    predictor_columns: list[str] = []
    for name in BASE_NUMERIC_COLUMNS:
        predictor_columns.append(
            f"CAST(p.{quoted_column(name)} AS DOUBLE) AS {quoted_column(name)}"
        )
    for name in BASE_CATEGORICAL_COLUMNS:
        predictor_columns.append(
            f"p.{quoted_column(name)} AS {quoted_column(name)}"
        )

    # Phase 3 binary columns use mixed text formats: True/False and 0.0/1.0.
    # Accept only genuine binary values; do not silently convert bad values.
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
    
    # Arrival delay is numeric, not a binary indicator.
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
        WHERE m.temporal_split IN ('development', 'validation')
    """

    connection = duckdb.connect()
    try:
        connection.execute(
            f"SET temp_directory = {sql_path(temp_directory)}"
        )

        # Obtain expected counts without looking at outcome values.
        expected = dict(
            connection.execute(
                f"""
                SELECT temporal_split, COUNT(*)
                FROM read_csv(
                    {sql_path(SPLITS)}, header=true, all_varchar=true
                )
                WHERE temporal_split IN ('development', 'validation')
                GROUP BY temporal_split
                """
            ).fetchall()
        )

        if set(expected) != {"development", "validation"}:
            raise ValueError(
                f"Expected development and validation splits; got {expected}"
            )

        connection.execute(
            f"""
            COPY ({query})
            TO {sql_path(TEMP_OUTPUT)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        actual_rows, unique_ids, other_splits = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT source_row_number),
                COUNT(*) FILTER (
                    WHERE temporal_split NOT IN ('development', 'validation')
                       OR temporal_split IS NULL
                )
            FROM read_parquet({sql_path(TEMP_OUTPUT)})
            """
        ).fetchone()

        split_counts = dict(
            connection.execute(
                f"""
                SELECT temporal_split, COUNT(*)
                FROM read_parquet({sql_path(TEMP_OUTPUT)})
                GROUP BY temporal_split
                """
            ).fetchall()
        )

        if split_counts != expected:
            raise ValueError(
                "Joined row counts do not match the split manifest. "
                f"Expected {expected}; found {split_counts}."
            )

        if actual_rows != unique_ids:
            raise ValueError(
                "Duplicate source_row_number found after joining the tables."
            )

        if other_splits != 0:
            raise ValueError("A locked-test or unknown split entered the output.")

        # Replace the previous local derived table only after all checks pass.
        os.replace(TEMP_OUTPUT, OUTPUT)

        print(f"Saved: {OUTPUT}")
        print(f"Columns: {2 + len(BASE_FEATURE_COLUMNS) + len(LABEL_COLUMNS)}")
        print(f"Development rows: {split_counts['development']:,}")
        print(f"Validation rows: {split_counts['validation']:,}")
        print("Locked final-test rows written: 0")
    finally:
        connection.close()
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()


if __name__ == "__main__":
    main()
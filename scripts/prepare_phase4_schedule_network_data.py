"""Add Phase 3 prior-month schedule and network features to Phase 4 data.

The existing base Parquet remains unchanged. This derived table includes
development and validation only; it contains labels for evaluation, so
training code must still select X through an explicit feature allowlist.
"""

import os
from pathlib import Path

import duckdb

from scripts.prepare_phase4_base_data import (
    OUTPUT as BASE_INPUT,
    PREDICTORS as PHASE3_PREDICTORS,
    ROOT,
    sql_path,
)


OUTPUT = (
    ROOT
    / "data/processed/model_ready/"
    "phase4_schedule_network_dev_validation.parquet"
)
TEMP_OUTPUT = OUTPUT.with_name(
    "phase4_schedule_network_dev_validation.tmp.parquet"
)

ADDITIONAL_FEATURES = (
    "prior_calendar_month_carrier_scheduled_flight_count",
    "prior_calendar_month_origin_scheduled_flight_count",
    "prior_calendar_month_route_scheduled_flight_count",
    "prior_month_origin_out_degree",
    "prior_month_origin_weighted_out_degree",
    "prior_month_destination_in_degree",
    "prior_month_destination_weighted_in_degree",
)


def main() -> None:
    for path in (BASE_INPUT, PHASE3_PREDICTORS):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = ROOT / "data/interim/duckdb_phase4_tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    # Every added predictor was constructed in Phase 3 from a preceding
    # calendar month's schedules, not from future flight outcomes.
    extra_columns = ",\n            ".join(
        f'CAST(p."{name}" AS DOUBLE) AS "{name}"'
        for name in ADDITIONAL_FEATURES
    )

    connection = duckdb.connect()
    try:
        connection.execute(
            f"SET temp_directory = {sql_path(temporary_directory)}"
        )

        expected_counts = dict(
            connection.execute(
                f"""
                SELECT temporal_split, COUNT(*)
                FROM read_parquet({sql_path(BASE_INPUT)})
                GROUP BY temporal_split
                """
            ).fetchall()
        )

        if set(expected_counts) != {"development", "validation"}:
            raise ValueError(
                "Base table must contain development and validation only."
            )

        # An inner join must retain every base row exactly once. The
        # checks below catch missing predictor rows or duplicate IDs.
        connection.execute(
            f"""
            COPY (
                SELECT
                    b.*,
                    {extra_columns}
                FROM read_parquet({sql_path(BASE_INPUT)}) AS b
                INNER JOIN read_csv(
                    {sql_path(PHASE3_PREDICTORS)},
                    header=true,
                    all_varchar=true
                ) AS p
                    ON b.source_row_number
                     = CAST(p.source_row_number AS BIGINT)
            )
            TO {sql_path(TEMP_OUTPUT)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        actual_counts = dict(
            connection.execute(
                f"""
                SELECT temporal_split, COUNT(*)
                FROM read_parquet({sql_path(TEMP_OUTPUT)})
                GROUP BY temporal_split
                """
            ).fetchall()
        )

        total_rows, unique_ids, total_columns = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT source_row_number),
                COUNT(*) -- Checked separately against the expected schema below.
            FROM read_parquet({sql_path(TEMP_OUTPUT)})
            """
        ).fetchone()

        schema_columns = [
            item[0]
            for item in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_path(TEMP_OUTPUT)})"
            ).fetchall()
        ]

        if actual_counts != expected_counts:
            raise ValueError(
                f"Split row counts changed: "
                f"expected {expected_counts}, got {actual_counts}."
            )

        if total_rows != unique_ids:
            raise ValueError("Duplicate source-row IDs after feature join.")

        if not set(ADDITIONAL_FEATURES).issubset(schema_columns):
            raise ValueError("An added feature is missing from the output.")

        if len(schema_columns) != 26:
            raise ValueError(
                f"Expected 26 columns; found {len(schema_columns)}."
            )

        # The temporary output becomes the official local derived table
        # only after its row and schema checks pass.
        os.replace(TEMP_OUTPUT, OUTPUT)

        print(f"Saved: {OUTPUT}")
        print(f"Columns: {len(schema_columns)}")
        print(
            f"Development rows: {actual_counts['development']:,}"
        )
        print(
            f"Validation rows: {actual_counts['validation']:,}"
        )
        print("Locked final-test rows written: 0")
    finally:
        connection.close()
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()


if __name__ == "__main__":
    main()
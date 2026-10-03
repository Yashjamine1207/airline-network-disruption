"""Aggregate validated flight outcomes by scheduled-departure UTC year.

Joins Phase 3 predictors to validated targets on source_row_number.
The output is retrospective: never use its outcome rates as pre-flight
features without a separate point-in-time availability design.
"""

from pathlib import Path
import os
import sys

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = ROOT / "data/features/base/base_predictors.csv"
TARGETS = ROOT / "data/processed/targets/flight_targets.csv"
GRAPH_ROUTES = (
    ROOT / "data/processed/network_windows/yearly_route_counts.csv"
)
OUTPUT = ROOT / "reports/tables/route_disruption_by_utc_year.csv"

EXPECTED_JOINED_ROWS = 2_999_999
EXPECTED_BOUNDARY_ROWS = 1


def sql_path(path: Path) -> str:
    """Return a safely quoted local path for a DuckDB SQL string."""
    return path.as_posix().replace("'", "''")


def write_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Write the completed output before replacing an existing CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    for path in (PREDICTORS, TARGETS, GRAPH_ROUTES):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET TimeZone = 'UTC'")

    try:
        # all_varchar avoids CSV type inference changing identifiers or
        # nullable target fields. Convert only fields needed below.
        connection.execute(f"""
            CREATE TEMP TABLE cohort AS
            SELECT
                CAST(p.source_row_number AS BIGINT)
                    AS source_row_number,
                p.FL_DATE AS predictor_flight_date,
                t.FL_DATE AS target_flight_date,
                p.origin_airport,
                p.destination_airport,
                t.ORIGIN AS target_origin,
                t.DEST AS target_destination,
                EXTRACT(
                    YEAR FROM CAST(
                        p.scheduled_departure_utc AS TIMESTAMPTZ
                    )
                )::INTEGER AS utc_year,
                CASE WHEN lower(trim(coalesce(
                    t.cancelled_target, ''
                ))) IN ('1', '1.0', 'true')
                    THEN 1 ELSE 0 END AS cancelled_flag,
                CASE WHEN lower(trim(coalesce(
                    t.diverted_source, ''
                ))) IN ('1', '1.0', 'true')
                    THEN 1 ELSE 0 END AS diverted_flag,
                CASE WHEN lower(trim(coalesce(
                    t.completed_flight, ''
                ))) IN ('1', '1.0', 'true')
                    THEN 1 ELSE 0 END AS completed_flag,
                CASE WHEN lower(trim(coalesce(
                    t.severe_delay_120, ''
                ))) IN ('1', '1.0', 'true')
                    THEN 1 ELSE 0 END AS severe_flag
            FROM read_csv(
                '{sql_path(PREDICTORS)}',
                all_varchar = true
            ) AS p
            INNER JOIN read_csv(
                '{sql_path(TARGETS)}',
                all_varchar = true
            ) AS t
                ON p.source_row_number = t.source_row_number
        """)

        audit = connection.execute("""
            SELECT
                COUNT(*) AS joined_rows,
                COUNT(DISTINCT source_row_number) AS distinct_ids,
                SUM(CASE
                    WHEN predictor_flight_date
                        IS DISTINCT FROM target_flight_date
                      OR origin_airport
                        IS DISTINCT FROM target_origin
                      OR destination_airport
                        IS DISTINCT FROM target_destination
                    THEN 1 ELSE 0
                END) AS identity_mismatches,
                SUM(CASE WHEN utc_year = 2018
                    THEN 1 ELSE 0 END) AS boundary_rows,
                SUM(CASE
                    WHEN utc_year NOT BETWEEN 2019 AND 2023
                         AND utc_year <> 2018
                    THEN 1 ELSE 0
                END) AS other_year_rows
            FROM cohort
        """).fetchone()

        joined, distinct_ids, mismatches, boundary, other_years = audit

        if joined != EXPECTED_JOINED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_JOINED_ROWS:,} joined rows; "
                f"found {joined:,}"
            )
        if distinct_ids != joined or mismatches != 0:
            raise ValueError(
                f"Invalid join: distinct IDs={distinct_ids:,}, "
                f"identity mismatches={mismatches:,}"
            )
        if boundary != EXPECTED_BOUNDARY_ROWS or other_years != 0:
            raise ValueError(
                f"Unexpected UTC-year coverage: "
                f"2018 boundary={boundary}, other={other_years}"
            )

        outcomes = connection.execute("""
            SELECT
                utc_year,
                origin_airport,
                destination_airport,
                COUNT(*)::BIGINT AS scheduled_flights,
                SUM(cancelled_flag)::BIGINT AS cancelled_flights,
                SUM(diverted_flag)::BIGINT AS diverted_flights,
                SUM(completed_flag)::BIGINT AS completed_arrivals,
                SUM(severe_flag)::BIGINT AS severe_delay_flights,
                SUM(cancelled_flag)::DOUBLE
                    / COUNT(*) AS cancellation_rate,
                SUM(diverted_flag)::DOUBLE
                    / COUNT(*) AS diversion_rate,
                SUM(severe_flag)::DOUBLE
                    / NULLIF(SUM(completed_flag), 0)
                    AS severe_delay_rate_completed
            FROM cohort
            WHERE utc_year BETWEEN 2019 AND 2023
            GROUP BY utc_year, origin_airport, destination_airport
            ORDER BY utc_year, origin_airport, destination_airport
        """).fetchdf()

        # The schedule count for every UTC-year route must agree with the
        # independently built, outcome-free graph edge table.
        graph = pd.read_csv(
            GRAPH_ROUTES,
            usecols=[
                "utc_year",
                "origin_airport",
                "destination_airport",
                "scheduled_flight_count",
            ],
        )
        keys = [
            "utc_year",
            "origin_airport",
            "destination_airport",
        ]
        compared = graph.merge(
            outcomes,
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not compared["_merge"].eq("both").all():
            raise ValueError(
                "UTC-year route keys differ from the graph table."
            )
        if not compared["scheduled_flight_count"].eq(
            compared["scheduled_flights"]
        ).all():
            raise ValueError(
                "UTC-year flight counts differ from graph weights."
            )

        if (
            int(outcomes["scheduled_flights"].sum())
            != 2_999_998
        ):
            raise ValueError("UTC-year cohort total does not reconcile.")

        write_atomically(outcomes, OUTPUT)

        print("UTC-year route disruption summary: PASS")
        print(f"Joined source rows: {joined:,}")
        print(f"Pre-2019 UTC boundary rows: {boundary:,}")
        print(f"Route rows matching graph edges: {len(outcomes):,}")
        print(
            outcomes.groupby("utc_year")[
                [
                    "scheduled_flights",
                    "cancelled_flights",
                    "diverted_flights",
                    "completed_arrivals",
                    "severe_delay_flights",
                ]
            ].sum().to_string()
        )
        print(f"Output: {OUTPUT}")
        print("Retrospective outcomes only; not pre-flight features.")

    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
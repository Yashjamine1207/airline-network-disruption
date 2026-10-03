"""Retrospective association between consecutive UTC airport-months.

Outcome labels are used for retrospective analysis only. This output
must not be joined into pre-flight predictive features.
"""

from pathlib import Path
import os
import sys

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = (
    ROOT / "data/processed/model_ready/"
    "phase4_base_full_splits.parquet"
)
MANIFEST = (
    ROOT / "data/processed/model_ready/"
    "temporal_split_manifest.csv"
)
DETAIL = (
    ROOT / "data/processed/network_windows/"
    "airport_month_lag_disruption.parquet"
)
SUMMARY = (
    ROOT / "reports/tables/"
    "lagged_airport_disruption_association.csv"
)

MIN_SCHEDULED = 50
MIN_COMPLETED = 50


def within_airport_correlation(frame, prior, current):
    centred_prior = prior - frame.groupby("airport")[prior.name].transform("mean")
    centred_current = current - frame.groupby("airport")[current.name].transform("mean")
    return centred_prior.corr(centred_current)


def main() -> None:
    for path in (FLIGHTS, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")

    flights_path = FLIGHTS.as_posix().replace("'", "''")
    manifest_path = MANIFEST.as_posix().replace("'", "''")

    connection.execute(f"""
        CREATE TEMP VIEW flights AS
        SELECT source_row_number, origin_airport,
               cancelled_target, completed_flight, severe_delay_120
        FROM read_parquet('{flights_path}')
    """)
    connection.execute(f"""
        CREATE TEMP VIEW manifest AS
        SELECT source_row_number, prediction_timestamp_utc
        FROM read_csv(
            '{manifest_path}',
            header=true,
            columns={{
                'source_row_number': 'BIGINT',
                'flight_date': 'VARCHAR',
                'prediction_timestamp_utc': 'VARCHAR',
                'temporal_split': 'VARCHAR'
            }}
        )
    """)

    flight_count, distinct_flights = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT source_row_number)
        FROM flights
    """).fetchone()
    manifest_count, distinct_manifest = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT source_row_number)
        FROM manifest
    """).fetchone()

    if (
        flight_count != manifest_count
        or flight_count != distinct_flights
        or manifest_count != distinct_manifest
    ):
        raise ValueError(
            "Flight/manifest counts or source row uniqueness failed"
        )

    connection.execute("""
        CREATE TEMP TABLE monthly AS
        SELECT
            f.origin_airport AS airport,
            DATE_TRUNC(
                'month',
                CAST(m.prediction_timestamp_utc AS TIMESTAMPTZ)
            )::DATE AS month_utc,
            COUNT(*) AS scheduled_flights,
            SUM(
                CASE WHEN TRY_CAST(f.cancelled_target AS INTEGER) = 1
                     THEN 1 ELSE 0 END
            ) AS cancelled_flights,
            SUM(
                CASE WHEN TRY_CAST(f.completed_flight AS INTEGER) = 1
                     THEN 1 ELSE 0 END
            ) AS completed_arrivals,
            SUM(
                CASE WHEN TRY_CAST(f.completed_flight AS INTEGER) = 1
                       AND TRY_CAST(f.severe_delay_120 AS INTEGER) = 1
                     THEN 1 ELSE 0 END
            ) AS severe_delay_flights
        FROM flights f
        JOIN manifest m USING (source_row_number)
        WHERE m.prediction_timestamp_utc IS NOT NULL
          AND f.origin_airport IS NOT NULL
        GROUP BY 1, 2
    """)

    matched_count = connection.execute("""
        SELECT SUM(scheduled_flights) FROM monthly
    """).fetchone()[0]
    if matched_count != flight_count:
        raise ValueError(
            "Joined or timestamped flight count differs from input"
        )

    pairs = connection.execute(f"""
        WITH rates AS (
            SELECT *,
                   severe_delay_flights::DOUBLE
                       / NULLIF(completed_arrivals, 0)
                       AS severe_rate_completed,
                   cancelled_flights::DOUBLE
                       / scheduled_flights AS cancellation_rate
            FROM monthly
        )
        SELECT
            current.airport,
            current.month_utc,
            previous.scheduled_flights AS previous_scheduled,
            current.scheduled_flights AS current_scheduled,
            previous.completed_arrivals AS previous_completed,
            current.completed_arrivals AS current_completed,
            previous.severe_rate_completed AS previous_severe_rate,
            current.severe_rate_completed AS current_severe_rate,
            previous.cancellation_rate AS previous_cancellation_rate,
            current.cancellation_rate AS current_cancellation_rate
        FROM rates current
        JOIN rates previous
          ON current.airport = previous.airport
         AND previous.month_utc =
             (current.month_utc - INTERVAL '1 month')::DATE
        WHERE current.month_utc >= DATE '2019-01-01'
          AND current.month_utc < DATE '2024-01-01'
          AND previous.month_utc >= DATE '2019-01-01'
          AND current.scheduled_flights >= {MIN_SCHEDULED}
          AND previous.scheduled_flights >= {MIN_SCHEDULED}
          AND current.completed_arrivals >= {MIN_COMPLETED}
          AND previous.completed_arrivals >= {MIN_COMPLETED}
        ORDER BY current.airport, current.month_utc
    """).df()

    if pairs.empty:
        raise ValueError("No airport-month pairs meet the minimums")

    if pairs[[
        "previous_severe_rate", "current_severe_rate",
        "previous_cancellation_rate", "current_cancellation_rate",
    ]].isna().any().any():
        raise ValueError("Unexpected missing association rates")

    findings = []
    for outcome in ("severe", "cancellation"):
        previous = pairs[f"previous_{outcome}_rate"]
        current = pairs[f"current_{outcome}_rate"]
        findings.append({
            "outcome": outcome,
            "airport_month_pairs": len(pairs),
            "airports": pairs["airport"].nunique(),
            "first_current_month_utc": pairs["month_utc"].min(),
            "last_current_month_utc": pairs["month_utc"].max(),
            "pooled_pearson_correlation": previous.corr(current),
            "within_airport_pearson_correlation":
                within_airport_correlation(pairs, previous, current),
            "minimum_scheduled_each_month": MIN_SCHEDULED,
            "minimum_completed_each_month": MIN_COMPLETED,
            "scope": "retrospective_association_not_predictive_feature",
        })

    result = pd.DataFrame(findings)
    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    temporary_detail = DETAIL.with_name(DETAIL.name + ".tmp")
    temporary_summary = SUMMARY.with_name(SUMMARY.name + ".tmp")
    try:
        pairs.to_parquet(temporary_detail, index=False)
        result.to_csv(temporary_summary, index=False)
        os.replace(temporary_detail, DETAIL)
        os.replace(temporary_summary, SUMMARY)
    finally:
        temporary_detail.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)

    print("Lagged airport-month association: PASS")
    print(f"Matched flights: {matched_count:,}")
    print(result.to_string(index=False))
    print(f"Detail: {DETAIL}")
    print(f"Summary: {SUMMARY}")
    print(
        "Retrospective association only; outcome availability "
        "at a pre-flight prediction time was not established."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
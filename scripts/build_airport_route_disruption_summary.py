"""Build retrospective airport- and route-level disruption summaries.

This script uses source flight outcomes only for descriptive analysis.
It does not create pre-flight predictors and does not claim causality,
aircraft-tail propagation, weather attribution, or operational control.
"""

from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"

AIRPORT_OUTPUT = (
    ROOT
    / "reports/tables/airport_disruption_by_year.csv"
)
ROUTE_OUTPUT = (
    ROOT
    / "reports/tables/route_disruption_by_year.csv"
)

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 3_000_000
SEVERE_DELAY_THRESHOLD = 120

INPUT_COLUMNS = [
    "FL_DATE",
    "ORIGIN",
    "DEST",
    "CANCELLED",
    "DIVERTED",
    "ARR_DELAY",
]


def validate_columns() -> None:
    """Confirm that the audited source fields are present."""
    available = set(pd.read_csv(INPUT, nrows=0).columns)
    missing = set(INPUT_COLUMNS) - available

    if missing:
        raise ValueError(
            f"Missing source columns: {sorted(missing)}"
        )


def classify_regime(year: int) -> str:
    """Return the documented operating-regime label."""
    if year == 2019:
        return "pre_covid_2019"
    if year == 2020:
        return "covid_shock_2020"
    if year in (2021, 2022):
        return "recovery_transition_2021_2022"
    if year == 2023:
        return "partial_2023"
    return "outside_project_scope"


def atomic_write(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Write a CSV through a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Aggregate cancellation and severe-delay outcomes."""
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing raw input: {INPUT}")

    validate_columns()

    airport_parts = []
    route_parts = []
    total_rows = 0

    reader = pd.read_csv(
        INPUT,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        chunk["year"] = pd.to_datetime(
            chunk["FL_DATE"].astype(str),
            format="%Y-%m-%d",
            errors="coerce",
        ).dt.year

        if chunk["year"].isna().any():
            raise ValueError(
                f"Invalid FL_DATE values in chunk {chunk_number}"
            )

        if not chunk["year"].isin(
            [2019, 2020, 2021, 2022, 2023]
        ).all():
            raise ValueError(
                f"Unexpected year in chunk {chunk_number}"
            )

        chunk["cancelled_flag"] = (
            pd.to_numeric(
                chunk["CANCELLED"],
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
            .astype("int8")
        )

        chunk["diverted_flag"] = (
            pd.to_numeric(
                chunk["DIVERTED"],
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
            .astype("int8")
        )

        arrival_delay = pd.to_numeric(
            chunk["ARR_DELAY"],
            errors="coerce",
        )

        chunk["completed_arrival_flag"] = (
            chunk["cancelled_flag"].eq(0)
            & chunk["diverted_flag"].eq(0)
            & arrival_delay.notna()
        ).astype("int8")

        chunk["severe_delay_flag"] = (
            chunk["completed_arrival_flag"].eq(1)
            & arrival_delay.ge(SEVERE_DELAY_THRESHOLD)
        ).astype("int8")

        chunk["valid_origin"] = (
            chunk["ORIGIN"].notna()
            & chunk["ORIGIN"].astype("string").str.strip().ne("")
        )

        chunk["valid_destination"] = (
            chunk["DEST"].notna()
            & chunk["DEST"].astype("string").str.strip().ne("")
        )

        valid_route = (
            chunk["valid_origin"]
            & chunk["valid_destination"]
            & chunk["ORIGIN"].ne(chunk["DEST"])
        )

        if not valid_route.all():
            raise ValueError(
                f"Invalid airport route in chunk {chunk_number}"
            )

        chunk["origin_airport"] = (
            chunk["ORIGIN"].astype("string").str.strip()
        )
        chunk["destination_airport"] = (
            chunk["DEST"].astype("string").str.strip()
        )

        airport_origin = pd.DataFrame({
            "year": chunk["year"],
            "airport": chunk["origin_airport"],
            "airport_role": "origin",
            "scheduled_flights": 1,
            "cancelled_flights": chunk["cancelled_flag"],
            "diverted_flights": chunk["diverted_flag"],
            "completed_arrivals": chunk["completed_arrival_flag"],
            "severe_delay_flights": chunk["severe_delay_flag"],
        })

        airport_destination = pd.DataFrame({
            "year": chunk["year"],
            "airport": chunk["destination_airport"],
            "airport_role": "destination",
            "scheduled_flights": 1,
            "cancelled_flights": chunk["cancelled_flag"],
            "diverted_flights": chunk["diverted_flag"],
            "completed_arrivals": chunk["completed_arrival_flag"],
            "severe_delay_flights": chunk["severe_delay_flag"],
        })

        airport_parts.extend([
            airport_origin,
            airport_destination,
        ])

        route_parts.append(pd.DataFrame({
            "year": chunk["year"],
            "origin_airport": chunk["origin_airport"],
            "destination_airport": chunk["destination_airport"],
            "scheduled_flights": 1,
            "cancelled_flights": chunk["cancelled_flag"],
            "diverted_flights": chunk["diverted_flag"],
            "completed_arrivals": chunk["completed_arrival_flag"],
            "severe_delay_flights": chunk["severe_delay_flag"],
        }))

        print(
            f"Outcome aggregation pass {chunk_number}: "
            f"{total_rows:,} rows"
        )

    if total_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; "
            f"found {total_rows:,}"
        )

    airports = pd.concat(airport_parts, ignore_index=True)
    routes = pd.concat(route_parts, ignore_index=True)

    airport_summary = (
        airports.groupby(
            ["year", "airport", "airport_role"],
            as_index=False,
        )
        .sum(numeric_only=True)
    )

    route_summary = (
        routes.groupby(
            ["year", "origin_airport", "destination_airport"],
            as_index=False,
        )
        .sum(numeric_only=True)
    )

    for frame in (airport_summary, route_summary):
        frame["regime"] = frame["year"].map(
            classify_regime
        )
        frame["cancellation_rate"] = (
            frame["cancelled_flights"]
            / frame["scheduled_flights"]
        )
        frame["diversion_rate"] = (
            frame["diverted_flights"]
            / frame["scheduled_flights"]
        )
        frame["severe_delay_rate_completed"] = (
            frame["severe_delay_flights"]
            / frame["completed_arrivals"].replace(0, pd.NA)
        )

    airport_summary["analysis_scope"] = (
        "retrospective_airport_disruption_association"
    )
    route_summary["analysis_scope"] = (
        "retrospective_route_disruption_association"
    )

    atomic_write(airport_summary, AIRPORT_OUTPUT)
    atomic_write(route_summary, ROUTE_OUTPUT)

    print("\nAirport and route disruption summaries created.")
    print(f"Input rows: {total_rows:,}")
    print(f"Airport summary rows: {len(airport_summary):,}")
    print(f"Route summary rows: {len(route_summary):,}")
    print(f"Airport output: {AIRPORT_OUTPUT}")
    print(f"Route output: {ROUTE_OUTPUT}")
    print(
        "\nThis is retrospective descriptive analysis only."
    )
    print(
        "No aircraft-tail, weather, causal, or pre-flight "
        "predictive claim is produced."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
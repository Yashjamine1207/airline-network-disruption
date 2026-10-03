"""Build descriptive yearly airport-route graphs from scheduled flights.

Each graph covers one UTC calendar year. A directed origin-to-destination
edge is weighted by its count of eligible scheduled flights.

This is retrospective descriptive analysis, not a pre-flight feature builder.
It reads no flight labels, actual outcomes, or 2023 model predictions.
"""

from collections import Counter, defaultdict
from pathlib import Path
import os
import sys

import pandas as pd

from airline_disruption.network.graph_builder import (
    build_directed_route_graph,
    summarise_route_graph,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/features/base/base_predictors.csv"
ROUTE_OUTPUT = (
    ROOT / "data/processed/network_windows/yearly_route_counts.csv"
)
SUMMARY_OUTPUT = ROOT / "reports/tables/yearly_network_summary.csv"

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999
YEARS = (2019, 2020, 2021, 2022, 2023)

INPUT_COLUMNS = [
    "source_row_number",
    "scheduled_departure_utc",
    "prediction_timestamp_utc",
    "origin_airport",
    "destination_airport",
]


def count_yearly_routes() -> tuple[Counter, dict, dict]:
    """Count routes in UTC-year windows and audit excluded rows."""
    route_counts: Counter = Counter()
    observed_bounds: dict[int, dict[str, pd.Timestamp]] = {}

    audit = {
        "input_rows": 0,
        "pre_2019_utc_boundary_rows": 0,
        "invalid_route_rows_in_window": 0,
        "boundary_source_row_examples": [],
    }
    previous_row_id = -1

    reader = pd.read_csv(
        INPUT,
        usecols=INPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        row_ids = pd.to_numeric(
            chunk["source_row_number"],
            errors="coerce",
        )
        if (
            row_ids.isna().any()
            or not row_ids.is_unique
            or not row_ids.is_monotonic_increasing
            or int(row_ids.iloc[0]) <= previous_row_id
        ):
            raise ValueError(
                f"Invalid source-row order in chunk {chunk_number}"
            )
        previous_row_id = int(row_ids.iloc[-1])

        departure = pd.to_datetime(
            chunk["scheduled_departure_utc"],
            utc=True,
            errors="coerce",
        )
        prediction = pd.to_datetime(
            chunk["prediction_timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        if departure.isna().any() or prediction.isna().any():
            raise ValueError(
                f"Invalid UTC timestamp in chunk {chunk_number}"
            )

        if not (
            departure - prediction
        ).eq(pd.Timedelta(hours=2)).all():
            raise ValueError(
                f"Prediction-horizon mismatch in chunk {chunk_number}"
            )

        # The source has a local-2019 SPN departure that occurred during
        # 2018-12-31 UTC. It is valid, but outside the defined UTC windows.
        boundary = (
            (departure >= pd.Timestamp("2018-12-31T00:00:00Z"))
            & (departure < pd.Timestamp("2019-01-01T00:00:00Z"))
        )
        in_window = departure.dt.year.isin(YEARS)
        unexpected = ~(boundary | in_window)

        if unexpected.any():
            examples = (
                chunk.loc[
                    unexpected,
                    ["source_row_number", "scheduled_departure_utc"],
                ]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                f"Unexpected UTC departures: {examples}"
            )

        audit["pre_2019_utc_boundary_rows"] += int(boundary.sum())
        remaining_example_slots = (
            10 - len(audit["boundary_source_row_examples"])
        )
        if remaining_example_slots > 0:
            audit["boundary_source_row_examples"].extend(
                row_ids.loc[boundary]
                .head(remaining_example_slots)
                .astype(int)
                .tolist()
            )

        origin = chunk["origin_airport"].astype("string")
        destination = chunk["destination_airport"].astype("string")

        valid_route = (
            origin.notna()
            & destination.notna()
            & origin.str.strip().ne("")
            & destination.str.strip().ne("")
            & origin.eq(origin.str.strip())
            & destination.eq(destination.str.strip())
            & origin.ne(destination)
        ).fillna(False)

        invalid_in_window = in_window & ~valid_route
        audit["invalid_route_rows_in_window"] += int(
            invalid_in_window.sum()
        )

        eligible = in_window & valid_route
        years = departure.dt.year

        grouped = (
            pd.DataFrame({
                "year": years.loc[eligible].to_numpy(),
                "origin": origin.loc[eligible].to_numpy(),
                "destination": destination.loc[eligible].to_numpy(),
            })
            .groupby(["year", "origin", "destination"], sort=False)
            .size()
        )

        for (year, start, end), count in grouped.items():
            route_counts[
                (int(year), str(start), str(end))
            ] += int(count)

        # Keep the original timezone-aware pandas timestamps. Do not
        # tz_localize() them again; they are already in UTC.
        for year in YEARS:
            year_departures = departure.loc[
                eligible & years.eq(year)
            ]
            if year_departures.empty:
                continue

            minimum = year_departures.min()
            maximum = year_departures.max()

            if year not in observed_bounds:
                observed_bounds[year] = {
                    "first": minimum,
                    "last": maximum,
                }
            else:
                bounds = observed_bounds[year]
                bounds["first"] = min(bounds["first"], minimum)
                bounds["last"] = max(bounds["last"], maximum)

        audit["input_rows"] += len(chunk)
        print(
            f"Count pass {chunk_number}: "
            f"{audit['input_rows']:,} rows read"
        )

    if audit["input_rows"] != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; "
            f"found {audit['input_rows']:,}"
        )

    if set(observed_bounds) != set(YEARS):
        raise ValueError(
            "Missing UTC-year graph window(s): "
            f"{sorted(set(YEARS) - set(observed_bounds))}"
        )

    reconciled_rows = (
        sum(route_counts.values())
        + audit["pre_2019_utc_boundary_rows"]
        + audit["invalid_route_rows_in_window"]
    )
    if reconciled_rows != audit["input_rows"]:
        raise ValueError(
            f"Row reconciliation failed: {reconciled_rows:,} "
            f"versus {audit['input_rows']:,}"
        )

    return route_counts, observed_bounds, audit


def build_output_tables(
    route_counts: Counter,
    observed_bounds: dict,
    audit: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the edge table and one structural summary per UTC year."""
    routes_by_year: dict[int, dict[tuple[str, str], int]] = (
        defaultdict(dict)
    )
    for (year, origin, destination), count in route_counts.items():
        routes_by_year[year][(origin, destination)] = int(count)

    route_rows = []
    summary_rows = []

    for year in YEARS:
        graph = build_directed_route_graph(routes_by_year[year])
        measures = summarise_route_graph(graph)
        bounds = observed_bounds[year]

        window_start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
        window_end = pd.Timestamp(
            f"{year + 1}-01-01T00:00:00Z"
        )

        for origin, destination, data in sorted(
            graph.edges(data=True)
        ):
            route_rows.append({
                "utc_year": year,
                "window_start_utc": window_start.isoformat(),
                "window_end_utc_exclusive": window_end.isoformat(),
                "origin_airport": origin,
                "destination_airport": destination,
                "scheduled_flight_count": (
                    data["scheduled_flight_count"]
                ),
            })

        summary_rows.append({
            "utc_year": year,
            "window_start_utc": window_start.isoformat(),
            "window_end_utc_exclusive": window_end.isoformat(),
            "first_observed_departure_utc": (
                bounds["first"].isoformat()
            ),
            "last_observed_departure_utc": (
                bounds["last"].isoformat()
            ),
            "partial_year_coverage": year == 2023,
            "pre_2019_utc_boundary_rows_excluded": (
                audit["pre_2019_utc_boundary_rows"]
                if year == 2019
                else 0
            ),
            **measures,
        })

    routes = pd.DataFrame(route_rows)
    summaries = pd.DataFrame(summary_rows)

    if (
        int(routes["scheduled_flight_count"].sum())
        != int(summaries["scheduled_flight_count"].sum())
    ):
        raise ValueError(
            "Route and graph-summary flight counts differ"
        )

    return routes, summaries


def write_csv_atomically(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Replace an output only after its complete CSV is written."""
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
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Missing predictor table: {INPUT}"
        )

    counts, bounds, audit = count_yearly_routes()
    routes, summaries = build_output_tables(
        counts, bounds, audit
    )

    write_csv_atomically(routes, ROUTE_OUTPUT)
    write_csv_atomically(summaries, SUMMARY_OUTPUT)
    print("\nYearly descriptive networks created.")
    print(f"Input rows: {audit['input_rows']:,}")
    print(
        "Pre-2019 UTC boundary rows excluded: "
        f"{audit['pre_2019_utc_boundary_rows']:,}"
    )
    print(
        "Boundary source-row examples: "
        f"{audit['boundary_source_row_examples']}"
    )
    print(
        "Invalid-route rows within UTC windows: "
        f"{audit['invalid_route_rows_in_window']:,}"
    )
    print(
        f"Counted scheduled flights: "
        f"{sum(counts.values()):,}"
    )
    print(f"Yearly route rows: {len(routes):,}")
    print(summaries.to_string(index=False))
    print(f"\nRoute counts: {ROUTE_OUTPUT}")
    print(f"Graph summaries: {SUMMARY_OUTPUT}")
    print("No labels or flight outcomes were read.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)

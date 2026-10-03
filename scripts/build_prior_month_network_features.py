"""Build previous-month, schedule-only airport network features.

Each flight receives origin/destination network measures from the
previous completed UTC calendar month's directed route graph.

No labels, outcomes, current-month flights, or future-month flights
are used for that flight's graph.
"""

from collections import Counter, defaultdict
from pathlib import Path
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/features/base/base_predictors.csv"
OUTPUT = (
    ROOT / "data/features/network/"
    "prior_month_airport_network.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
EXPECTED_ROWS = 2_999_999

COLUMNS = [
    "source_row_number",
    "prediction_timestamp_utc",
    "scheduled_departure_utc",
    "origin_airport",
    "destination_airport",
]


def chunks():
    """Read schedule fields only."""
    return pd.read_csv(
        INPUT,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )


def utc_month(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(
        values, utc=True, errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("Invalid UTC timestamp.")
    return (
        timestamps.dt.year * 12 + timestamps.dt.month
    ).astype("int32")


def count_prior_route_activity() -> Counter:
    """Count sampled scheduled routes by their UTC departure month."""
    edges = Counter()
    total = 0

    for number, chunk in enumerate(chunks(), start=1):
        month = utc_month(chunk["scheduled_departure_utc"])
        origin = chunk["origin_airport"].astype("string")
        destination = chunk["destination_airport"].astype("string")

        valid = (
            origin.notna()
            & destination.notna()
            & origin.ne(destination)
        )

        grouped = (
            pd.DataFrame({
                "month": month.loc[valid],
                "origin": origin.loc[valid],
                "destination": destination.loc[valid],
            })
            .groupby(
                ["month", "origin", "destination"],
                sort=False,
            )
            .size()
        )

        for (m, o, d), count in grouped.items():
            edges[(int(m), str(o), str(d))] += int(count)

        total += len(chunk)
        print(f"Network count pass {number}: {total:,} rows")

    if total != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows; got {total:,}."
        )

    return edges


def build_monthly_metrics(edges: Counter) -> dict:
    """Build one directed graph and metric lookup per UTC month."""
    by_month = defaultdict(list)

    for (month, origin, destination), weight in edges.items():
        by_month[month].append(
            (origin, destination, weight)
        )

    metrics = {}

    for month, weighted_edges in by_month.items():
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(weighted_edges)

        metrics[month] = {
            "out_degree": dict(graph.out_degree()),
            "in_degree": dict(graph.in_degree()),
            "weighted_out_degree": dict(
                graph.out_degree(weight="weight")
            ),
            "weighted_in_degree": dict(
                graph.in_degree(weight="weight")
            ),
        }

    print(f"Past-month graphs built: {len(metrics)}")
    return metrics


def write_features(metrics: dict) -> None:
    """Look up only the graph from each prediction's prior month."""
    total = 0
    previous_id = -1
    TEMP.unlink(missing_ok=True)

    try:
        for number, chunk in enumerate(chunks(), start=1):
            chunk = chunk.reset_index(drop=True)

            ids = pd.to_numeric(
                chunk["source_row_number"],
                errors="coerce",
            )

            if (
                ids.isna().any()
                or ids.duplicated().any()
                or not ids.is_monotonic_increasing
                or int(ids.iloc[0]) <= previous_id
            ):
                raise ValueError(
                    f"Invalid source-row order in chunk {number}."
                )
            previous_id = int(ids.iloc[-1])

            prediction = pd.to_datetime(
                chunk["prediction_timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            departure = pd.to_datetime(
                chunk["scheduled_departure_utc"],
                utc=True,
                errors="coerce",
            )
            if (
                prediction.isna().any()
                or departure.isna().any()
                or not (
                    departure - prediction
                ).eq(pd.Timedelta(hours=2)).all()
            ):
                raise ValueError(
                    f"Invalid prediction horizon in chunk {number}."
                )

            history_month = utc_month(
                chunk["prediction_timestamp_utc"]
            ) - 1

            origin = (
                chunk["origin_airport"]
                .astype("string")
                .fillna("MISSING")
                .astype(str)
                .to_numpy()
            )
            destination = (
                chunk["destination_airport"]
                .astype("string")
                .fillna("MISSING")
                .astype(str)
                .to_numpy()
            )
            months = history_month.to_numpy()

            def lookup(
                metric_name: str,
                airports: np.ndarray,
            ) -> np.ndarray:
                return np.fromiter(
                    (
                        metrics.get(int(month), {})
                        .get(metric_name, {})
                        .get(airport, 0)
                        for month, airport in zip(
                            months, airports
                        )
                    ),
                    dtype=np.int32,
                    count=len(chunk),
                )

            result = pd.DataFrame({
                "source_row_number": ids.to_numpy(
                    dtype=np.int64
                ),
                "network_history_utc_month_number": months,
                "prior_month_origin_out_degree": lookup(
                    "out_degree", origin
                ),
                "prior_month_origin_weighted_out_degree": lookup(
                    "weighted_out_degree", origin
                ),
                "prior_month_destination_in_degree": lookup(
                    "in_degree", destination
                ),
                "prior_month_destination_weighted_in_degree": lookup(
                    "weighted_in_degree", destination
                ),
                "network_feature_version": (
                    "prior_month_schedule_network_v1"
                ),
            })

            result.to_csv(
                TEMP,
                mode="w" if number == 1 else "a",
                header=(number == 1),
                index=False,
            )

            total += len(result)
            print(f"Network feature pass {number}: {total:,} rows")

        if total != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS:,} rows; got {total:,}."
            )

        os.replace(TEMP, OUTPUT)

    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nPast-only network feature table created.")
    print(f"Rows: {total:,}")
    print(f"Output: {OUTPUT}")
    print("No outcomes or labels were read.")


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    edges = count_prior_route_activity()
    metrics = build_monthly_metrics(edges)
    write_features(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
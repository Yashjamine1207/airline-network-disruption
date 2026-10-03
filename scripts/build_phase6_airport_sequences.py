#!/usr/bin/env python
"""Phase 6, Step 5: build the schedule-only airport sequence inputs.

Save as:   scripts/build_phase6_airport_sequences.py
Run from the project root, after build_phase6_cohort.py and build_phase6_predictors.py:

    python scripts/build_phase6_airport_sequences.py

What it does
------------
1. Counts scheduled departures, scheduled arrivals and distinct destinations for every
   airport in every 3-hour UTC bin (see airline_disruption.sequences.airport_bins).
2. Checks the counts against an independent recount of random cells.
3. Gives every flight the index of its newest closed bin at the prediction time
   (scheduled departure UTC minus 2 hours) and checks that no bin closes after that time.
4. Reports how many flights have a full, partial or empty 48-hour lookback.
5. Measures how dense a lookback would be for an airport, a route and a carrier at an
   airport, on DEVELOPMENT rows only. This is the evidence for using the airport sequence.

Writes (large files stay out of Git; the three small tables can be committed):
    data/processed/phase6/airport_bins_v1.npz
    data/processed/phase6/phase6_sequence_index_v1.parquet
    reports/tables/phase6_sequence_availability.csv
    reports/tables/phase6_sequence_density.csv
    reports/tables/phase6_sequences_v1_manifest.json

Nothing here reads an outcome column. The final-test rows get an index like every other
row (it is built from the schedule) but nothing is measured on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.features.feature_sets import ID_COLUMN, TIMESTAMP_COLUMN
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.sequences import lookback as lb
    from airline_disruption.utils.run_info import hardware_context, package_versions
    from airline_disruption.validation.temporal_splits import SPLIT_DEVELOPMENT
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.features.feature_sets import ID_COLUMN, TIMESTAMP_COLUMN  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.sequences import lookback as lb  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402
    from airline_disruption.validation.temporal_splits import SPLIT_DEVELOPMENT  # noqa: E402

COHORT_PATH = "data/processed/phase6/phase6_cohort_v1.parquet"
PREDICTORS_PATH = "data/processed/phase6/phase6_predictors_v1.parquet"
TABLE_PATH = "data/processed/phase6/airport_bins_v1.npz"
INDEX_PATH = "data/processed/phase6/phase6_sequence_index_v1.parquet"

LEAD_SECONDS = 2 * 3600  # prediction time = scheduled departure - 2 hours
CHUNK = 250_000


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def recount_cells(table, airports, origin, destination, departure_bin, arrival_bin, arrival_ok, n_cells, seed) -> int:
    """Recount random cells with plain boolean masks. Returns the number of cells checked."""
    rng = np.random.default_rng(seed)
    busy = np.flatnonzero(table[:, :, 0].ravel() > 0)
    pick = np.concatenate([rng.choice(busy, n_cells // 2, replace=False), rng.integers(0, table.shape[0] * table.shape[1], n_cells // 2)])
    n_bins = table.shape[1]
    for flat in pick:
        a, b = divmod(int(flat), n_bins)
        leaving = (origin == a) & (departure_bin == b)
        expected = (
            float(leaving.sum()),
            float(((destination == a) & arrival_ok & (arrival_bin == b)).sum()),
            float(len(np.unique(destination[leaving]))),
        )
        got = tuple(float(v) for v in table[a, b])
        if got != expected:
            fail(f"cell (airport {airports[a]}, bin {b}) is {got}, an independent recount gives {expected}")
    return len(pick)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase 6 airport sequence inputs")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recount-cells", type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()

    for relative in (COHORT_PATH, PREDICTORS_PATH):
        if not (root / relative).exists():
            fail(f"file not found: {root / relative}  (run the Step 2 and Step 3 build scripts first)")

    # ------------------------------------------------------------------
    # Load schedule columns and the prediction times
    # ------------------------------------------------------------------
    print("Loading schedule columns ...")
    cohort = pd.read_parquet(root / COHORT_PATH, columns=[ID_COLUMN, "prediction_timestamp_utc", "phase6_split"])
    predictors = pd.read_parquet(
        root / PREDICTORS_PATH,
        columns=[ID_COLUMN, TIMESTAMP_COLUMN, ab.ORIGIN_COLUMN, ab.DESTINATION_COLUMN, ab.ELAPSED_COLUMN, "route", "carrier_identifier"],
    )
    flights = cohort.merge(predictors, on=ID_COLUMN, how="inner", validate="one_to_one")
    del predictors
    if len(flights) != len(cohort):
        fail(f"{len(cohort) - len(flights):,} cohort rows have no predictor row")
    print(f"  {len(flights):,} flights")

    departure = ab.to_epoch_seconds(flights[TIMESTAMP_COLUMN])
    prediction = ab.to_epoch_seconds(flights["prediction_timestamp_utc"])
    if not np.array_equal(departure - prediction, np.full(len(flights), LEAD_SECONDS)):
        fail("prediction_timestamp_utc is not exactly scheduled departure minus 2 hours for every row")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    airports = ab.airport_codes(flights[ab.ORIGIN_COLUMN], flights[ab.DESTINATION_COLUMN])
    grid = ab.grid_for_flights(flights)
    print(f"Building the airport table: {len(airports)} airports x {grid.n_bins:,} bins of {grid.bin_hours} h x {len(ab.CHANNELS)} channels ...")
    table, stats = ab.build_airport_bins(flights, airports, grid)
    print(f"  arrivals placed on the timeline: {stats['arrivals_counted']:,}; without a usable scheduled elapsed time: {stats['arrivals_not_placeable']:,}")

    if table[:, :, 0].sum() != len(flights):
        fail("departure counts do not add up to the number of flights")
    if table[:, :, 1].sum() != stats["arrivals_counted"]:
        fail("arrival counts do not add up to the number of placed arrivals")

    origin = ab.codes_to_index(flights[ab.ORIGIN_COLUMN], airports)
    destination = ab.codes_to_index(flights[ab.DESTINATION_COLUMN], airports)
    elapsed = flights[ab.ELAPSED_COLUMN].to_numpy(dtype="float64")
    placeable = np.isfinite(elapsed) & (elapsed > 0) & (elapsed <= ab.MAX_PLAUSIBLE_ELAPSED_MINUTES)
    arrival_bin = np.zeros(len(flights), dtype="int64")
    arrival_bin[placeable] = grid.bin_of(departure[placeable] + np.rint(elapsed[placeable] * 60).astype("int64"))
    checked = recount_cells(table, airports, origin, destination, grid.bin_of(departure), arrival_bin, placeable, args.recount_cells, seed=1)
    print(f"  independent recount of {checked} random cells: all match")
    del elapsed, arrival_bin

    # ------------------------------------------------------------------
    # Sequence index and point-in-time check
    # ------------------------------------------------------------------
    end_bin = lb.end_bins(prediction, grid)
    try:
        lb.assert_no_bin_after_prediction(end_bin, prediction, grid)
    except ValueError as problem:
        fail(str(problem))
    n_valid = lb.count_valid_bins(end_bin, grid)
    print("  point-in-time check: no sequence contains a bin that closes after its prediction time")

    index = pd.DataFrame(
        {
            ID_COLUMN: flights[ID_COLUMN].to_numpy(),
            "origin_index": origin.astype("int16"),
            "destination_index": destination.astype("int16"),
            "end_bin": end_bin.astype("int32"),
            "n_valid_bins": n_valid,
        }
    )

    # ------------------------------------------------------------------
    # Availability by split
    # ------------------------------------------------------------------
    print("Measuring lookback availability and content ...")
    origin_departures = np.empty(len(flights), dtype="float32")  # departures at the origin over the lookback
    for lo in range(0, len(flights), CHUNK):
        hi = min(lo + CHUNK, len(flights))
        values, _ = lb.gather_sequences(table, origin[lo:hi], end_bin[lo:hi], grid)
        origin_departures[lo:hi] = values[:, :, 0].sum(axis=1)
    availability = (
        pd.DataFrame(
            {
                "phase6_split": flights["phase6_split"].astype(str).to_numpy(),
                "full_lookback": n_valid == ab.LOOKBACK_BINS,
                "partial_lookback": (n_valid > 0) & (n_valid < ab.LOOKBACK_BINS),
                "empty_lookback": n_valid == 0,
                "origin_quiet": (n_valid == ab.LOOKBACK_BINS) & (origin_departures == 0),
                "origin_departures_in_lookback": origin_departures,
            }
        )
        .groupby("phase6_split")
        .agg(
            flights=("full_lookback", "size"),
            full_lookback=("full_lookback", "sum"),
            partial_lookback=("partial_lookback", "sum"),
            empty_lookback=("empty_lookback", "sum"),
            origin_quiet_full_lookback=("origin_quiet", "sum"),
            median_origin_departures_in_lookback=("origin_departures_in_lookback", "median"),
        )
        .reset_index()
    )
    availability["share_not_full"] = 1 - availability["full_lookback"] / availability["flights"]
    print(availability.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # Density: airport, route or carrier at airport? Development rows only.
    # ------------------------------------------------------------------
    print("Comparing lookback density on development rows ...")
    window = ab.LOOKBACK_BINS * grid.bin_seconds
    route_code = pd.factorize(flights["route"].astype(str))[0].astype("int64")
    carrier_code = pd.factorize(flights["carrier_identifier"].astype(str))[0].astype("int64")
    keys = {
        "origin airport": origin.astype("int64"),
        "route (origin to destination)": route_code,
        "carrier at origin airport": carrier_code * len(airports) + origin.astype("int64"),
    }
    development = (flights["phase6_split"].astype(str) == SPLIT_DEVELOPMENT).to_numpy()
    density_rows = []
    for name, key in keys.items():
        counts = lb.prior_counts(key, departure, key[development], prediction[development], window)
        density_rows.append(
            {
                "sequence_entity": name,
                "flights_measured": int(development.sum()),
                "p10_prior_flights": float(np.percentile(counts, 10)),
                "median_prior_flights": float(np.median(counts)),
                "p90_prior_flights": float(np.percentile(counts, 90)),
                "share_with_zero_prior_flights": float((counts == 0).mean()),
                "share_with_fewer_than_5": float((counts < 5).mean()),
            }
        )
    density = pd.DataFrame(density_rows)
    print(density.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  (scheduled flights of the same entity in the {ab.LOOKBACK_BINS * grid.bin_hours} hours before the prediction time)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    processed = root / "data/processed/phase6"
    tables = root / "reports/tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    ab.save_bin_table(root / TABLE_PATH, table, airports, grid)
    index.to_parquet(root / INDEX_PATH, index=False)
    availability.to_csv(tables / "phase6_sequence_availability.csv", index=False)
    density.to_csv(tables / "phase6_sequence_density.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sequence_version": ab.SEQUENCE_VERSION,
        "entity": "origin airport (destination index also stored)",
        "bin_hours": ab.BIN_HOURS,
        "lookback_bins": ab.LOOKBACK_BINS,
        "lookback_hours": ab.BIN_HOURS * ab.LOOKBACK_BINS,
        "burn_in_hours": ab.BURN_IN_HOURS,
        "first_valid_bin": grid.first_valid_bin(),
        "channels": list(ab.CHANNELS),
        "source_columns": list(ab.REQUIRED_COLUMNS),
        "prediction_lead_hours": 2,
        "grid_start_utc": pd.Timestamp(grid.start_seconds, unit="s", tz="UTC").isoformat(),
        "n_bins": grid.n_bins,
        "n_airports": int(len(airports)),
        "rows": stats["rows"],
        "arrivals_counted": stats["arrivals_counted"],
        "arrivals_not_placeable": stats["arrivals_not_placeable"],
        "table_shape": list(table.shape),
        "table_sha256": hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest(),
        "recounted_cells": checked,
        "outcome_columns_read": False,
        "final_test_used": False,
        "build_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / "phase6_sequences_v1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {root / TABLE_PATH}\n      {root / INDEX_PATH}\n      and three files in {tables}")
    print(f"Done in {time.perf_counter() - started:,.0f} s. No outcome column was read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

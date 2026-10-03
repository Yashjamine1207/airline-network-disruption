"""Phase 2: plan NOAA GHCNh weather acquisition scope from audited flight coverage."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

FLIGHT_CSV = Path("data/raw/flights_kaggle/flights_sample_3m.csv")
AIRPORT_LIST = Path("data/external/noaa_required_airports.csv")
WEATHER_PLAN = Path("docs/data/noaa_weather_acquisition_plan.md")

USECOLS = ["FL_DATE", "ORIGIN", "DEST"]
CHUNK = 250_000


def main() -> None:
    for path in (AIRPORT_LIST, WEATHER_PLAN):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not FLIGHT_CSV.exists():
        raise FileNotFoundError(f"Raw flight CSV not found: {FLIGHT_CSV}")

    print("Scanning unique airports and date range from raw flight CSV...")
    airports: set[str] = set()
    min_date, max_date = None, None

    for number, chunk_df in enumerate(pd.read_csv(FLIGHT_CSV, usecols=USECOLS, chunksize=CHUNK, low_memory=False), start=1):
        airports.update(chunk_df["ORIGIN"].dropna().unique())
        airports.update(chunk_df["DEST"].dropna().unique())
        dates = pd.to_datetime(chunk_df["FL_DATE"], format="%Y-%m-%d", errors="coerce").dropna()
        if not dates.empty:
            chunk_min, chunk_max = dates.min(), dates.max()
            min_date = min(chunk_min, min_date) if min_date else chunk_min
            max_date = max(chunk_max, max_date) if max_date else chunk_max
        print(f"Processed chunk {number:,}: {len(chunk_df):,} rows")

    if min_date is None or max_date is None:
        raise ValueError("No valid flight dates found in the raw CSV.")

    airports = sorted(airports)
    start_str = min_date.date().isoformat()
    end_str = max_date.date().isoformat()
    years = sorted(min_date.year for min_date in [min_date]) + sorted(max_date.year for max_date in [max_date])
    years = list(range(min_date.year, max_date.year + 1))

    airport_frame = pd.DataFrame({"airport_iata": airports})
    airport_frame.to_csv(AIRPORT_LIST, index=False)

    lines = [
        "# NOAA GHCNh Weather Acquisition Plan", "",
        "## Purpose", "",
        "This document defines the minimum required NOAA Global Historical Climatology Network hourly (GHCNh) weather data for this project.",
        "It uses only the audited flight airports and date coverage to avoid downloading an unnecessary global archive.",
        "", "## Flight-data coverage", "",
        f"- Date range: {start_str} to {end_str}.",
        f"- Years covered: {years[0]} to {years[-1]}.",
        f"- Unique airports (origin or destination): {len(airports):,}.",
        "", "## Required weather scope", "",
        "The project will retrieve hourly GHCNh observations for:", "",
        "- All airports listed in `data/external/noaa_required_airports.csv`.",
        f"- The full date range {start_str} to {end_str}.",
        "- Only the variables required for disruption analysis (temperature, precipitation, wind, visibility, and weather-type flags where available).",
        "", "## Retrieval constraints", "",
        "- Raw NOAA files must be stored under `data/raw/noaa_ghcnh/` and kept out of Git.",
        "- Station metadata (station ID, coordinates, observation time, quality flags) must be preserved.",
        "- Only stations needed to cover the audited airports will be downloaded.",
        "- No global NOAA archive or unrelated stations will be retrieved.",
        "", "## Next steps", "",
        "1. Identify GHCNh stations that serve each audited airport.",
        "2. Download only required station files and date ranges.",
        "3. Create an airport-to-station mapping with coverage and quality metadata.",
        "4. Implement weather matching using only observations available before the flight prediction timestamp.",
        "", "## Generated outputs", "",
        f"- Required airport list: `{AIRPORT_LIST.as_posix()}`",
        f"- This plan: `{WEATHER_PLAN.as_posix()}`",
        "",
    ]
    WEATHER_PLAN.write_text("\n".join(lines), encoding="utf-8")
    print("\nNOAA weather acquisition plan complete.")
    print(f"- Airport list: {AIRPORT_LIST}")
    print(f"- Plan document: {WEATHER_PLAN}")


if __name__ == "__main__":
    main()
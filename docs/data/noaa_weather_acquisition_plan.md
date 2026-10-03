# NOAA GHCNh Weather Acquisition Plan

## Purpose

This document defines the minimum required NOAA Global Historical Climatology Network hourly (GHCNh) weather data for this project.
It uses only the audited flight airports and date coverage to avoid downloading an unnecessary global archive.

## Flight-data coverage

- Date range: 2019-01-01 to 2023-08-31.
- Years covered: 2019 to 2023.
- Unique airports (origin or destination): 380.

## Required weather scope

The project will retrieve hourly GHCNh observations for:

- All airports listed in `data/external/noaa_required_airports.csv`.
- The full date range 2019-01-01 to 2023-08-31.
- Only the variables required for disruption analysis (temperature, precipitation, wind, visibility, and weather-type flags where available).

## Retrieval constraints

- Raw NOAA files must be stored under `data/raw/noaa_ghcnh/` and kept out of Git.
- Station metadata (station ID, coordinates, observation time, quality flags) must be preserved.
- Only stations needed to cover the audited airports will be downloaded.
- No global NOAA archive or unrelated stations will be retrieved.

## Next steps

1. Identify GHCNh stations that serve each audited airport.
2. Download only required station files and date ranges.
3. Create an airport-to-station mapping with coverage and quality metadata.
4. Implement weather matching using only observations available before the flight prediction timestamp.

## Generated outputs

- Required airport list: `data/external/noaa_required_airports.csv`
- This plan: `docs/data/noaa_weather_acquisition_plan.md`

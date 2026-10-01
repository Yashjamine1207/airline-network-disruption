# Schedule Timestamp Normalisation Benchmark

## Purpose

This benchmark measures the baseline performance of the initial row-wise
schedule timestamp-normalisation implementation before full-file processing.

The benchmark uses the immutable raw Kaggle flight CSV, the controlled
airport-to-IANA-timezone mapping, and the tested schedule timestamp utilities.

It does not create targets, features, model inputs, weather joins, or
aircraft-rotation outputs.

## Environment

- Operating system: Windows
- Python interpreter: Python 3.13.5
- Data-processing environment: Anaconda Python environment
- Timestamp implementation: pandas, Python `zoneinfo`, project schedule
  timestamp modules
- Benchmark execution date: 2026-09-20

## Input

- Raw source file: `data/raw/flights_kaggle/flights_sample_3m.csv`
- Benchmark rows: 10,000
- Airport timezone mapping:
  `data/external/airport_timezone_mapping.csv`
- Prediction lead time: 2 hours before scheduled departure
- Maximum arrival-date offset considered: 2 local calendar days
- Scheduled-duration reconciliation tolerance: 30 minutes

## Results

| Measure | Result |
|---|---:|
| Benchmark rows | 10,000 |
| Raw CSV read time | 0.04 seconds |
| Standardisation and timezone join time | 0.05 seconds |
| Schedule timestamp normalisation time | 4.52 seconds |
| Total pipeline time | 4.63 seconds |
| Approximate normalisation speed | 2,213 rows/second |
| Valid scheduled departures | 10,000 |
| Reconciled scheduled arrivals | 10,000 |
| Available prediction timestamps | 10,000 |

## Interpretation

The tested row-wise implementation is functionally correct for the benchmark
sample: all 10,000 rows produced valid scheduled departure timestamps,
reconciled scheduled arrival timestamps, and available two-hour pre-flight
prediction timestamps.

However, an approximate normalisation rate of 2,213 rows per second would
require approximately 23 minutes for 3,000,000 rows before allowing for
Parquet-writing overhead and full-file quality reporting.

The full pipeline should therefore use a vectorised or grouped-by-timezone
implementation before processing the complete source file. The optimised
implementation must preserve the same validated behaviour for:

- Source local date preservation.
- Short HHMM values.
- `2400` local-midnight rollovers.
- Origin and destination IANA timezones.
- UTC-normalised timestamps.
- Cross-timezone and overnight scheduled-arrival reconciliation.
- Two-hour pre-flight prediction timestamps.
- Explicit DST ambiguity and nonexistent-time statuses.
- Reconciliation audit fields.
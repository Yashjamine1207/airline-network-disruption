# Data Source Metadata and Scope

## Official project data boundary

**Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience** uses exactly two external data sources in its official analytical pipeline.

No FAA data, separate NOAA Storm Events data, commercial aviation data, passenger data, airline API data, airport API data, or other operational datasets will be added unless the research protocol is formally revised.

## Source 1: Flight operations data

### Dataset identity

| Field | Recorded value |
|---|---|
| Source category | Flight-level operational data |
| Dataset name | Flight Delay and Cancellation Dataset |
| Publisher or platform | Kaggle |
| Local raw file | `data/raw/flights_kaggle/flights_sample_3m.csv` |
| Supporting source dictionary | `data/raw/flights_kaggle/dictionary.html` |
| Access date | 2026-09-20 |
| Intended analytical use | Flight-level delay, cancellation, route, carrier, airport, aircraft-rotation, propagation, recovery, and network analysis |
| Expected coverage | To be confirmed from the downloaded file |
| Expected scale | Approximately 3 million flight records, subject to audit |
| Source URL | To be recorded during the Phase 1 source audit |
| Dataset author or publisher | To be recorded during the Phase 1 source audit |
| Licence or terms | To be recorded during the Phase 1 source audit |
| File hash | To be calculated during the Phase 1 source audit |
| Raw row count | To be confirmed during the Phase 1 source audit |
| Raw schema | To be confirmed from the downloaded CSV and dictionary |

### Raw-data rule

The CSV file and HTML dictionary are immutable source materials.

They must not be manually edited, cleaned, overwritten, or committed to Git.

All cleaned, standardised, joined, transformed, or feature-engineered copies must be created as new outputs in the appropriate ignored project folders.

## Source 2: Historical weather data

### Dataset identity

| Field | Recorded value |
|---|---|
| Source category | Station-level historical weather observations |
| Dataset name | NOAA Global Historical Climatology Network hourly |
| Dataset abbreviation | GHCNh |
| Publisher | National Oceanic and Atmospheric Administration |
| Local raw location | `data/raw/noaa_ghcnh/` |
| Access date | To be recorded when data is retrieved |
| Intended analytical use | Historical weather matching and weather-feature analysis for selected airports and prediction timestamps |
| Stations | To be selected after the flight-airport audit |
| Date range | To be selected after confirmation of flight-data coverage |
| Weather variables | To be selected and documented before retrieval |
| Source URL | To be recorded before retrieval |
| Licence or terms | To be recorded before retrieval |
| File hashes | To be calculated where feasible |
| Station metadata | To be retained with each weather extract |

### Weather acquisition rule

The project will retrieve only the weather stations, date ranges, and variables needed for the audited flight-data scope.

The project will not download an unnecessary global NOAA archive.

Each weather record used in analysis must retain, where available:

- Station identifier.
- Station latitude and longitude.
- UTC observation timestamp.
- Observation quality metadata.
- Airport-to-station matching method.
- Time gap between the weather observation and the flight prediction timestamp.

## Data-source documentation requirements

Before any model or final analysis uses either source, the Phase 1 data-source audit must confirm:

1. Actual file names.
2. Actual date coverage.
3. Actual row counts.
4. Actual column names and data types.
5. Field-level missingness.
6. Dataset source URL.
7. Publisher or dataset author.
8. Licence or terms of use.
9. File hashes where feasible.
10. Any differences between the downloaded files and the published dataset description.

## Scope-control rule

This two-source boundary is frozen for the official project pipeline:

```text
1. Kaggle flight delay and cancellation data
2. NOAA GHCNh historical weather observations
```

Any additional external dataset requires an explicit protocol change, a decision-log entry, and an updated source register before use.
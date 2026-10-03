# Yearly airport-route network methodology

## Scope

This is Phase 5.1 retrospective, flight-only network analysis for
*Airline Network Disruption Intelligence: Delay Prediction, Propagation,
Recovery and Network Resilience*.

These yearly graphs describe scheduled airport-route structure. They are
not pre-flight prediction features, aircraft-rotation graphs, evidence
of aircraft-level propagation, or a live airline operating system.
No NOAA weather data or flight outcomes are used in this graph build.

## Input and construction

- Input: `data/features/base/base_predictors.csv`, the Phase 3
  timestamp-eligible, label-free predictor table.
- Input rows: 2,999,999.
- Time field: `scheduled_departure_utc`.
- Graph assignment: UTC calendar year of scheduled departure.
- Nodes: origin and destination airports appearing in eligible routes.
- Directed edge: origin airport → destination airport.
- Edge weight: count of eligible scheduled flights on that directed
  route in the UTC-year window.
- Implementation:
  `src/airline_disruption/network/graph_builder.py` and
  `scripts/build_yearly_airport_networks.py`.

The build checks source-row ordering, valid UTC timestamps, and the
documented two-hour difference between scheduled departure and
prediction timestamps. Routes with missing, blank, whitespace-padded,
or identical origin and destination identifiers are excluded and
counted. No such invalid routes were found within the five UTC-year
windows in this run.

## Windows and boundary handling

The windows are half-open: start UTC is included; end UTC is excluded.

| Graph | Start UTC | End UTC, exclusive | Scheduled flights | Airports | Directed routes |
| --- | --- | --- | ---: | ---: | ---: |
| 2019 | 2019-01-01 | 2020-01-01 | 757,340 | 360 | 6,365 |
| 2020 | 2020-01-01 | 2021-01-01 | 479,564 | 367 | 6,339 |
| 2021 | 2021-01-01 | 2022-01-01 | 611,450 | 371 | 6,672 |
| 2022 | 2022-01-01 | 2023-01-01 | 687,851 | 370 | 6,642 |
| 2023 | 2023-01-01 | 2024-01-01 | 463,793 | 348 | 6,220 |

The 2023 graph is **partial-year**: its last observed scheduled
departure was 2023-09-01 08:55 UTC. The UTC-year window end shown above
defines the grouping rule; it does not claim that flights are present
through December 2023. Observed first and last departure timestamps
for every graph are stored in the summary CSV.

One valid flight, source row 120840 (SPN → GUM), has a scheduled
departure of 2018-12-31 23:15 UTC. It is outside all five UTC-year
windows and is explicitly counted as a pre-2019 boundary exclusion.
Its timestamp was not changed or treated as invalid.

Row reconciliation:

- Timestamp-eligible input rows: 2,999,999.
- Pre-2019 UTC boundary rows: 1.
- Invalid-route rows within the five windows: 0.
- Flights counted in the five graphs: 2,999,998.

## Outputs and checks

- `data/processed/network_windows/yearly_route_counts.csv`:
  32,238 year-route rows with window boundaries and scheduled-flight
  counts. This is generated data and stays out of Git.
- `reports/tables/yearly_network_summary.csv`: five rows recording
  observed timestamp bounds, coverage flags, node and edge counts,
  scheduled-flight totals, density, and connected-component counts.

A read-back validation confirmed the expected five years, positive
edge weights, no self-routes, and matching edge and graph flight-count
totals of 2,999,998.

## Interpretation limits

Graph weights count flights in this selected dataset, not all flights
operated in the aviation system. A route edge records scheduled
connectivity, not a passenger connection or aircraft movement.
Network measures describe associations and structure; they do not
establish that one airport caused disruption at another.

Do not compare the partial 2023 graph's raw flight or route totals
directly with full-year totals as though exposure time were equal.
Descriptive yearly graphs may use their full retrospectively observed
windows. They must **not** be substituted for Phase 3's past-only
network features when predicting an individual flight.
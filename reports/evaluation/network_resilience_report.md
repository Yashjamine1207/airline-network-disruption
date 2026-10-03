# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience

## Purpose and scope

This is an advanced retrospective portfolio study for Data Science, applied science, forecasting, operations research, decision science, transportation, and aviation analytics roles. It is not an airline operating platform, dispatch system, real-time disruption-control tool, or production service.

The official external-source scope is the selected Kaggle Flight Delay and Cancellation Dataset (approximately 2019–2023) and NOAA Global Historical Climatology Network hourly weather data. The Phase 5 network analyses below use the sampled flight records; they do not establish a weather effect. Verify actual downloaded coverage, licensing/terms, schemas, counts, and transformations in the project's source documentation.

## UTC-year network

The yearly directed graphs count scheduled flights on origin-to-destination routes using validated scheduled-departure UTC years. The tables describe the analysed sample, not the entire airline industry. The 2023 observation window is partial.

| UTC year | Scheduled flights | Airports | Directed routes | Density | Strong components |
|---|---:|---:|---:|---:|---:|
| 2019 | 757,340 | 360 | 6,365 | 0.0492 | 1 |
| 2020 | 479,564 | 367 | 6,339 | 0.0472 | 2 |
| 2021 | 611,450 | 371 | 6,672 | 0.0486 | 2 |
| 2022 | 687,851 | 370 | 6,642 | 0.0486 | 1 |
| 2023 (partial) | 463,793 | 348 | 6,220 | 0.0515 | 1 |

The 2019 and 2020 full-year network counts are 757,340 and 479,564: a 36.68% decline in this sample. The cancellation rate rose from 1.79% to 6.00%; the severe arrival-delay rate **among completed flights** changed from 2.66% to 1.08%. The lower completed-flight severe-delay rate in 2020 does not demonstrate better resilience: cancellations and flight composition also changed. These comparisons are descriptive, not causal.

The last observed 2023 scheduled departure is `2023-09-01T08:55:00+00:00`. Do not compare its raw flight or route total with a full year.

![Fixed-airport 2019 and 2020 network illustration](../figures/airport_network.png)

The illustration fixes 20 airports using pooled 2019–2020 traffic and displays up to 60 busy directed routes per year. It is not the complete graph.

## Airport association and recovery

The centrality-versus-disruption figure includes 449 origin airport-years meeting its minimum completed-flight count. Same-year centrality and disruption rates are associations only; they do not identify disruption transmission or causal effects.

![Airport centrality and disruption](../figures/centrality_vs_disruption.png)

Under the primary airport recovery definition—at least 20 completed flights on a qualifying airport-day, disruption rate at least 5.00%, normal rate at most 2.00%, and 2 consecutive normal days—the analysis found 2,547 episodes across 50 airports: 1,677 observed recoveries and 870 censored episodes. The reported 5-day median is conditional on observed recovery; it is **not** a Kaplan–Meier median. Definitions and sensitivity scenarios are recorded in `tables/airport_recovery_sensitivity.csv`.

## Month-to-month association

Flight outcomes were grouped by the manifest's prediction-timestamp UTC month, then paired with the preceding UTC month at the same origin airport. Each month in a retained pair has at least 50 scheduled and 50 completed flights. This is retrospective: the flight-level outcome timestamps needed to establish pre-flight availability were not checked.

| Outcome | Airport-month pairs | Airports | Pooled Pearson r | Within-airport Pearson r |
|---|---:|---:|---:|---:|
| severe | 6,263 | 159 | 0.378 | 0.256 |
| cancellation | 6,263 | 159 | 0.340 | 0.321 |

Within-airport centring removes each airport's average level, but does not remove seasonality, network-wide shocks, changing flight mix, or other confounding. Consecutive pairs can share a month. These correlations neither establish propagation nor constitute validated predictive performance.

## Interpretation boundaries

- Never use the same-year disruption table or lagged *outcome* rates as pre-flight features without separately proving that every contributing outcome was available before the precise prediction timestamp.
- Prior-calendar-month schedule and graph features in the Phase 4 schedule-network table are distinct from these retrospective outcome associations. Their production timing and feature provenance still require an explicit audit.
- The route-regime summaries and UTC-year graph summaries use different year groupings. Do not combine their flight denominators or infer a discrepancy in source completeness from that difference alone.
- Recovery results depend on the qualifying-flight minimum, disruption and normal thresholds, consecutive-day rule, and censoring. The observed-only median must not be reported as the population recovery median.

Reconstruct from the documented immutable raw flight source, timestamp mappings, transformation scripts, tests, and configuration. Keep processed flight data, feature matrices, and model artifacts out of Git.

The dedicated survival and airport/route association reports are `survival_recovery_report.md` and `propagation_report.md` in this directory.

# Airport disruption and recovery

## Scope

This is a retrospective, flight-only analysis for *Airline Network
Disruption Intelligence: Delay Prediction, Propagation, Recovery and
Network Resilience*. It studies airport-level episodes in the sampled
Kaggle flight data. It does not reconstruct aircraft rotations, measure
tail-based recovery, use weather data, or provide operational forecasts.

The existing episode file covers starts in **2019–2022**. It does not
support a 2023 recovery comparison.

## Episode definition

The primary Phase 3 episode rule uses airport-days with at least
20 completed flights in the selected data. A qualifying airport-day
with a severe-arrival-delay rate of at least 5% starts an episode.
Recovery requires two consecutive observed, qualifying days with a
severe-delay rate at or below 2%. Severe arrival delay means at least
120 minutes; these episode cutoffs are project definitions.

`duration_days` counts the start and end dates inclusively. An episode
is censored if recovery is not observed before an observation gap or
the last qualifying day. Censoring does **not** mean the airport never
recovered. The measures describe the sampled records and qualifying
days, not every flight operating at an airport.

Input:
`data/processed/survival_episodes/airport_recovery_episodes.csv`

## Episode counts

| Start cohort | Episodes | Observed recovery | Censored | Gap-censored |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 910 | 617 | 293 | 292 |
| 2020 | 202 | 145 | 57 | 57 |
| 2021–2022 | 1,435 | 915 | 520 | 504 |
| **All 2019–2022** | **2,547** | **1,677** | **870** | **853** |

The primary rule produced 36,153 qualifying airport-days. Of the
870 censored episodes, 853 ended at an observation gap and 17 at the
last qualifying day.

## Exploratory Kaplan–Meier results

Recovery is the event; a censored episode contributes follow-up until
its censoring day. The plotted survival curve represents the estimated
fraction **not yet recovered**. Estimates below are conditional on the
Kaplan–Meier censoring assumptions, which are not established here.

| Start cohort | KM median days | Estimated recovered by day 7 | Follow-up reaches day 7 | Estimated recovered by day 14 | Follow-up reaches day 14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 2019–2022 | 5 | 70.6% | 574 | 91.1% | 148 |
| 2019 | 5 | 69.1% | 224 | 90.9% | 58 |
| 2020 | 4 | 92.2% | 18 | Not estimable | 0 |
| 2021–2022 | 5 | 68.4% | 332 | 89.9% | 90 |

The 2020 day-14 estimate is deliberately withheld: no 2020 episode
has observed follow-up reaching day 14. Only 18 of its 202 episodes
reach day 7, so its day-7 estimate is also especially fragile.
Do not interpret the different cohort curves as a causal effect of
the operating regime.

The overall five-day KM median is an exploratory estimate, **not** a
verified airport recovery time. Observation gaps account for nearly
all censoring, and whether those gaps are independent of recovery is
unknown. Consequently, the probability estimates may be biased.

Outputs:
- `reports/tables/recovery_summary.csv`
- `reports/figures/kaplan_meier_recovery.png`

## Definition sensitivity

Phase 3 varied the episode rules. These counts come from separate
episode-definition scenarios; the medians in that sensitivity file
use **observed recoveries only** and are not censoring-adjusted KM
medians.

| Scenario | Episodes | Observed recovery | Censored |
| --- | ---: | ---: | ---: |
| Primary: minimum 20 flights, 5% disruption, 2% normal, two normal days | 2,547 | 1,677 | 870 |
| Minimum 10 flights | 5,859 | 3,933 | 1,926 |
| Minimum 50 flights | 643 | 304 | 339 |
| Disruption threshold 10% | 1,003 | 689 | 314 |
| Normal threshold 3% | 2,835 | 1,978 | 857 |
| One normal day | 3,205 | 2,545 | 660 |
| Three normal days | 2,087 | 1,116 | 971 |

Changing the daily minimum or the number of required normal days
substantially changes the episodes available for analysis. Do not
present any one scenario as the definitive airport recovery pattern.

Sensitivity source:
`reports/tables/airport_recovery_sensitivity.csv`

## Appropriate interpretation

These results show how a documented episode definition behaves in
the selected flight sample. They do not show aircraft-level cascades,
a complete airport operating picture, a causal recovery mechanism, or
a service-level recovery guarantee. No Cox or accelerated-failure-time
model has been fitted in this report; those models would require
separate checks of data fitness, covariates, and assumptions.
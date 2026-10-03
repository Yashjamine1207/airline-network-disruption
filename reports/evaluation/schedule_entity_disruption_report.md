# Schedule and Entity Disruption Analysis

## Scope

This Phase 2 descriptive analysis compares cancellation, diversion, mean arrival delay, and project-defined severe arrival delay across scheduled local departure hours, carriers, origin airports, and routes.

## Cohorts and restrictions

- Cancellation and diversion rates use all scheduled flights.
- Arrival-delay and severe-delay statistics use completed, non-cancelled, non-diverted flights with recorded arrival delay.
- Severe delay is project-defined as arrival delay ≥ 120 minutes.
- Entity rankings exclude groups with fewer than 1,000 scheduled flights in a regime to reduce unstable rare-group rates.
- Results are descriptive associations, not causal explanations.

## Highest severe-delay-rate entity in each eligible group

| Regime | Entity type | Entity | Scheduled flights | Severe-delay rate | Cancellation rate |
|---|---|---|---:|---:|---:|
| 2019 pre-COVID reference | carrier | B6 | 30,395 | 4.96% | 1.31% |
| 2019 pre-COVID reference | origin_airport | EWR | 13,950 | 4.99% | 2.82% |
| 2019 pre-COVID reference | route | ORD-LGA | 1,445 | 6.29% | 3.18% |
| 2020 COVID operational shock | carrier | G4 | 10,030 | 2.78% | 14.84% |
| 2020 COVID operational shock | origin_airport | SAV | 1,158 | 1.92% | 5.61% |
| 2021-2023 recovery and transition | carrier | B6 | 67,928 | 5.67% | 2.78% |
| 2021-2023 recovery and transition | origin_airport | ASE | 1,685 | 6.28% | 8.90% |
| 2021-2023 recovery and transition | route | MCO-JFK | 1,274 | 7.76% | 2.83% |

## Generated outputs

- Hour summary: `reports/tables/scheduled_departure_hour_disruption_summary.csv`
- Full entity summary: `reports/tables/entity_disruption_summary.csv`
- Top entity table: `reports/tables/top_entity_disruption_summary.csv`
- Figure: `reports/figures/scheduled_departure_hour_disruption_rates.png`

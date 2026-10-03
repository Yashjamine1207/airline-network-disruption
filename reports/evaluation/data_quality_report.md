# Data Quality and Regime Analysis Report — Phase 2 Summary

## Purpose

This report summarises Phase 2 data-quality checks, regime analysis, and limitations for the selected flight dataset.
It consolidates evidence from earlier Phase 2 scripts and documents constraints for later phases.

## Source audit

- Raw file: `data/raw/flights_kaggle/flights_sample_3m.csv`.
- Date coverage: 2019-01-01 to 2023-08-31.
- Total scheduled flights: 3,000,000.
- Unique airports: 380.
- Unique carriers: 18.
- No tail-number field present.

## Chronological validation design

- Training and development: 2019 to 2021.
- Validation and model selection: 2022.
- Final untouched test period: 2023-01-01 to 2023-08-31 (partial year).
- All evaluation uses chronological splits only; random splitting is prohibited.

## Regime definitions

- **2019 pre-COVID reference**: 757,673 scheduled flights, 1.79% cancellation rate, 2.66% severe-delay rate.
- **2020 COVID operational shock**: 479,350 scheduled flights, 6.00% cancellation rate, 1.08% severe-delay rate.
- **2021-2023 recovery and transition**: 1,762,977 scheduled flights, 2.09% cancellation rate, 2.62% severe-delay rate.

## Regime-shift diagnostics

- 2020 COVID operational shock vs 2019: scheduled_flights relative change -36.73%, Cohen's d -2.15, permutation p-value 0.00020.
- 2020 COVID operational shock vs 2019: cancellation_rate_pct relative change 231.24%, Cohen's d 0.48, permutation p-value 0.35813.
- 2020 COVID operational shock vs 2019: severe_delay_rate_pct relative change -64.51%, Cohen's d -2.85, permutation p-value 0.00005.
- 2020 COVID operational shock vs 2019: mean_arrival_delay_minutes relative change -215.30%, Cohen's d -3.05, permutation p-value 0.00005.
- 2021-2023 recovery and transition vs 2019: scheduled_flights relative change -12.74%, Cohen's d -1.36, permutation p-value 0.00030.
- 2021-2023 recovery and transition vs 2019: cancellation_rate_pct relative change 16.61%, Cohen's d 0.23, permutation p-value 0.50597.
- 2021-2023 recovery and transition vs 2019: severe_delay_rate_pct relative change -3.53%, Cohen's d -0.11, permutation p-value 0.74966.
- 2021-2023 recovery and transition vs 2019: mean_arrival_delay_minutes relative change 9.64%, Cohen's d 0.10, permutation p-value 0.76421.

## Schedule and entity disruption findings

- 2019 pre-COVID reference, carrier: B6 with severe-delay rate 4.96%, based on 30,395 scheduled flights.
- 2019 pre-COVID reference, origin_airport: EWR with severe-delay rate 4.99%, based on 13,950 scheduled flights.
- 2019 pre-COVID reference, route: ORD-LGA with severe-delay rate 6.29%, based on 1,445 scheduled flights.
- 2020 COVID operational shock, carrier: G4 with severe-delay rate 2.78%, based on 10,030 scheduled flights.
- 2020 COVID operational shock, origin_airport: SAV with severe-delay rate 1.92%, based on 1,158 scheduled flights.
- 2021-2023 recovery and transition, carrier: B6 with severe-delay rate 5.67%, based on 67,928 scheduled flights.
- 2021-2023 recovery and transition, origin_airport: ASE with severe-delay rate 6.28%, based on 1,685 scheduled flights.
- 2021-2023 recovery and transition, route: MCO-JFK with severe-delay rate 7.76%, based on 1,274 scheduled flights.

## Rotation feasibility

A formal audit in `docs/data/rotation_reconstruction_audit.md` concluded that aircraft-level rotation reconstruction is not supported by this source file.
Later phases must not claim aircraft-propagation or tail-based recovery results.

## Weather data

This version of the project does not use NOAA or any other weather dataset.
Disruption analysis is based solely on flight operations, schedule, carrier, airport, and route information.

## Consequences for later phases

- Phase 3 (targets and features): no tail-based rotation features, no weather features.
- Phase 5 (propagation and recovery): no aircraft-level propagation; only airport-route network patterns.
- Phase 6 (sequence models): no rotation-based sequences; any temporal sequences must use non-tail designs.
- All reports and the README must state these limitations explicitly.

## Generated outputs

- Data card: `docs/data/data_card.md`
- This report: `reports/evaluation/data_quality_report.md`

# Data Card — Airline Network Disruption Intelligence

## Source and coverage

- Primary source: Kaggle Flight Delay and Cancellation dataset (sample of approximately 3 million flights).
- Flight-date coverage: 2019-01-01 to 2023-08-31.
- Total scheduled flights: 3,000,000.
- Unique airports (origin or destination): 380.
- Unique carriers: 18 (from entity audit).
- No tail-number or aircraft identifier field in this source file.
- No weather data used in this version of the project.

## Operating regimes

- **2019 pre-COVID reference**: 757,673 scheduled flights, 1.79% cancellation rate, 2.66% severe-delay rate.
- **2020 COVID operational shock**: 479,350 scheduled flights, 6.00% cancellation rate, 1.08% severe-delay rate.
- **2021-2023 recovery and transition**: 1,762,977 scheduled flights, 2.09% cancellation rate, 2.62% severe-delay rate.

## Regime-shift diagnostics (monthly series)

- 2020 COVID operational shock vs 2019: scheduled_flights relative change -36.73%, Cohen's d -2.15, permutation p-value 0.00020.
- 2020 COVID operational shock vs 2019: cancellation_rate_pct relative change 231.24%, Cohen's d 0.48, permutation p-value 0.35813.
- 2020 COVID operational shock vs 2019: severe_delay_rate_pct relative change -64.51%, Cohen's d -2.85, permutation p-value 0.00005.
- 2020 COVID operational shock vs 2019: mean_arrival_delay_minutes relative change -215.30%, Cohen's d -3.05, permutation p-value 0.00005.
- 2021-2023 recovery and transition vs 2019: scheduled_flights relative change -12.74%, Cohen's d -1.36, permutation p-value 0.00030.
- 2021-2023 recovery and transition vs 2019: cancellation_rate_pct relative change 16.61%, Cohen's d 0.23, permutation p-value 0.50597.
- 2021-2023 recovery and transition vs 2019: severe_delay_rate_pct relative change -3.53%, Cohen's d -0.11, permutation p-value 0.74966.
- 2021-2023 recovery and transition vs 2019: mean_arrival_delay_minutes relative change 9.64%, Cohen's d 0.10, permutation p-value 0.76421.

## Highest severe-delay-rate entities (minimum 1,000 flights per regime)

- 2019 pre-COVID reference, carrier: B6 with severe-delay rate 4.96%, based on 30,395 scheduled flights.
- 2019 pre-COVID reference, origin_airport: EWR with severe-delay rate 4.99%, based on 13,950 scheduled flights.
- 2019 pre-COVID reference, route: ORD-LGA with severe-delay rate 6.29%, based on 1,445 scheduled flights.
- 2020 COVID operational shock, carrier: G4 with severe-delay rate 2.78%, based on 10,030 scheduled flights.
- 2020 COVID operational shock, origin_airport: SAV with severe-delay rate 1.92%, based on 1,158 scheduled flights.
- 2021-2023 recovery and transition, carrier: B6 with severe-delay rate 5.67%, based on 67,928 scheduled flights.
- 2021-2023 recovery and transition, origin_airport: ASE with severe-delay rate 6.28%, based on 1,685 scheduled flights.
- 2021-2023 recovery and transition, route: MCO-JFK with severe-delay rate 7.76%, based on 1,274 scheduled flights.

## Key limitations

- No aircraft-rotation reconstruction, propagation, cascade, or tail-based recovery analysis is possible due to the absence of a tail identifier.
- No weather features are used; disruption patterns are analysed from flight operations alone.
- The 2023 period ends on 2023-08-31 and is therefore partial-year coverage.
- All findings are descriptive associations, not causal conclusions.

## Intended use

This dataset supports flight-level severe-delay and cancellation prediction, arrival-delay regression, carrier and airport analysis, and airport-route network analysis.
It does not support aircraft-level propagation or tail-based recovery claims.

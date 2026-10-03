# Regime Shift Analysis

## Method

This report quantifies observed differences in monthly operational metrics relative to the 2019 pre-COVID reference period.

- Unit of analysis: calendar month, not individual flight.
- Comparison periods: 2020 COVID operational shock and 2021-2023 recovery/transition.
- Permutation tests: 20,000 two-sided random permutations with random seed 2026.
- Effect size: Cohen's d based on monthly values.
- These diagnostics identify distributional differences; they do not establish causes.
- The recovery/transition period includes more months than 2019 and ends in August 2023, so partial-year coverage remains a limitation.

## Diagnostics

| Comparison | Metric | 2019 monthly mean | Comparison monthly mean | Difference | Relative change | Cohen's d | Permutation p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| 2020 COVID operational shock | scheduled_flights | 63139.417 | 39945.833 | -23193.583 | -36.73% | -2.148 | 0.00020 |
| 2020 COVID operational shock | cancellation_rate_pct | 1.811 | 5.998 | 4.187 | 231.24% | 0.485 | 0.35813 |
| 2020 COVID operational shock | severe_delay_rate_pct | 2.660 | 0.944 | -1.716 | -64.51% | -2.852 | 0.00005 |
| 2020 COVID operational shock | mean_arrival_delay_minutes | 5.303 | -6.114 | -11.416 | -215.30% | -3.051 | 0.00005 |
| 2021-2023 recovery and transition | scheduled_flights | 63139.417 | 55093.031 | -8046.385 | -12.74% | -1.360 | 0.00030 |
| 2021-2023 recovery and transition | cancellation_rate_pct | 1.811 | 2.112 | 0.301 | 16.61% | 0.232 | 0.50597 |
| 2021-2023 recovery and transition | severe_delay_rate_pct | 2.660 | 2.566 | -0.094 | -3.53% | -0.109 | 0.74966 |
| 2021-2023 recovery and transition | mean_arrival_delay_minutes | 5.303 | 5.814 | 0.511 | 9.64% | 0.102 | 0.76421 |

## Interpretation rules

- A small permutation p-value means the observed difference in monthly means is unusual under the test's exchangeability assumption; it is not proof of a causal mechanism.
- Cohen's d sign shows direction relative to 2019; magnitude shows separation in units of pooled monthly standard deviation.
- The results justify retaining 2020 as a distribution-shift regime and reporting later model performance by regime.

## Outputs

- Diagnostics table: `reports/tables/regime_shift_diagnostics.csv`
- Figure: `reports/figures/regime_shift_standardised_metrics.png`

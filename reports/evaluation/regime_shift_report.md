# Regime Shift Analysis

## Scope

This report is a descriptive Phase 2 analysis of the audited raw Kaggle flight CSV. It does not provide causal claims or predictive model results.

- Source coverage analysed: 2019-01-01 to 2023-08-31.
- The 2023 period ends on 2023-08-31 and is therefore partial-year coverage.
- Severe delay is a project-defined threshold: arrival delay ≥ 120 minutes.
- Arrival-delay statistics use completed, non-cancelled, non-diverted flights with a recorded arrival delay.
- Cancellation and diversion rates use all scheduled source rows.

## Regime definitions

| Regime | Period |
|---|---|
| 2019 pre-COVID reference | 2019-01-01 to 2019-12-31 |
| 2020 COVID operational shock | 2020-01-01 to 2020-12-31 |
| 2021-2023 recovery and transition | 2021-01-01 to 2023-08-31 |

## Exact regime-level summary

| Regime | Period | Scheduled flights | Cancelled flights | Cancellation rate | Diverted flights | Diversion rate | Completed arrival-delay cohort | Mean arrival delay | Severe delays | Severe-delay rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 pre-COVID reference | 2019-01-01 to 2019-12-31 | 757,673 | 13,594 | 1.79% | 1,983 | 0.26% | 742,096 | 5.31 minutes | 19,761 | 2.66% |
| 2020 COVID operational shock | 2020-01-01 to 2020-12-31 | 479,350 | 28,757 | 6.00% | 770 | 0.16% | 449,823 | -5.01 minutes | 4,863 | 1.08% |
| 2021-2023 recovery and transition | 2021-01-01 to 2023-08-31 | 1,762,977 | 36,789 | 2.09% | 4,303 | 0.24% | 1,721,883 | 6.23 minutes | 45,177 | 2.62% |

## Interpretation guardrails

- Differences between regimes describe observed changes in this source dataset.
- The 2020 regime is retained as a distribution-shift period; it is not discarded as an outlier.
- The recovery/transition regime has more months than the 2019 and 2020 regimes, so compare rates and monthly patterns rather than raw counts alone.
- The arrival-delay distribution figure is based on a bounded analysis sample for plotting only. The monthly and regime tables use all available source rows.
- Cancellation reasons and delay-cause fields are not used in this report to explain or predict disruption outcomes.

## Generated outputs

- Monthly table: `reports/tables/monthly_operational_disruption_summary.csv`
- Regime table: `reports/tables/regime_operational_disruption_summary.csv`
- Flight-volume figure: `reports/figures/monthly_flight_volume_by_regime.png`
- Disruption-rate figure: `reports/figures/monthly_cancellation_diversion_severe_delay_rates.png`
- Arrival-delay distribution figure: `reports/figures/arrival_delay_distribution_by_regime.png`
- Distribution plotting sample rows: 60,000

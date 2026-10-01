## Audited flight-data coverage and confirmed date boundaries

The raw Kaggle flight CSV was audited before formalising the validation periods.

| Audit item | Confirmed result |
|---|---:|
| Raw flight records | 3,000,000 |
| Flight-date column | `FL_DATE` |
| Source date format | `YYYY-MM-DD` |
| Earliest date | 2019-01-01 |
| Latest date | 2023-08-31 |
| Valid parsed flight dates | 3,000,000 |
| Missing or invalid flight dates | 0 |
| Calendar years present | 2019, 2020, 2021, 2022, 2023 |

The source contains a **partial 2023 period only**, ending on 31 August 2023. Therefore, the final temporal test set is 1 January 2023 to 31 August 2023. All reports, tables, charts, and README statements must describe this as a partial-year final test period and must not imply full-year 2023 coverage.

## Confirmed chronological validation design

| Dataset role | Start date | End date | Permitted use |
|---|---|---|---|
| Development and training | 2019-01-01 | 2021-12-31 | Model development and expanding-window backtests |
| Validation | 2022-01-01 | 2022-12-31 | Model selection, hyperparameter tuning, calibration, and threshold selection |
| Final temporal test | 2023-01-01 | 2023-08-31 | One final untouched evaluation after all choices are locked |

The final test period must not influence feature selection, preprocessing, imputation, encoding, scaling, model family selection, hyperparameter tuning, probability calibration, threshold selection, or error-driven iteration.

## Operating-regime definitions

| Regime | Period | Analytical purpose |
|---|---|---|
| Pre-COVID reference | 2019-01-01 to 2019-12-31 | Establish a pre-shock operational baseline |
| COVID shock | 2020-01-01 to 2020-12-31 | Measure distribution shift and operational disruption |
| Recovery and transition | 2021-01-01 to 2023-08-31 | Examine the post-shock operating environment |

The 2020 period will not be removed simply because it is unusual. It is a central distribution-shift regime. Later experiments will compare models trained with and without 2020 when evaluating recovery-period generalisation.
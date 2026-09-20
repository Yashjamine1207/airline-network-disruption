# Validation Strategy

## Purpose

This document defines how models and analytical results will be validated.

Flight disruption data are time-dependent. Random train-test splitting would allow models to learn from future patterns and would produce overly optimistic results.

Therefore, the headline project evaluation uses chronological validation only.

## Primary validation principle

For all predictive models:

```text
Train on earlier time periods
Validate on later time periods
Test once on the final untouched time period
```

A record from the future must never be used to train, tune, calibrate, preprocess, or select features for an earlier record.

## Proposed chronology

The intended dataset period is approximately 2019 to 2023.

The proposed temporal structure is:

| Purpose | Intended period | Role |
|---|---|---|
| Development and training | 2019 to 2021 | Model fitting and expanding-window backtests |
| Validation | 2022 | Model selection, calibration, threshold selection, and ablations |
| Final temporal test | 2023 | One final untouched evaluation |

These dates are provisional.

The Phase 1 source-data audit must confirm the actual date coverage in `flights_sample_3m.csv` before the project freezes the final boundaries.

If the available coverage differs from the intended period, this document will be updated before modelling begins.

## Final-test protection rule

The final temporal test period must remain untouched until all of the following are locked:

- Target definition.
- Prediction timestamp definition.
- Feature families.
- Feature lookback windows.
- Missing-value strategy.
- Model family.
- Hyperparameters.
- Class-weighting approach.
- Calibration method.
- Classification threshold or ranked alert policy.
- Regression evaluation policy.

The final test period must not be used for:

- Exploratory feature selection.
- Hyperparameter tuning.
- Threshold optimisation.
- Calibration fitting.
- Choosing between models.
- Choosing between weather, rotation, or network feature variants.
- Selecting the preferred severe-delay threshold.
- Selecting the preferred sequence length.
- Repeated trial-and-error analysis.

## Expanding-window backtests

Within the development period, the project will use expanding-window or rolling-origin backtests where data coverage supports them.

The intended backtest structure is:

```text
Backtest 1:
Train: 2019
Validate: 2020

Backtest 2:
Train: 2019 to 2020
Validate: 2021

Backtest 3:
Train: 2019 to 2021
Validate: 2022
```

The exact dates and cut-offs will be finalised after the source-data audit.

Each backtest must preserve time order.

## Operating-regime analysis

The project treats the data as multiple operating regimes rather than one stable population:

| Regime | Intended period | Analytical purpose |
|---|---|---|
| Pre-COVID reference | 2019 | Baseline operational behaviour |
| COVID disruption shock | 2020 | Distribution shift and structural disruption |
| Recovery and transition | 2021 to 2023 | Post-shock operational behaviour |

The project will not remove 2020 merely because it is unusual.

Instead, it will evaluate whether including the 2020 shock improves or harms model generalisation to later recovery and transition periods.

## Evaluation subgroups

Where sufficient data are available, results will be reported across meaningful subgroups:

- Time period and operating regime.
- Carrier.
- Origin airport.
- Destination airport.
- Route.
- Severe-delay threshold.
- Weather-coverage availability.
- Tail-number availability.
- Rotation-eligibility status.
- Flight completion, cancellation, or diversion status where relevant.
- Airport and route frequency groups.

Subgroup reporting will describe uncertainty and sample-size limits where applicable.

## Entity-generalisation experiments

Cross-carrier and cross-airport evaluation may be conducted only as separate experiments.

Examples include:

- Holding out one or more carriers during training and evaluating on those carriers.
- Holding out one or more airports during training and evaluating on those airports.

These experiments answer whether patterns transfer across entities.

They do not replace the primary chronological evaluation because they answer a different question.

## Feature-construction rule

For each chronological fold:

- Historical rates and rolling statistics must use earlier records only.
- Imputation must be fitted on training data only.
- Encoding must be fitted on training data only.
- Scaling must be fitted on training data only.
- Feature selection must be performed using training data only.
- Resampling, if tested, must occur only within the training fold.
- Calibration must be fitted using validation data only after model training.
- Final-test data must not influence any preprocessing or model-selection decision.

## Model-comparison rule

All models compared within an experiment must use:

- The same target definition.
- The same prediction timestamp.
- The same chronological split.
- Equivalent eligible cohorts.
- Equivalent feature availability rules.
- The same primary metric.
- The same final-test protection rule.

A more complex model will be retained only if it shows meaningful incremental value in predictive performance, calibration, robustness, analytical insight, or error behaviour.

## Experiment recording

Every experiment must record:

- Experiment identifier.
- Data version.
- Raw-data file name and hash where feasible.
- Date cut-off.
- Target version.
- Feature version.
- Prediction timestamp policy version.
- Split version.
- Training and validation dates.
- Model name and model version.
- Hyperparameters.
- Random seed.
- Class weights or resampling method.
- Calibration method.
- Decision threshold or ranking capacity.
- Evaluation metrics.
- Notes on exclusions, failures, and limitations.

The project may use MLflow or a reproducible experiment-results table for this record.

## Validation limitations

Historical validation estimates how methods perform on later data within the selected dataset. It does not prove performance in live airline operations, future years, other countries, other data providers, or airlines not represented in the data.

All results will be presented as retrospective analytical evidence.
# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience

An advanced retrospective aviation Data Science portfolio study of severe flight delays, cancellations, disruption propagation through aircraft rotations, recovery behaviour, airport-route network resilience, and historical weather context.

> **Project status:** Phase 0 — research protocol and repository setup  
> **Current flight source file:** `flights_sample_3m.csv`  
> **Raw-data audit status:** Not yet completed

## Project purpose

This project investigates how severe flight disruptions arise, spread through connected parts of an airline network, and recover over time.

It combines leakage-safe machine learning, time-aware feature engineering, aircraft-rotation reconstruction, network science, survival analysis, historical weather matching, calibration, explainability, and chronological model evaluation.

The project is aimed at Data Science, applied science, forecasting, operations research, decision science, transportation, and aviation analytics roles.

## Research questions

1. Which flights are likely to experience a severe arrival delay or cancellation?
2. Which schedule, carrier, airport, route, historical, rotation, network, and weather variables are associated with disruption risk?
3. Does disruption on one eligible aircraft leg propagate to later flights in the same validated rotation?
4. How long do aircraft rotations take to recover after disruption?
5. Which airports and routes are most exposed to disruption propagation?
6. Do rotation, network, and historical NOAA weather features improve performance beyond schedule, carrier, airport, and route features?
7. Do LSTM, GRU, and compact Transformer sequence models provide meaningful value beyond strong tabular models such as LightGBM and XGBoost?

## Appropriate use

This is a **retrospective analytical portfolio project**.

It is not:

- Airline operating software
- A dispatch system
- Crew or aircraft optimisation software
- A real-time monitoring platform
- A booking or rebooking system
- A live decision engine
- A safety-critical system
- A source of operational instructions for airlines, airports, passengers, or regulators

All outputs are historical analytical evidence, not real-time operational recommendations.

See [`docs/decisions/limitations_and_appropriate_use.md`](docs/decisions/limitations_and_appropriate_use.md) for the complete statement.

## Official data sources

The official project pipeline uses exactly two external data sources:

| Source | Role | Current status |
|---|---|---|
| Kaggle Flight Delay and Cancellation Dataset | Primary flight-level operational dataset | Local CSV downloaded; source audit pending |
| NOAA Global Historical Climatology Network hourly (GHCNh) | Historical station-level weather context | To be acquired after the flight-airport audit |

The currently downloaded raw flight source is:

```text
data/raw/flights_kaggle/flights_sample_3m.csv
```

The supporting field dictionary is:

```text
data/raw/flights_kaggle/dictionary.html
```

The actual CSV schema, date coverage, row count, field definitions, missingness, source URL, publisher details, licence or terms, and file hash will be confirmed in Phase 1 before any modelling claim is made.

No FAA data, separate NOAA Storm Events data, commercial aviation data, passenger data, airport API data, airline API data, or other external datasets are included in the official project pipeline.

## Primary targets

| Analysis task | Primary definition | Notes |
|---|---|---|
| Severe arrival-delay classification | Final arrival delay greater than or equal to 120 minutes | Project-defined threshold; sensitivity analysis at 60, 90, 120, and 180 minutes |
| Cancellation classification | Source cancellation field indicates cancelled | Exact source field confirmed during data audit |
| Arrival-delay regression | Final arrival delay in minutes | Completed-flight cohort only |
| Aircraft propagation | Prior-leg disruption, delay amplification, cascade depth and duration | Eligible validated tail-number sequences only |
| Recovery analysis | First later eligible flight below 15 minutes arrival delay | Project-defined normal-operation threshold; includes censoring analysis |
| Airport/network disruption | Airport and route disruption, exposure, centrality, resilience | Descriptive and associational analysis |
| Weather-enhanced analysis | NOAA weather as an explanatory feature family | Weather is not a target |

See [`docs/decisions/target_definitions.md`](docs/decisions/target_definitions.md).

## Prediction timestamp and leakage control

The primary predictive setting is pre-flight disruption-risk prediction.

For each flight:

```text
PredictionTimestampUTC = ScheduledDepartureTimestampUTC - 2 hours
```

Only information available on or before this timestamp may be used as a predictive feature.

Pre-flight models do not use:

- Actual departure or arrival outcomes from the predicted flight
- Final departure or arrival delay
- Delay-cause fields
- Cancellation reason
- Future weather observations
- Future aircraft legs
- Future airport disruption measures
- Future network information
- Same-day aggregates that include later information

See:

- [`docs/decisions/prediction_timestamp_policy.md`](docs/decisions/prediction_timestamp_policy.md)
- [`docs/data/leakage_audit.md`](docs/data/leakage_audit.md)
- [`docs/architecture/prediction_timestamp_design.md`](docs/architecture/prediction_timestamp_design.md)

## Timezone methodology

The project preserves original local flight-time fields and reconstructs auditable timezone-aware timestamps.

UTC-normalised timestamps are used for:

- Aircraft-rotation ordering
- Cross-airport time comparisons
- Weather matching
- Prediction timestamp construction
- Network-window construction
- Propagation analysis
- Recovery analysis

The project explicitly handles date rollovers, cross-midnight flights, origin/destination timezone differences, daylight-saving transitions, invalid local clock times, cancelled flights, diverted flights, and missing actual outcomes.

See [`docs/data/timezone_methodology.md`](docs/data/timezone_methodology.md).

## Validation design

Headline results use chronological validation only.

The intended structure, pending confirmation from the raw-data audit, is:

```text
2019–2021 → Development and training
2022      → Validation, model selection, calibration, and threshold selection
2023      → Untouched final temporal test
```

The project also uses expanding-window backtests and examines three operating regimes:

```text
2019       → Pre-COVID reference
2020       → COVID disruption shock
2021–2023  → Recovery and transition
```

The final test period stays untouched until target definitions, feature families, model selection, hyperparameters, calibration, and ranking policy are locked.

See [`docs/decisions/validation_strategy.md`](docs/decisions/validation_strategy.md).

## Modelling approach

The project compares models in this order:

1. Majority-class, historical-rate, global-mean, global-median, and route-median baselines
2. Logistic Regression for classification
3. Linear Regression, Ridge, and Elastic Net for arrival-delay regression
4. LightGBM and XGBoost tabular models
5. TensorFlow/Keras LSTM and GRU sequence challengers
6. A compact TensorFlow/Keras Transformer challenger

Deep-learning models are ablations, not a requirement. A simpler model is a strong final result if it performs better on predictive performance, calibration, robustness, or interpretability.

## Evaluation metrics

### Classification

Primary metric:

```text
PR-AUC
```

Supporting metrics:

- Precision
- Recall
- F1 score
- Precision@K
- Recall@K
- ROC-AUC
- Brier score
- Calibration curves
- Reliability diagrams
- Threshold confusion matrices

### Regression

Primary metric:

```text
MAE in minutes
```

Supporting metrics:

- RMSE
- Median absolute error
- Bias
- Residual distribution
- Upper-tail error

### Recovery and network analysis

- Kaplan-Meier recovery curves
- Median recovery time
- Concordance index where suitable
- Censoring rates
- Propagation probability
- Delay amplification
- Cascade depth and duration
- Airport and route exposure
- Network centrality and resilience measures

## Repository structure

```text
airline-network-disruption-intelligence/
├── configs/                  # Versioned project, data, feature, model, and policy configuration
├── data/
│   ├── external/             # Small mappings, source register, and field documentation
│   ├── raw/                  # Immutable downloaded source files; excluded from Git
│   ├── interim/              # Standardised tables; excluded from Git
│   ├── processed/            # Analytical tables; excluded from Git
│   ├── features/             # Model matrices and sequences; excluded from Git
│   └── sample/               # Small safe or synthetic examples only
├── docs/
│   ├── architecture/         # Analytical, lineage, and timestamp diagrams
│   ├── data/                 # Data cards, audits, mappings, and leakage documentation
│   ├── decisions/            # Research protocol and decision records
│   └── experiments/          # Experiment plans and results
├── models/                   # Metrics and registry documentation; artefacts excluded from Git
├── notebooks/                # Exploratory analysis and visual development
├── reports/                  # Final figures, tables, and evaluation reports
├── scripts/                  # Reproducible command-line workflow scripts
├── src/airline_disruption/   # Reusable project Python package
└── tests/                    # Unit and data-validation tests
```

## Setup

### 1. Create a virtual environment

In the VS Code terminal:

```powershell
py -3.11 -m venv .venv
```

### 2. Activate it in PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the project and development tools

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Install pre-commit hooks

```powershell
pre-commit install
```

Do not install or run the modelling stack until the Phase 0 repository files have been reviewed and committed.

## Git and data safety

The following are intentionally excluded from Git:

- Raw Kaggle flight files
- Raw NOAA weather files
- Parquet outputs
- DuckDB databases
- Processed analytical tables
- Feature matrices
- Model artefacts
- TensorFlow checkpoints
- MLflow outputs
- Local environments
- Credentials and `.env` files
- Logs and generated outputs

Only source code, tests, configurations, documentation, small permitted samples, curated figures, and small final result tables should be committed.

## Project documentation

| Area | Key document |
|---|---|
| Appropriate use and limitations | [`docs/decisions/limitations_and_appropriate_use.md`](docs/decisions/limitations_and_appropriate_use.md) |
| Target definitions | [`docs/decisions/target_definitions.md`](docs/decisions/target_definitions.md) |
| Prediction timestamp | [`docs/decisions/prediction_timestamp_policy.md`](docs/decisions/prediction_timestamp_policy.md) |
| Validation strategy | [`docs/decisions/validation_strategy.md`](docs/decisions/validation_strategy.md) |
| Metric definitions | [`docs/decisions/metric_definitions.md`](docs/decisions/metric_definitions.md) |
| Recovery definition | [`docs/decisions/recovery_definition.md`](docs/decisions/recovery_definition.md) |
| Data-source metadata | [`docs/data/source_metadata.md`](docs/data/source_metadata.md) |
| Timezone methodology | [`docs/data/timezone_methodology.md`](docs/data/timezone_methodology.md) |
| Leakage audit | [`docs/data/leakage_audit.md`](docs/data/leakage_audit.md) |
| Decision log | [`docs/decisions/decision_log.md`](docs/decisions/decision_log.md) |
| Analytical architecture | [`docs/architecture/analytical_architecture.md`](docs/architecture/analytical_architecture.md) |
| Data lineage | [`docs/architecture/data_lineage.md`](docs/architecture/data_lineage.md) |
| Prediction timestamp design | [`docs/architecture/prediction_timestamp_design.md`](docs/architecture/prediction_timestamp_design.md) |

## Current next milestone

**Phase 1 — Data acquisition, source audit, ingestion, and timestamp normalisation**

Before feature engineering or modelling, the project will audit the downloaded CSV and dictionary to establish the actual schema, date coverage, row count, data types, missingness, cancellation/diversion fields, tail-number coverage, carrier identifiers, and timestamp fields.
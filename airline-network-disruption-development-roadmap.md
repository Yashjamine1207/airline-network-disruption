# Airline Network Disruption Intelligence — Development Roadmap

## Purpose

Build an advanced, reproducible Data Science investigation of flight delay, cancellation, disruption propagation, recovery, and network resilience using flight-level operational data and historical weather observations.

**Project name:** Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience

**Positioning:** Advanced Applied Data Science, Aviation Analytics, Machine Learning, Time-Series and Sequence Modelling, Network Science, Statistical Analysis.

**Audience:** Data Science recruiters, hiring managers, applied scientists, technical interviewers, aviation analytics teams, and operations research/decision science teams.

**Duration:** 10–12 weeks at approximately 10–15 hours per week.

## Final claim

> This project predicts severe airline disruptions from flight-level operational data and investigates how delays propagate through aircraft rotations and airport networks, combining leakage-safe machine learning, temporal sequence modelling, survival analysis and network science to quantify disruption risk, recovery and resilience.

This is an analytical portfolio project, not airline production software, a dispatch system, a crew/aircraft optimiser, or a real-time operational platform.

## Core questions

1. Which flights are likely to experience severe arrival delay or cancellation?
2. Which operational variables are associated with disruption risk?
3. Does disruption on one aircraft leg propagate to later flights in its rotation?
4. How quickly do aircraft rotations, airports, and networks recover after disruption?
5. Which airports/routes are most exposed to disruption propagation?
6. Do weather, aircraft-rotation, and network features add measurable value beyond schedule, carrier, route, and historical features?
7. Do LSTM, GRU, or Transformer sequence models provide meaningful improvement over strong tabular models?

## Scope and stopping point

### In scope

- Flight-level severe-delay and cancellation prediction.
- Arrival-delay regression for completed flights.
- Aircraft-rotation reconstruction where tail identifiers and time ordering are sufficiently reliable.
- Propagation, cascade, and recovery analysis.
- Airport and route network analysis using NetworkX.
- Historical weather feature matching using the selected NOAA GHCNh source.
- Statistical and event-based analysis of weather shocks.
- Classically statistical, boosting, and TensorFlow/Keras sequence models.
- Uncertainty/calibration, explainability, ablations, error analysis, and scenario-based operational prioritisation.

### Optional extensions

- HDBSCAN disruption regimes.
- Dynamic daily/weekly airport networks.
- Cross-airline and cross-airport generalisation.
- Airport-level next-day disruption forecasting.
- Difference-in-differences only if a credible treatment/control construction and its assumptions can be documented.

### Explicitly out of scope

- FastAPI, REST APIs, production dashboards, airline software integration, real-time scoring, real-time crew or aircraft dispatch, booking/rebooking, or a full digital twin.
- Kafka, streaming, queues, microservices, Kubernetes, Docker deployment, AWS/Azure/GCP, Terraform, production monitoring, or complex CI/CD.
- Graph neural networks as production architecture.
- LLMs, RAG, agents, or chatbots.
- Passenger-level data or passenger identification.

## Data strategy

### External sources

Use exactly two official project data sources:

1. **Flight Delay and Cancellation Dataset (2019–2023), Kaggle** — primary operational flight dataset. The exact file coverage, fields, licence, and access date must be confirmed and recorded after download.
2. **NOAA Global Historical Climatology Network hourly (GHCNh)** — secondary station-level weather dataset, restricted to needed stations, times, and fields.

Do not add FAA, Storm Events, or other external datasets to the official pipeline. They may be noted only as future possibilities.

### Working period

Use only dates actually present in the downloaded flight dataset. The intended period is approximately 2019–2023. After audit, define the real chronological boundaries.

Recommended structure:

```text
2019–2021 → development/training
2022      → validation and model selection
2023      → final temporal test
```

If the available 2023 coverage is partial, use the final available segment rather than claiming full-year coverage.

### Regime analysis

Treat the period as a sequence of operating regimes rather than one stationary population:

- 2019: pre-COVID reference regime.
- 2020: major operational shock.
- 2021–2023: recovery/transition regime.

Do not discard 2020 automatically as an outlier. Explicitly test whether model performance, delay distributions, network structure, feature distributions, and recovery behaviour change across regimes.

### Data handling rules

- Retain raw downloads immutable under `data/raw/`; do not commit them to Git.
- Convert working tables to Parquet and query large flight tables using DuckDB/Polars/Pandas as appropriate.
- Retrieve only weather stations relevant to flight airports, the selected date window, required observation windows, and selected variables.
- Preserve station metadata, timestamps, quality fields, airport/station matching method, and weather observation availability time.
- Document the Kaggle dataset licence/terms and NOAA attribution/use requirements before public release.

## Targets

### Target 1 — Severe arrival delay

Primary rare-event classification target:

```text
SevereDelay = 1 if arrival delay ≥ 120 minutes
SevereDelay = 0 otherwise
```

The 120-minute threshold is a project-defined analytical threshold, not an official dataset classification. Conduct sensitivity analysis at 60, 90, 120, and 180 minutes.

### Target 2 — Cancellation

```text
Cancelled = 1 if cancelled
Cancelled = 0 otherwise
```

Calculate event prevalence from the selected data; do not import an assumed rate.

### Target 3 — Arrival-delay regression

For completed flights with an available outcome:

```text
Target = arrival delay in minutes
```

### Target 4 — Aircraft-level propagation

For reliably reconstructed aircraft sequences, study previous-leg delay, current-flight disruption, delay amplification, cascade depth, cumulative rotation delay, subsequent severe-delay risk, and subsequent cancellation risk.

### Target 5 — Recovery

Create operational disruption episodes and model time to recovery. A candidate definition is recovery when a later operated flight returns below 15 minutes of arrival delay. This is a project definition; evaluate sensitivity to alternatives.

### Target 6 — Airport/network disruption

Derive airport/route disruption measures from the flight data: delay/cancellation pressure, route disruption, centrality, network exposure, downstream impact, and network recovery.

### Target 7 — Weather-enhanced disruption

Use weather as an external explanatory feature family, not a target. Test whether origin/destination weather improves delay, cancellation, propagation, recovery, or airport-disruption analysis.

## Non-negotiable leakage rules

For each prediction, define a concrete **prediction timestamp**. Only variables that were available by that timestamp may be used.

Never use:

- actual arrival/departure information from the predicted flight;
- final arrival delay, cancellation cause, delay-cause fields, or other post-outcome attributes in pre-flight models;
- future flights, future aircraft legs, future airport congestion, future network values, or future weather;
- daily aggregates that accidentally include later flights on the same day;
- target encodings, statistics, scaling, imputation, calibration, feature selection, or hyperparameter tuning fitted on validation/final-test observations;
- future-period network graphs to generate current-period features.

For a flight in an aircraft rotation, previous-leg actual arrival data may be used only if that leg had already arrived before the current-flight prediction timestamp. The rules must be implemented and unit-tested, not merely described.

Delay-cause fields may be used only in retrospective attribution analysis, never as predictors in a model that claims pre-outcome disruption prediction.

## Validation design

### Chronological holdout

After the data audit, freeze the final date boundaries. Use the proposed 2019–2021 training, 2022 validation, 2023 final-test structure only if confirmed by actual coverage.

The final period is used once after selecting features, model family, hyperparameters, calibration, and alert threshold.

### Rolling temporal backtests

Use expanding windows, for example:

```text
Train 2019          → validate 2020
Train 2019–2020     → validate 2021
Train 2019–2021     → validate 2022
```

Evaluate models by regime, time, carrier, airport, route, event severity, and missingness/sequence-availability subgroup.

### Entity generalisation

Run separate experiments when adequate data permit:

- Hold out carrier(s): does the model learn general disruption patterns or carrier-specific behaviour?
- Hold out airport(s): does the model generalise to unseen airports?

Clearly distinguish entity generalisation from future-time generalisation.

## Metrics

### Severe delay and cancellation

Use PR-AUC as the headline metric because events are rare. Also report:

- Precision, recall, F1.
- Precision@K and Recall@K at realistic analyst capacities such as top 0.5%, 1%, 5%, and 10% of the ranking.
- ROC-AUC as secondary context only.
- Brier score, reliability diagrams, and calibration curves.
- Confusion matrices at documented thresholds.
- Event rate and performance by operating regime.

### Regression

Use MAE as the headline regression metric, reported in minutes. Also report:

- RMSE.
- Median absolute error.
- Bias/residual distribution.
- Upper-tail errors and performance conditional on disruption severity.
- Quantile loss if probabilistic/quantile regression is implemented.

### Recovery/survival

Report:

- Kaplan–Meier curves and median recovery time.
- Concordance index.
- Survival calibration and integrated Brier score where implemented appropriately.
- Censoring treatment and episode-definition sensitivity.

### Network/propagation

Report descriptive and comparative quantities, not a false single “network accuracy” metric:

- Propagation probability.
- Delay amplification.
- Cascade depth and duration.
- Downstream delay/cancellation impact.
- Airport/route exposure.
- Centrality distributions and centrality/disruption relationships.
- Network recovery/change over operating regimes.

## Project phases

## Phase 0 — Research protocol and repository setup

**Goal:** Define the research claims, data boundary, prediction timestamp, and reproducible repository before modelling.

**Tasks**

- Create `airline-network-disruption-intelligence` with the approved modular/research-first structure.
- Write data-source, target, leakage, timezone, validation, metric, recovery, and operational-value decision documents.
- Create architecture, data-lineage, and prediction-timestamp diagrams.
- Define the project’s appropriate-use statement: analytical portfolio research, not airline operational software.
- Create versioned configuration files for source data, time zones, targets, features, validation, models, calibration, and policy scenarios.

**Completion check**

- The two-source data scope is frozen.
- Every primary target has a candidate definition and a sensitivity plan.
- No raw data, credentials, or large model artefacts are tracked by Git.

## Phase 1 — Acquisition, ingestion, and timestamp normalisation

**Goal:** Produce reproducible, audit-ready operational and weather tables.

**Tasks**

- Download flight data; record source page, access date, actual coverage, schema, file hashes where feasible, licence/terms, and raw row counts.
- Acquire only required GHCNh station/data subsets; retain station IDs, coordinates, UTC timestamps, quality data, and retrieval metadata.
- Build/verify airport identifier, coordinate, and timezone mapping. Use Python `zoneinfo` or equivalent.
- Convert scheduled/actual local flight timestamps to an auditable consistent representation, normally UTC plus retained local timestamps/timezone.
- Correctly handle midnight crossings, date rollovers, time-zone changes, daylight-saving transitions, cancelled/diverted observations, and missing timestamps.
- Convert clean source tables to Parquet.
- Validate uniqueness, schema, source coverage, and field-level missingness.

**Completion check**

- A time-ordered flight table can be rebuilt from raw source files.
- Timestamp conversions are tested on normal, cross-midnight, and daylight-saving cases.
- The project does not assume all tail IDs or actual times are available.

## Phase 2 — Data quality, EDA, and regime analysis

**Goal:** Establish data fitness and understand disruption behaviour before constructing models.

**Tasks**

- Audit duplicates, missingness, extreme values, cancellation/diversion handling, tail-ID completeness, carrier-ID consistency, and route/airport coverage.
- Use the appropriate unique carrier identifier for longitudinal analysis when source fields support it; document carrier-code changes/mergers/joint reporting issues.
- Analyse flight volume, delay distributions, cancellation rates, airport/carrier/route variation, heavy tails, seasonality, hour-of-day, and route-frequency distribution.
- Compare 2019, 2020, and 2021–2023 patterns: activity, delays, cancellations, routes, network structure, rotations, and weather associations.
- Investigate structural breaks/distribution shift using visual and reproducible statistical diagnostics.
- Audit weather coverage, airport/station matching quality, time distance to matched observations, and missingness.

**Completion check**

- Data-card limitations and meaningful regime boundaries are documented.
- A reviewer can see how much rotation and weather analysis is supported by actual data availability.

## Phase 3 — Target construction and leakage-safe features

**Goal:** Construct reliable labels and time-safe tabular/sequence feature sets.

**Tasks**

- Implement severe-delay thresholds and cancellation target.
- Build completed-flight arrival-delay regression target.
- Reconstruct eligible aircraft rotations using tail ID and UTC-normalised ordering; produce a rotation-quality/audit table and exclude ambiguous sequences from propagation claims.
- Create event episodes and recovery-time/censoring data.
- Build schedule, carrier, route, airport, historical, rolling, temporal, rotation, weather, and network feature families.
- Ensure historical route/carrier/airport rates are calculated from past-only rows within each validation fold.
- Build dynamic/rolling network features from past-only windows.
- Define weather matching: airport, eligible station, nearest valid observation before prediction timestamp or documented preceding time window.
- Write the feature catalogue, target definitions, and leakage audit.

**Completion check**

- Every feature has source field(s), formula, availability time, lookback window, missing-data rule, and code version.
- Unit tests prove no future flight, weather observation, or future graph state can enter a prediction row.

## Phase 4 — Baselines, statistical analysis, and strong tabular models

**Goal:** Establish whether complex methods add value over transparent comparators.

**Tasks**

- Implement majority/historical-rate baselines for classification and global/route historical median baselines for regression.
- Train Logistic Regression for severe delay/cancellation; Linear/Ridge/Elastic Net for delay regression.
- Train XGBoost and LightGBM on equivalent temporal folds.
- Compare base schedule/route/carrier features with additions of historical, rotation, network, and weather features.
- Use class weighting and threshold optimisation; do not apply naïve SMOTE across temporal data.
- Track data version, feature version, split, parameters, random seed, and metrics using MLflow or a reproducible experiment table.

**Completion check**

- The project contains a fair baseline-to-boosting comparison across all main targets.
- Accuracy is explicitly not the headline metric for rare disruption events.

## Phase 5 — Propagation, network, recovery, and weather-event analysis

**Goal:** Turn flight-level modelling into a disruption-system analysis.

**Tasks**

- Quantify prior-leg delay to current delay relationships, delay amplification, severe-delay propagation probability, cascade depth, cascade duration, and cumulative rotation delay.
- Analyse airport congestion/traffic-pressure relationships using cautious statistical methods; describe association, not causation.
- Build airport-route graphs with nodes as airports, edges as routes, and documented weights/time windows.
- Calculate degree/weighted degree, betweenness, closeness, PageRank, route diversity/concentration, density, exposure, and downstream disruption indicators.
- Compare network structure during normal and high-disruption periods.
- Build recovery episodes; estimate Kaplan–Meier curves and Cox/AFT models if model assumptions and sample size are adequate.
- Define weather shocks from observed weather distributions or documented domain-informed thresholds.
- Compare before/during/after weather-shock windows and consider a cautious event-study design.

**Completion check**

- Aircraft propagation results are limited to eligible/reliable reconstructed rotations.
- Network and event-analysis conclusions explicitly state their descriptive/associational limitations.

## Phase 6 — TensorFlow sequence models and ablations

**Goal:** Test whether sequence modelling genuinely improves disruption prediction.

**Tasks**

- Develop TensorFlow/Keras LSTM and GRU models with documented sequence design and lookback tests.
- Use sequences based on only past operational observations, e.g., eligible previous aircraft legs or documented operational time steps.
- Build one compact Transformer encoder challenger only after LSTM/GRU are stable.
- Use comparable temporal splits, event prevalence, metrics, calibration, and resource reporting across tabular and sequence models.
- Run ablations: base; base + historical; base + rotation; base + network; base + weather.
- Run deep-learning ablation: LightGBM; LSTM; LSTM + rotation sequence; GRU; Transformer.
- Run regime ablation: train with/without 2020 and evaluate later recovery periods.

**Completion check**

- Each advanced method has a retained/rejected decision justified by incremental value, robustness, calibration, and complexity.
- It is explicitly acceptable for LightGBM, logistic regression, or simpler models to win.

## Phase 7A — Calibration, explainability, and error analysis

**Goal:** Make predictions understandable and expose model weaknesses.

**Tasks**

- Compare uncalibrated and validation-only calibrated probabilities using Platt/sigmoid and isotonic methods where appropriate.
- Generate reliability plots, Brier scores, and calibration comparisons across operating regimes.
- Generate SHAP global/local explanations for the selected tabular model.
- Use temporal ablation, masking, permutation removal, or feature-family ablation for sequence models; do not treat attention weights as causal explanations.
- Analyse false positives, false negatives, high-confidence errors, upper-tail regression errors, and failure by carrier/airport/weather/regime.
- Include representative successful alerts, missed severe events, false alarms, and recovered/non-recovered rotations.

## Phase 7B — Operational prioritisation and final portfolio story

**Goal:** Evaluate practical usefulness without claiming real airline costs or operational authority.

**Tasks**

- Evaluate ranked analyst-review scenarios at top 0.5%, 1%, 5%, and 10% of flights.
- Report precision@K, recall@K, severe disruptions captured, cancellation events captured, and false alerts.
- Compare thresholds under scenario-based penalties for missed severe disruption versus unnecessary attention.
- Label all cost/utility values as illustrative analytical assumptions, not airline financial estimates.
- Write the final technical report, model-selection report, operational-value report, README, and limitations statement.
- Optionally create a read-only Streamlit or Power BI presentation only after reports are complete.

**Completion check**

- A reviewer can identify the selected model, prediction-time information, calibration, value of rotation/network/weather features, failure modes, recovery findings, and limitations without running a live system.

## Final deliverables

- Reproducible repository and data-download instructions.
- Data card, data dictionary, source register, timezone methodology, rotation-reconstruction audit, weather-matching audit, leakage audit, and feature catalogue.
- Forecasting/classification/regression/survival/network evaluation reports.
- Model-comparison, calibration, ablation, propagation, network, and recovery tables.
- Final technical report and recruiter-focused README.
- Optional read-only demo/dashboard using safe prepared outputs only.

## Final narrative

```text
Flight and weather data
→ source/timestamp/data-quality audit
→ regime-shift analysis
→ leakage-safe targets and features
→ baseline, statistical, and boosting models
→ aircraft rotations and propagation
→ dynamic network analysis
→ survival/recovery analysis
→ TensorFlow LSTM/GRU/Transformer ablations
→ weather-shock analysis
→ calibration, SHAP, and error analysis
→ ranked operational-prioritisation scenarios
→ final evidence-led model-selection decision
```

The project ends after the final report. Complexity is included only where the evidence shows it improves prediction, propagation understanding, recovery analysis, calibration, resilience insight, or practical prioritisation.
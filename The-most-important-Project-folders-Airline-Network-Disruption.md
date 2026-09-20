# Most important project folders

The project should be judged by temporal correctness, source provenance, credible propagation/recovery analysis, and evidence-led model comparison—not by the number of frameworks or infrastructure folders.

## `docs/data/`

This is the foundation of the project. Aviation data can become invalid for temporal analysis if local timestamps, carrier IDs, tail sequences, or weather joins are treated carelessly.

Prioritise:

- `timezone_methodology.md` — timestamp definitions, airport time zones, UTC conversion, midnight crossings, daylight-saving transitions, and uncertainty/limitations.
- `leakage_audit.md` — exact prediction timestamp, allowed/forbidden fields, post-outcome delay-cause exclusions, prior-leg availability, weather cutoff, and past-only network construction.
- `rotation_reconstruction_audit.md` — tail-ID completeness, eligible sequence logic, ambiguous/missing rotations, turnaround treatment, and excluded records.
- `airport_station_matching_audit.md` — station-selection method, coverage, matching quality, time gaps, weather quality flags, and unmatched airports.
- `carrier_identifier_policy.md` — the longitudinal carrier field policy and reporting-code caveats.
- `regime_shift_analysis.md` — evidence and treatment of 2019, 2020, and 2021–2023 as different operating regimes.
- `data_card.md`, `data_dictionary.md`, and `source_metadata.md` — provenance, exact field definitions, data access/usage notes, and limitations.

These documents are more important than a Transformer. Without them, claims about propagation or pre-flight prediction may be invalid.

## `src/airline_disruption/timestamps/`

Timezone handling is a core analytical module, not a minor preprocessing detail.

Prioritise:

- Airport-to-timezone mapping with a versioned source/method.
- Construction of scheduled and actual timestamp fields while preserving original local-time fields.
- UTC normalisation for sequencing and time comparison.
- Cross-midnight and daylight-saving tests.
- A clear availability timestamp for each feature/prediction task.

Do not reconstruct aircraft rotations or align weather to flights until this module has passed tests and an audit.

## `src/airline_disruption/data/`, `weather/`, and `rotation/`

These three areas create the trustworthy analytical dataset.

### Data

- Immutable raw ingestion and schema validation.
- Flight identifiers, date/carrier/airport standardisation, and Parquet output.
- Source and coverage checks.

### Weather

- Relevant GHCNh station selection.
- Station metadata/quality preservation.
- Nearest valid **pre-prediction-time** observation matching.
- Missing-weather handling and match-quality reporting.
- Clearly documented derived weather indicators.

### Rotation

- Tail/date/time sorting based on normalised timestamps.
- Eligibility criteria for prior-leg links.
- Turnaround calculations.
- Delay/cancellation propagation labels.
- Episode/cascade/recovery construction.

A weak aircraft rotation can create false propagation. A weak weather join can create leakage. Build and audit these before model training.

## `src/airline_disruption/features/` and `targets/`

This is where the scientific design becomes executable.

Prioritise:

- Schedule/calendar/route/carrier/airport features.
- Past-only route, carrier, and airport historical statistics.
- Tail/rotation features only where a prior leg was genuinely available at the prediction timestamp.
- Past-window network features, never graph metrics computed from the future test period.
- Weather features using only eligible historical observations.
- Severe delay/cancellation/regression target builders.
- Recovery/episode definitions with censoring logic.
- Feature/target tests and documentation.

The feature catalogue must state formula, source, lookback, time availability, missingness rule, and version for every feature.

## `src/airline_disruption/validation/`

This folder protects the central claim that results are out-of-sample and time-valid.

Prioritise:

- 2019–2021/2022/2023 split implementation based on actual audited coverage.
- Expanding-window/rolling-origin folds.
- Horizon/availability embargo logic where needed.
- Carrier and airport holdout experiments.
- Shared folds across baseline, boosting, and sequence-model experiments.
- Regime-specific evaluation slices.

Do not permit random splitting for the main project claims.

## `src/airline_disruption/models/` and `evaluation/`

Use models to answer research questions, not to collect methods.

### Models

Build in this order:

1. Historical/majority/median baselines.
2. Logistic Regression; Linear/Ridge/Elastic Net.
3. LightGBM and XGBoost.
4. TensorFlow LSTM and GRU.
5. Compact Transformer challenger.
6. Validation-only calibration.

### Evaluation

Make this folder the source of reproducible evidence:

- PR-AUC, precision/recall/F1, Recall@K, Precision@K, Brier score, and calibration for rare classification targets.
- MAE, RMSE, median absolute error, bias, residual, and tail-error analysis for delay regression.
- Kaplan–Meier/Cox/AFT diagnostics and recovery metrics.
- Propagation probability, delay amplification, cascade depth/duration, and network exposure.
- Regime, carrier, airport, weather, and missingness subgroup analysis.
- Scenario-based operational-prioritisation tables with clearly illustrative assumptions.

## `src/airline_disruption/network/` and `survival/`

These folders distinguish the project from a standard flight-delay classifier.

### Network

- Build graphs from documented, past-only date windows.
- Separate static descriptive network analysis from predictive network features.
- Compute centrality, connectivity, route concentration/diversity, exposure, and resilience measures.
- Assess normal versus disrupted network periods.

### Survival

- Create valid disruption episodes and time-to-recovery/censoring fields.
- Compare Kaplan–Meier curves across meaningful groups.
- Use Cox/AFT only when diagnostics and assumptions are addressed.
- Present survival results as associations and recovery patterns, not operational guarantees.

## `configs/` and `docs/decisions/`

Keep critical assumptions outside notebooks.

Highest-value configuration/decision assets are:

- `airports_timezones.yaml` and `timezone_methodology.md`.
- `weather_matching.yaml` and the weather-matching audit.
- `targets.yaml` and `target_definitions.md`.
- `rotation_reconstruction.yaml` and reconstruction audit.
- `network_features.yaml` and network-analysis plan.
- `validation.yaml` and prediction-timestamp policy.
- `calibration.yaml`, `decision_policy.yaml`, `alert_threshold_policy.md`, and `utility_scenarios.md`.
- `limitations_and_appropriate_use.md`.

## `reports/evaluation/`

The final evidence should be reviewable without rerunning 30 million records.

Prioritise:

- `timezone_and_sequence_report.md`
- `regime_shift_report.md`
- `disruption_prediction_report.md`
- `delay_regression_report.md`
- `propagation_report.md`
- `network_resilience_report.md`
- `survival_recovery_report.md`
- `weather_event_analysis_report.md`
- `calibration_and_alert_policy_report.md`
- `explainability_report.md`
- `error_analysis_report.md`
- `operational_value_report.md`
- `final_model_selection_report.md`
- `final_technical_report.md`

The final model-selection report must identify selected target(s), prediction timestamp, data cut-off, eligible cohort, features, validation design, model/calibration/threshold, test results, ablations, errors, limitations, and reasons advanced models were retained or rejected.

## `tests/`

The highest-value tests are analytical correctness tests:

- Flight uniqueness/schema and required source-field checks.
- Local-time/UTC, midnight, and DST conversion tests.
- Target construction and cancellation cohort tests.
- Tail ordering and rotation eligibility tests.
- No-future-data tests for historical/rolling/rotation/weather/network features.
- Temporal split, embargo, carrier holdout, and airport holdout tests.
- Classification/regression/survival/calibration/policy metric tests.

## Lower-priority folders

- `notebooks/` are valuable for exploratory work but must not be the sole home of stable logic.
- `streamlit_app/` and `presentation/` are strictly end-stage, read-only communication assets.
- Do not create application, API, cloud, streaming, monitoring, or deployment folders.

## Effort allocation if time is limited

1. Source provenance, timestamp conversion, data audit, and leakage policy.
2. Correct targets, rotation eligibility, weather matching, past-only features, and temporal validation.
3. Baselines, Logistic/Ridge, LightGBM/XGBoost, calibration, and error analysis.
4. Propagation, network, recovery, and regime analysis.
5. Final reports, README, and reproducible figures.
6. LSTM/GRU.
7. Transformer, DiD, HDBSCAN, cross-entity extensions, dashboard/demo.

The project remains excellent if sophisticated deep learning is rejected. It fails if it makes pre-flight or propagation claims using information that was only known later.
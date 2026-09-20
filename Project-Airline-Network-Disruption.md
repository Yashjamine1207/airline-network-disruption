# Airline Network Disruption Intelligence — Repository Structure

**Repository:** `airline-network-disruption-intelligence`

This is a professional modular, research-first repository for a large-scale aviation disruption study. It supports data provenance, timezone-safe temporal reconstruction, delay/cancellation prediction, aircraft-rotation propagation, weather matching, network science, survival/recovery analysis, TensorFlow sequence models, and rigorous evaluation.

It intentionally excludes APIs, streaming, cloud infrastructure, airline-system integration, and production operations tooling.

```text
airline-network-disruption-intelligence/
├── configs/
│   ├── base.yaml
│   ├── data_sources.yaml
│   ├── ingestion.yaml
│   ├── airports_timezones.yaml
│   ├── weather_matching.yaml
│   ├── targets.yaml
│   ├── features.yaml
│   ├── rotation_reconstruction.yaml
│   ├── network_features.yaml
│   ├── validation.yaml
│   ├── classification.yaml
│   ├── regression.yaml
│   ├── survival.yaml
│   ├── calibration.yaml
│   ├── decision_policy.yaml
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   ├── lstm.yaml
│   ├── gru.yaml
│   └── transformer.yaml
│
├── data/
│   ├── external/
│   │   ├── dataset_readme.md
│   │   ├── source_register.csv
│   │   ├── flight_field_mapping.md
│   │   ├── ghcnh_field_mapping.md
│   │   ├── airport_station_mapping.csv
│   │   └── airport_timezone_mapping.csv
│   ├── raw/                              # ignored; immutable external source files
│   │   ├── flights_kaggle/
│   │   └── noaa_ghcnh/
│   ├── interim/                          # ignored; standardised/normalised tables
│   │   ├── flights_standardised/
│   │   ├── weather_standardised/
│   │   └── airport_reference/
│   ├── processed/                        # ignored; analytical tables
│   │   ├── flight_panel/
│   │   ├── rotations/
│   │   ├── network_windows/
│   │   ├── survival_episodes/
│   │   └── model_ready/
│   ├── features/                         # ignored; model matrices and sequences
│   │   ├── tabular/
│   │   ├── sequences/
│   │   └── network/
│   └── sample/                           # tiny safe/synthetic recruiter-facing examples
│       ├── sample_flights.csv
│       ├── sample_weather.csv
│       ├── sample_rotation.csv
│       ├── sample_predictions.csv
│       └── sample_network_metrics.csv
│
├── docs/
│   ├── architecture/
│   │   ├── analytical_architecture.png
│   │   ├── data_lineage.png
│   │   ├── prediction_timestamp_design.png
│   │   ├── timezone_normalisation_flow.png
│   │   ├── aircraft_rotation_flow.png
│   │   └── disruption_decision_flow.png
│   ├── data/
│   │   ├── data_card.md
│   │   ├── data_dictionary.md
│   │   ├── source_metadata.md
│   │   ├── flight_data_audit.md
│   │   ├── weather_data_audit.md
│   │   ├── timezone_methodology.md
│   │   ├── carrier_identifier_policy.md
│   │   ├── airport_station_matching_audit.md
│   │   ├── rotation_reconstruction_audit.md
│   │   ├── regime_shift_analysis.md
│   │   ├── leakage_audit.md
│   │   └── feature_catalog.md
│   ├── decisions/
│   │   ├── decision_log.md
│   │   ├── target_definitions.md
│   │   ├── prediction_timestamp_policy.md
│   │   ├── metric_definitions.md
│   │   ├── recovery_definition.md
│   │   ├── weather_shock_definition.md
│   │   ├── alert_threshold_policy.md
│   │   ├── utility_scenarios.md
│   │   └── limitations_and_appropriate_use.md
│   └── experiments/
│       ├── modelling_plan.md
│       ├── baseline_plan.md
│       ├── propagation_analysis_plan.md
│       ├── network_analysis_plan.md
│       ├── survival_analysis_plan.md
│       ├── weather_event_study_plan.md
│       ├── advanced_model_ablation_plan.md
│       ├── cross_carrier_generalisation_plan.md
│       └── advanced_signal_results.md
│
├── models/
│   ├── artifacts/                        # ignored
│   ├── checkpoints/                      # ignored; TensorFlow checkpoints
│   ├── registry/
│   │   └── model_registry.md
│   └── metrics/
│       ├── classification_model_comparison.csv
│       ├── regression_model_comparison.csv
│       ├── calibration_metrics.csv
│       ├── propagation_metrics.csv
│       ├── survival_metrics.csv
│       ├── network_metrics.csv
│       ├── ablation_results.csv
│       └── policy_evaluation.csv
│
├── notebooks/
│   ├── 01_data_source_audit.ipynb
│   ├── 02_data_quality_and_coverage.ipynb
│   ├── 03_timezone_reconstruction.ipynb
│   ├── 04_delay_and_cancellation_eda.ipynb
│   ├── 05_regime_shift_analysis.ipynb
│   ├── 06_target_construction.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_baseline_and_statistical_models.ipynb
│   ├── 09_boosting_models.ipynb
│   ├── 10_aircraft_rotation_reconstruction.ipynb
│   ├── 11_propagation_analysis.ipynb
│   ├── 12_network_analysis.ipynb
│   ├── 13_survival_recovery_analysis.ipynb
│   ├── 14_weather_matching_and_analysis.ipynb
│   ├── 15_weather_event_study.ipynb
│   ├── 16_tensorflow_lstm_gru.ipynb
│   ├── 17_transformer_challenger.ipynb
│   ├── 18_calibration_and_alert_policy.ipynb
│   ├── 19_explainability.ipynb
│   ├── 20_ablation_studies.ipynb
│   ├── 21_cross_entity_generalisation.ipynb
│   └── 22_error_analysis.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── coverage_by_period.png
│   │   ├── missingness_analysis.png
│   │   ├── delay_distribution.png
│   │   ├── cancellation_rate_by_regime.png
│   │   ├── regime_shift_summary.png
│   │   ├── previous_vs_current_delay.png
│   │   ├── propagation_probability.png
│   │   ├── cascade_length_distribution.png
│   │   ├── airport_network.png
│   │   ├── centrality_vs_disruption.png
│   │   ├── weather_shock_trajectory.png
│   │   ├── pr_curve.png
│   │   ├── calibration_curve.png
│   │   ├── recall_at_k.png
│   │   ├── kaplan_meier_recovery.png
│   │   ├── model_comparison.png
│   │   ├── training_curves.png
│   │   ├── shap_summary.png
│   │   └── local_flight_explanation.png
│   ├── tables/
│   │   ├── source_coverage_summary.csv
│   │   ├── rotation_coverage_summary.csv
│   │   ├── weather_match_quality.csv
│   │   ├── feature_summary.csv
│   │   ├── classification_model_comparison.csv
│   │   ├── regression_model_comparison.csv
│   │   ├── calibration_summary.csv
│   │   ├── propagation_summary.csv
│   │   ├── recovery_summary.csv
│   │   ├── network_exposure_summary.csv
│   │   ├── ablation_results.csv
│   │   └── error_breakdown.csv
│   └── evaluation/
│       ├── data_quality_report.md
│       ├── timezone_and_sequence_report.md
│       ├── regime_shift_report.md
│       ├── disruption_prediction_report.md
│       ├── delay_regression_report.md
│       ├── calibration_and_alert_policy_report.md
│       ├── propagation_report.md
│       ├── network_resilience_report.md
│       ├── survival_recovery_report.md
│       ├── weather_event_analysis_report.md
│       ├── explainability_report.md
│       ├── error_analysis_report.md
│       ├── cross_entity_generalisation_report.md
│       ├── operational_value_report.md
│       ├── final_model_selection_report.md
│       └── final_technical_report.md
│
├── scripts/
│   ├── download_flight_data.py
│   ├── download_weather_subset.py
│   ├── validate_raw_sources.py
│   ├── ingest_flights.py
│   ├── ingest_weather.py
│   ├── normalise_timestamps.py
│   ├── build_rotations.py
│   ├── build_network_windows.py
│   ├── build_targets.py
│   ├── build_features.py
│   ├── run_backtests.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── run_survival_analysis.py
│   ├── run_event_study.py
│   ├── create_portfolio_tables.py
│   └── create_portfolio_figures.py
│
├── src/
│   └── airline_disruption/
│       ├── __init__.py
│       ├── data/
│       │   ├── acquisition.py
│       │   ├── source_metadata.py
│       │   ├── ingestion.py
│       │   ├── schema.py
│       │   ├── validation.py
│       │   ├── standardisation.py
│       │   └── parquet_io.py
│       ├── timestamps/
│       │   ├── airport_timezones.py
│       │   ├── local_to_utc.py
│       │   ├── schedule_times.py
│       │   ├── availability.py
│       │   └── validation.py
│       ├── targets/
│       │   ├── severe_delay.py
│       │   ├── cancellation.py
│       │   ├── delay_regression.py
│       │   ├── propagation.py
│       │   └── recovery.py
│       ├── features/
│       │   ├── schedule.py
│       │   ├── historical.py
│       │   ├── carrier.py
│       │   ├── route.py
│       │   ├── airport.py
│       │   ├── rotation.py
│       │   ├── calendar.py
│       │   ├── weather.py
│       │   ├── network.py
│       │   ├── sequences.py
│       │   └── point_in_time.py
│       ├── rotation/
│       │   ├── reconstruction.py
│       │   ├── eligibility.py
│       │   ├── propagation.py
│       │   └── episodes.py
│       ├── network/
│       │   ├── graph_builder.py
│       │   ├── dynamic_windows.py
│       │   ├── centrality.py
│       │   ├── exposure.py
│       │   └── resilience.py
│       ├── weather/
│       │   ├── stations.py
│       │   ├── matching.py
│       │   ├── quality.py
│       │   ├── features.py
│       │   └── events.py
│       ├── validation/
│       │   ├── temporal_splits.py
│       │   ├── rolling_origin.py
│       │   ├── embargo.py
│       │   ├── carrier_holdout.py
│       │   ├── airport_holdout.py
│       │   └── backtesting.py
│       ├── models/
│       │   ├── baselines.py
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── boosting.py
│       │   ├── lstm.py
│       │   ├── gru.py
│       │   ├── transformer.py
│       │   ├── calibration.py
│       │   └── training.py
│       ├── survival/
│       │   ├── kaplan_meier.py
│       │   ├── cox.py
│       │   ├── aft.py
│       │   └── diagnostics.py
│       ├── events/
│       │   ├── weather_shocks.py
│       │   ├── event_study.py
│       │   └── did.py
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── calibration.py
│       │   ├── propagation.py
│       │   ├── survival.py
│       │   ├── network.py
│       │   ├── ablations.py
│       │   ├── operational_value.py
│       │   ├── error_analysis.py
│       │   └── reporting.py
│       ├── explainability/
│       │   ├── shap_explainer.py
│       │   ├── permutation.py
│       │   ├── temporal_ablation.py
│       │   └── case_studies.py
│       ├── visualisation/
│       │   ├── data_quality.py
│       │   ├── propagation.py
│       │   ├── network.py
│       │   ├── survival.py
│       │   ├── weather.py
│       │   ├── models.py
│       │   └── explainability.py
│       └── utils/
│           ├── config.py
│           ├── logging.py
│           ├── paths.py
│           ├── reproducibility.py
│           └── versions.py
│
├── streamlit_app/                       # optional; prepared results only
│   ├── Home.py
│   └── pages/
│       ├── 01_Project_Overview.py
│       ├── 02_Disruption_Risk.py
│       ├── 03_Propagation_and_Network.py
│       ├── 04_Recovery_and_Weather.py
│       └── 05_Limitations.py
│
├── presentation/                        # optional Power BI, slides, screenshots
│   ├── README.md
│   └── screenshots/
│
├── tests/
│   ├── data/
│   │   ├── test_schema_validation.py
│   │   ├── test_flight_keys.py
│   │   ├── test_carrier_identifiers.py
│   │   └── test_weather_quality.py
│   ├── timestamps/
│   │   ├── test_local_to_utc.py
│   │   ├── test_midnight_rollover.py
│   │   ├── test_dst.py
│   │   └── test_prediction_availability.py
│   ├── targets/
│   │   ├── test_severe_delay.py
│   │   ├── test_cancellation.py
│   │   └── test_recovery.py
│   ├── features/
│   │   ├── test_historical_features.py
│   │   ├── test_rotation_features.py
│   │   ├── test_weather_matching.py
│   │   └── test_point_in_time_safety.py
│   ├── rotation/
│   │   ├── test_rotation_ordering.py
│   │   └── test_eligibility.py
│   ├── validation/
│   │   ├── test_temporal_split.py
│   │   ├── test_embargo.py
│   │   ├── test_carrier_holdout.py
│   │   └── test_airport_holdout.py
│   └── unit/
│       ├── test_classification_metrics.py
│       ├── test_regression_metrics.py
│       ├── test_calibration.py
│       ├── test_survival_metrics.py
│       └── test_policy.py
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile                             # optional convenience commands only
├── MLproject                            # optional; retain only if MLflow is used
├── README.md
├── pyproject.toml
└── CITATION.cff                         # optional for public portfolio/research release
```

## Structural rules

- `data/raw/`, `data/interim/`, `data/processed/`, and `data/features/` are generated or downloaded data areas and must be ignored by Git.
- `data/sample/` may include only a very small safe synthetic or permitted example, not the full Kaggle or NOAA data.
- `docs/data/` is mandatory: source provenance, time zones, weather matching, rotation eligibility, and leakage decisions are central to this project’s credibility.
- `configs/` must hold target thresholds, time boundaries, sequence windows, feature lookbacks, model parameters, calibration, and policy assumptions.
- `notebooks/` are for investigation and visual development; reusable production-quality analytical logic belongs in `src/`.
- `models/artifacts/`, `models/checkpoints/`, and MLflow runs are local artefacts, not repository content.
- `streamlit_app/` is optional and read-only. It must not use live aviation data, authentication, an API, a database, or operational controls.

Do not create API, cloud, Docker deployment, streaming, queue, monitoring, or airline-integration folders. The quality of the repository should come from data/time correctness, comparable experiments, and evidence-led conclusions.
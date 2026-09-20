# Rules for Airline Network Disruption Intelligence

## Identity and use

- Build an advanced Data Science, aviation analytics, time-series, sequence-modelling, network-science, and survival-analysis portfolio project.
- Do not present this as airline production software, a dispatch system, a crew/aircraft optimiser, an airline decision engine, or a live monitoring product.
- Use flight operations, aircraft/tail, carrier, route, airport, timing, network, and weather information only. Do not introduce passenger-level personal data or attempt to identify passengers.
- State throughout the README, reports, and optional demo that outputs are retrospective analytical evidence and not real-time operational recommendations.

## Data boundary and provenance

- The official project pipeline contains exactly two external datasets: the selected Kaggle flight-delay/cancellation dataset and NOAA GHCNh weather observations.
- Do not add FAA, separate storm-event, commercial aviation, passenger, or other data sources to the core scope without revising the research protocol.
- Record source URL, publisher, dataset title, access date, actual coverage, file name, schema, hash where feasible, and licence/terms note.
- Confirm actual source fields and available dates after download. Do not write code that assumes the published summary exactly matches the files.
- Keep raw downloads immutable in `data/raw/`; keep raw, interim, processed, feature, checkpoint, and model-artifact data out of Git.
- Store analytical tables in Parquet. Use DuckDB/Polars/Pandas in ways appropriate to the approximately 30-million-record scale; do not repeatedly load an entire raw CSV into memory without a planned subset/query strategy.
- Retrieve only weather stations, time periods, and variables required for the selected airport/flight analysis. Do not download an unnecessary global archive.

## Timestamps and time zones

- Treat timestamp handling as a scientific requirement, not an implementation detail.
- Preserve original local-time fields and create documented normalised timestamps, normally UTC, for ordering and joining.
- Maintain a versioned airport-to-timezone mapping.
- Test and handle overnight flights, midnight crossings, different origin/destination time zones, daylight-saving transitions, missing schedules, and invalid clock values.
- Do not calculate a duration or sequence order by naïvely subtracting local timestamps from different airport time zones.
- Define a prediction timestamp for every model task and document it in `prediction_timestamp_policy.md`.

## Leakage prevention

- The main evaluation uses chronological splits only. Never use a random train/test split for headline results.
- For a prediction at time `T`, use only information demonstrably available by `T`.
- Never use actual future arrival/departure information, final delay, cancellation reasons, delay-cause fields, later flights, future airport congestion, future network features, future weather, or complete-day aggregates that include observations after `T`.
- Delay-cause fields are excluded from all pre-flight prediction models. They may be used only in retrospective post-event attribution analysis with that distinction made clear.
- Use actual information from a previous aircraft leg only when that leg’s outcome would have been known before the current prediction timestamp.
- Build historical route/carrier/airport rates, rolling aggregates, encoders, scalers, imputers, calibration maps, feature selection, and hyperparameters within each training fold only.
- Calculate network features on past-only temporal windows. Do not build a graph from the final test period and use its centralities to predict earlier/test flights.
- Match weather using only observations available before the flight’s defined prediction timestamp. Never use weather observed after departure/prediction time.
- Include explicit unit tests for future-flight, future-weather, and future-network leakage.

## Targets and cohorts

- Treat `arrival delay ≥ 120 minutes` as the primary project-defined severe-delay target, not an official standard. Run sensitivity analysis at 60, 90, and 180 minutes.
- Calculate cancellation prevalence from the selected data.
- Build arrival-delay regression only for a clearly documented completed-flight cohort.
- Restrict rotation/propagation analysis to tail sequences that pass documented eligibility and timestamp-quality checks.
- Treat recovery below 15 minutes delay as a candidate project-defined normal-state threshold. Document sensitivity analyses and right-censoring treatment.
- Never claim aircraft-level propagation for records where aircraft identity, ordering, or prior-leg linkage cannot be established reliably.

## Validation and experiments

- Use actual audited dates to finalise the proposed 2019–2021 training, 2022 validation, 2023 final-test chronology.
- Keep the final temporal test period untouched until model, threshold, calibration, and policy decisions are locked.
- Use expanding-window/rolling-origin backtests inside development.
- Report results separately across 2019, 2020, and 2021–2023 regimes where data support it. Do not treat 2020 as a disposable outlier.
- Treat cross-carrier and cross-airport evaluation as distinct entity-generalisation experiments, not replacements for temporal evaluation.
- Log data version, date cut-off, target version, feature version, split version, code version, random seed, parameters, and metrics through MLflow or a reproducible experiment log.

## Modelling rules

- Start with majority/historical-rate/median baselines, then Logistic/Linear/Ridge/Elastic Net, then LightGBM/XGBoost, then TensorFlow LSTM/GRU, then a compact Transformer challenger.
- Use TensorFlow/Keras for deep learning. Do not add PyTorch unless scope is formally changed.
- Do not include a model because it is fashionable. Retain it only when measured improvements justify added complexity, resource use, instability, or reduced interpretability.
- It is a good result if a historical baseline, logistic regression, Ridge, LightGBM, or XGBoost outperforms sequence models.
- Use class weighting, PR-AUC, calibration, and threshold optimisation for rare events. Do not apply SMOTE blindly over the full temporal dataset.
- If resampling is tested, apply it only within the relevant training folds and document its temporal limitations.
- Do not build a graph neural network. Use NetworkX for network science/feature analysis.
- Do not claim causal effect from correlation, SHAP, network centrality, weather association, or event-study patterns.

## Metrics and interpretation

- Accuracy is never the headline metric for severe delay or cancellation because the events are rare.
- Use PR-AUC as the primary classification metric. Also report precision, recall, F1, Precision@K, Recall@K, Brier score, calibration, and threshold-level confusion matrices.
- Use MAE as the primary delay-regression metric, with RMSE, median absolute error, residual analysis, and upper-tail error as supporting evidence.
- Use appropriate recovery/survival metrics and clearly document censoring/assumptions.
- Treat network evaluation as descriptive/analytical: propagation probability, delay amplification, cascade depth/duration, exposure, centrality, and resilience. Do not invent a single network “accuracy.”
- Report results by regime, carrier, airport, route, weather coverage, sequence eligibility, and severity where meaningful.

## Explainability and event analysis

- Use SHAP for selected tree/tabular models after valid evaluation exists.
- Use local explanations for high-risk flights, true positives, false positives, false negatives, and low-risk cases—not only successful examples.
- For LSTM/GRU/Transformer, use temporal ablation, masking, permutation, or feature-family removal. Do not present attention weights as proof of a causal mechanism.
- Describe variables as contributing to or being associated with a prediction. Do not claim they caused a delay/cancellation/recovery outcome.
- Define weather shocks transparently from the observed data or clearly stated criteria.
- Present weather event studies as associations around observed shocks. Difference-in-differences may be attempted only with explicit discussion of parallel trends, control selection, spillovers, seasonality, concurrent disruptions, and limits to causal interpretation.

## Operational-value analysis

- Treat the system as a ranking/prioritisation analysis, not an autonomous airline action engine.
- Evaluate analyst-capacity scenarios such as top 0.5%, 1%, 5%, and 10% of flights, and report Precision@K, Recall@K, events captured, and false alerts.
- Label all penalties, utility weights, capacity limits, or costs as illustrative analytical assumptions. Never present them as airline financial estimates or operational policy.
- Separate measured model metrics from assumed scenario values.

## Code, testing, and repository conduct

- Work in VS Code. Use notebooks for exploration, figure development, and experiments only; move stable logic into `src/airline_disruption/`.
- Use `pyproject.toml` as the primary dependency/tooling definition. Do not use `requirements-dev.txt` by default.
- Use pre-commit and focused pytest coverage.
- Prioritise tests for timestamps, DST/midnight handling, feature availability, tail ordering, rotation eligibility, weather matching, target alignment, temporal splits, holdouts, metrics, calibration, and scenario-policy edge cases.
- Keep data, model files, TensorFlow checkpoints, MLflow runs, DuckDB databases, `.env`, Kaggle credentials, logs, and local IDE settings out of version control.

## Scope control

- Stop after the final technical report, final model-selection report, README, and optional read-only presentation.
- Build a Streamlit or Power BI asset only after the analytical reports are complete. It may display prepared safe results, but must not use a live data pipeline, API, authentication, database, or operational controls.
- Do not add FastAPI, Kafka, queues, real-time streaming, cloud deployment, Kubernetes, Terraform, microservices, Docker deployment work, monitoring infrastructure, LLMs, RAG, agents, or airline operations software.

## Final quality bar

A reviewer must be able to determine:

1. The precise prediction and disruption/propagation questions.
2. The provenance, coverage, licence caveats, and limitations of the two data sources.
3. How local times, UTC ordering, prior-leg availability, weather matching, and future-data leakage were controlled.
4. How models were compared over chronological/regime-based evaluations.
5. Whether rotation, network, and weather features had measurable incremental value.
6. Whether probabilities were calibrated and events could be prioritised at realistic alert rates.
7. How disruption propagated, how recovery was defined, and where the evidence is limited.
8. Which model was selected, why, where it fails, and why the result is not a production-airline claim.

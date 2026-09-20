# Important repository files

Create these files at the start of the `airline-network-disruption-intelligence` repository.

## Required root files

- `README.md` — the recruiter-facing entry point. It must explain the disruption problem, two-source data scope, prediction-time rules, methods, results, limitations, and reproducibility.
- `LICENSE` — choose before public release. MIT is often suitable for portfolio code, but it does not relicense third-party flight or weather data.
- `.gitignore` — use the project-specific pattern. It must exclude full flight data, NOAA extracts, processed Parquet files, DuckDB databases, model artefacts, TensorFlow checkpoints, MLflow output, local environments, and secrets.
- `.env.example` — names only, never real values. The project should not require API keys, but may document optional local paths or Kaggle configuration variables if genuinely needed.
- `pyproject.toml` — the primary package, dependency, formatter, linter, test, and tool configuration. Use optional dependency groups rather than `requirements-dev.txt`.
- `.pre-commit-config.yaml` — enforce formatting, whitespace, YAML/JSON checks, secrets detection, and accidental large-file checks before commits.

## Required project documentation

Create these documents early. They are evidence of analytical rigour, not end-stage decoration.

### `docs/data/`

- `data_card.md` — dataset provenance, time coverage, unit of analysis, granularity, licence/usage note, limitations, ethical scope, and data-access instructions.
- `data_dictionary.md` — exact flight, airport, carrier, weather, target, and derived-variable definitions.
- `source_metadata.md` — source URLs, access dates, file names, file hashes where feasible, raw row counts, coverage, and transformation history.
- `timezone_methodology.md` — local-time fields, airport timezone mapping, UTC conversion, daylight-saving treatment, overnight flights, and known limitations.
- `carrier_identifier_policy.md` — carrier fields used, unique-carrier policy, and treatment of reporting-code/merger/joint-reporting changes.
- `airport_station_matching_audit.md` — airport/station mapping method, distance/selection logic, coverage, unmatched airports, time-gap distribution, and quality flags.
- `rotation_reconstruction_audit.md` — tail-ID coverage, ordering rules, eligible rotations, exclusions, missing/ambiguous sequence handling, and data-quality limitations.
- `regime_shift_analysis.md` — pre-COVID, shock, and recovery-period definitions/evidence.
- `leakage_audit.md` — allowed/forbidden fields, prediction timestamp, post-outcome exclusions, past-only aggregation, future-graph restrictions, and validation safeguards.
- `feature_catalog.md` — source fields, formulas, lookback windows, availability time, missing-data rules, and version for each feature.

### `docs/decisions/`

- `target_definitions.md` — severe delay thresholds/sensitivity analysis, cancellation handling, regression cohort, propagation outcomes, and recovery definitions.
- `prediction_timestamp_policy.md` — exactly when each model is assumed to make its prediction and which previous-leg/weather information is available then.
- `metric_definitions.md` — PR-AUC, recall@K, precision@K, MAE, calibration, recovery/survival metrics, and network measures.
- `recovery_definition.md` — disruption episode rules, normal-state definition, censoring, and sensitivity analysis.
- `weather_shock_definition.md` — threshold construction, exposure window, station quality, and event-study windows.
- `alert_threshold_policy.md` — ranking/threshold logic and validation-only selection.
- `utility_scenarios.md` — illustrative analyst capacity and false-positive/false-negative scenarios. It must state that values are not real airline costs.
- `limitations_and_appropriate_use.md` — analytical scope and reasons the project is not a live airline decision system.

### `docs/experiments/`

- `modelling_plan.md`
- `baseline_plan.md`
- `propagation_analysis_plan.md`
- `network_analysis_plan.md`
- `survival_analysis_plan.md`
- `weather_event_study_plan.md`
- `advanced_model_ablation_plan.md`
- `cross_carrier_generalisation_plan.md`
- `advanced_signal_results.md`

## Required configuration files

Version all material analytical decisions under `configs/`:

- `base.yaml` — paths, keys, random seeds, reproducibility settings.
- `data_sources.yaml` — source metadata and standardised field mapping references.
- `ingestion.yaml` — schema/date/carrier/airport parsing rules.
- `airports_timezones.yaml` — mapping version and timestamp-normalisation choices.
- `weather_matching.yaml` — station selection, observation cutoff, weather-variable selection, and quality filtering.
- `targets.yaml` — severe-delay thresholds, cancellation rules, regression cohort, episode/recovery definitions.
- `features.yaml` — historical aggregation windows, lag rules, categories, missingness handling, and availability restrictions.
- `rotation_reconstruction.yaml` — tail/sequence eligibility and turnaround rules.
- `network_features.yaml` — graph window, nodes/edges, weights, centrality methods, and past-only construction rules.
- `validation.yaml` — chronological split boundaries, rolling-origin backtests, embargo rules, carrier/airport holdouts.
- `classification.yaml`, `regression.yaml`, `survival.yaml`, and model-specific YAML files — experiment parameters/search spaces.
- `calibration.yaml` and `decision_policy.yaml` — validation-only probability calibration and alert-policy selection.

## Optional files

Keep these only when they support the finished analytical work:

- `Makefile` — short commands such as `make ingest`, `make features`, `make test`, `make train`, and `make report`.
- `MLproject` — keep only if MLflow is actually used.
- `CITATION.cff` — useful for a public research/portfolio release.
- `.github/workflows/tests.yml` — optional minimal lint/test workflow. Do not make CI/CD or deployment an additional project stream.
- `streamlit_app/` — a read-only prepared-results showcase, only after reports are complete.
- `presentation/` — Power BI assets, slides, and screenshots only after analytical work is complete.

## Do not add

Do not create these to make the repository look more “production-like”:

- FastAPI/Flask routes, API schemas, API clients, authentication, user accounts, or service databases.
- Docker/Compose/container publishing except for a truly necessary small local reproducibility case; by default, do not use them.
- Cloud, Terraform, Kubernetes, Helm, ECS/Fargate, serverless, queues, Kafka, live streaming, or monitoring folders.
- Airline dispatch, rebooking, crew, booking, or real-time scheduling modules.
- GNN production architecture, LLM/RAG, agents, chatbots, or unrelated generative-AI components.

## Minimum README structure

The README should have these sections in order:

1. Title and a concise disruption/propagation/recovery summary.
2. A clear statement that this is a retrospective Data Science portfolio study, not production airline software.
3. Research questions and headline final claim.
4. Two-source data architecture: flight operations data plus NOAA GHCNh weather data.
5. Data coverage, license/access caveat, and explicit instruction that full data are not in the repository.
6. Prediction timestamp and leakage safeguards, including timezones, prior-leg availability, post-outcome field exclusion, and past-only network features.
7. Methodology diagram: data audit → temporal reconstruction → features → baselines/boosting → propagation/network/survival → sequence models → calibration/error analysis.
8. Validation design: final temporal holdout, rolling-origin backtests, regime evaluation, and optional carrier/airport holdouts.
9. Model ladder and ablation plan.
10. Results table, with separate classification, regression, calibration, recovery, and propagation findings.
11. Key figures: propagation, network, calibration, recovery, and a local explanation.
12. Limitations and appropriate-use statement.
13. Setup/reproduction/data-acquisition instructions.
14. Repository structure, licence, and citation.

## Dependency strategy

Keep `pyproject.toml` as the single dependency source of truth. Suggested optional groups:

- `core`: pandas, NumPy, Polars, PyArrow, DuckDB, PyYAML.
- `statistics`: SciPy, statsmodels, lifelines, ruptures.
- `ml`: scikit-learn, LightGBM, XGBoost, Optuna, imbalanced-learn where justified, SHAP.
- `deep-learning`: TensorFlow.
- `network`: NetworkX.
- `visualisation`: Matplotlib, Seaborn, Plotly.
- `tracking`: MLflow.
- `dev`: pytest, pytest-cov, ruff, pre-commit, type-checking tool only if actively used.
- `demo`: Streamlit only if the optional presentation is built.

Use package constraints/lock strategy appropriate to your environment and record the Python version, platform, and package environment for final experiments.
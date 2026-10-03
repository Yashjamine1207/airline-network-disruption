# Airline Network Disruption Intelligence — Phase 5–7B Scope Update

## Authoritative scope decision

The project will continue through Phase 7B. Phase 0–4 decisions and completed work remain unchanged unless explicitly revised in a later decision log.

From Phase 5 onward, the official analytical pipeline uses **only the audited Kaggle flight dataset** already in the repository:

- `data/raw/flights_kaggle/flights_sample_3m.csv`
- Coverage: 2019-01-01 to 2023-08-31.
- Approximately 3,000,000 flight records.
- No NOAA, weather, FAA, passenger, commercial aviation, airport API, airline API, or other external dataset will be added.

This is a deliberate flight-only scope change. NOAA-related files, weather modules, weather figures, weather-event studies, and weather claims must not be presented as completed or required work. Existing Phase 0–4 documentation should remain historically accurate, while this decision document defines the revised plan for Phases 5–7B.

## What the Kaggle data supports

The audited source supports:

- Flight-level cancellation, severe-delay, and arrival-delay analysis.
- Schedule, calendar, carrier, airport, route, distance, and elapsed-time features.
- Past-only schedule counts and schedule-derived airport-network features.
- Airport-day disruption episodes and recovery analysis.
- Airport-route network construction across operating regimes.
- Statistical analysis of airport and route disruption pressure.
- Flight-level sequence modelling using ordered operational/schedule records, without claiming aircraft rotations.
- Temporal ablations, calibration, explainability, error analysis, and ranked analyst-prioritisation scenarios.

The source does not provide a usable tail identifier. Therefore the project will not claim aircraft-level rotations, aircraft-leg propagation, cascade depth through a specific aircraft, or tail-based recovery.

## Cross-phase rules

- Keep the prediction timestamp at scheduled departure UTC minus two hours for pre-flight prediction tasks.
- Preserve original local fields and use audited UTC timestamps for ordering and comparison.
- Use chronological evaluation only.
- Keep 2023-01-01 to 2023-08-31 as the final temporal evaluation period after all model family, feature, calibration, and threshold decisions are locked. Any earlier invalid final-test run must remain quarantined and must not be reported.
- Build every historical, rolling, network, and sequence feature point-in-time safely.
- Never use actual outcome fields from the predicted flight as pre-flight predictors.
- Use delay, cancellation, and arrival outcomes for retrospective analysis only when the analysis is explicitly post-event.
- Do not infer aircraft identity from carrier, route, flight number, airport, or schedule similarity.
- Separate descriptive network/recovery analysis from predictive features.
- Report associations, not causal effects, unless a credible design and assumptions are documented.
- Compare simple models with advanced models; retain LSTM, GRU, or Transformer only if they add measured value, robustness, calibration, or insight.
- Do not present the work as an airline operating platform, dispatch system, optimisation product, real-time control system, or production service.

# Phase 5 — Flight-only network, airport disruption, and recovery analysis

## Goal

Use the Kaggle flight records to study how disruption pressure varies across airports, routes, operating regimes, and airport-level recovery episodes, without making unsupported aircraft-rotation or weather claims.

## Work packages

### 5.1 Airport-route network construction

Build directed airport-route graphs with:

- Airports as nodes.
- Origin-to-destination routes as directed edges.
- Edge weights based on scheduled flight counts within a documented time window.
- Separate graph windows for 2019, 2020, 2021, 2022, and the partial 2023 period where coverage permits.
- Optional rolling 7-day, 30-day, and 90-day descriptive windows.

Calculate, where graph size and stability support them:

- In-degree and out-degree.
- Weighted in-degree and out-degree.
- Betweenness, closeness, and PageRank.
- Network density and connected-component summaries.
- Route diversity and concentration.
- Airport and route disruption exposure.
- Downstream scheduled-flight pressure.

### 5.2 Past-only predictive network features

Use only graph information available by the prediction timestamp. Candidate predictors include:

- Prior-month scheduled-flight counts.
- Prior-month origin out-degree and weighted out-degree.
- Prior-month destination in-degree and weighted in-degree.
- Past-window route frequency.
- Past-window airport traffic pressure.
- Past-window route concentration.

Every network feature must document its window end, source fields, formula, missingness rule, and leakage test. The completed Phase 3 prior-month network features are the starting point, not evidence that every planned network feature is complete.

### 5.3 Airport disruption and recovery

Use airport-day or airport-window aggregates to define disruption episodes from flight outcomes. The primary recovery work should:

- Preserve episode start, end, duration, observed-recovery indicator, and censoring indicator.
- Compare the primary recovery definition with sensitivity scenarios already established in Phase 3.
- Report episode counts, censoring rates, observed recovery summaries, and distributions by airport and regime.
- Fit Kaplan–Meier estimates when episode construction and censoring assumptions support them.
- Consider Cox or accelerated-failure-time models only after diagnostics, proportional-hazard checks, sample-size checks, and covariate availability checks.
- Treat results as airport-level associations, not causal or operational guarantees.

### 5.4 Regime and resilience analysis

Compare 2019 pre-COVID, 2020 shock, 2021–2022 transition/recovery, and partial 2023 behaviour using:

- Flight volume.
- Cancellation and severe-delay rates.
- Airport disruption pressure.
- Route concentration and network density.
- Centrality distributions.
- Episode duration and censoring.
- Time-to-recovery distributions.

Do not remove 2020 because it is unusual. Treat it as a distribution-shift regime.

### 5.5 Propagation scope correction

The project will not perform aircraft-tail propagation. Instead, it may analyse **airport and route disruption association** using documented temporal ordering:

- Prior airport disruption pressure versus subsequent flight disruption.
- Prior route disruption pressure versus subsequent route disruption.
- Airport-day disruption episodes and downstream scheduled exposure.
- Route-level delay pressure and next-window disruption.

These are not aircraft-level propagation measures and must not be labelled as such.

## Phase 5 outputs

- `reports/evaluation/network_resilience_report.md`
- `reports/evaluation/survival_recovery_report.md`
- `reports/evaluation/propagation_report.md` with airport/route scope clearly stated.
- `reports/tables/network_exposure_summary.csv`
- `reports/tables/recovery_summary.csv`
- `reports/tables/airport_route_disruption_summary.csv`
- `reports/figures/airport_network.png`
- `reports/figures/centrality_vs_disruption.png`
- `reports/figures/kaplan_meier_recovery.png`
- `configs/network_features.yaml`
- `configs/survival.yaml`
- Tests for point-in-time network construction, episode definitions, censoring, and split safety.

## Phase 5 completion check

A reviewer can identify:

- How airport-route graphs were built.
- Which network windows were used.
- Which features were available at prediction time.
- How airport disruption and recovery were defined.
- How censoring was handled.
- Why no aircraft-tail propagation claim is made.
- Why network and recovery findings are descriptive or associational.

# Phase 6 — Flight-only sequence models and ablations

## Goal

Test whether sequence models add value beyond the selected tabular baselines using only ordered Kaggle flight records and point-in-time-safe schedule, carrier, airport, route, and network information.

## Sequence design

Because no tail identifier exists, sequences must not be described as aircraft rotations. Approved alternatives are:

### Option A — Airport temporal sequences

For each airport, order historical flight or airport-window observations by validated UTC time. A sequence may contain:

- Prior scheduled flight counts.
- Prior cancellation and disruption pressure, only when outcome availability is explicitly approved for the retrospective task.
- Past airport network measures.
- Past route concentration and exposure.
- Calendar and regime indicators.

### Option B — Route temporal sequences

For each route, order past route-window records by UTC time and create fixed lookback sequences. Use only preceding windows and enforce the prediction timestamp cutoff.

### Option C — Flight-context sequences

Use a fixed number of preceding schedule records for the same airport, route, carrier, or documented entity. Do not call these aircraft-leg sequences. Do not use same-flight outcomes or future records.

The implementation must choose one primary sequence definition and document why it is scientifically defensible. The other options can be ablations if data volume and compute allow.

## Models

Train in this order:

1. Selected LightGBM tabular benchmark.
2. TensorFlow/Keras LSTM.
3. TensorFlow/Keras GRU.
4. One compact Transformer encoder challenger.

Use the same target definitions, eligible cohorts, chronological splits, prediction timestamp policy, and headline metrics across models.

## Sequence ablations

At minimum compare:

- Base tabular features.
- Base plus past-only schedule history.
- Base plus past-only airport/route network features.
- Base plus airport sequence.
- Base plus route sequence.
- LSTM versus GRU.
- Best recurrent model versus compact Transformer.
- Training with 2020 versus training without 2020, evaluated on later periods.

Do not create a rotation ablation because the source lacks a reliable aircraft identifier.

## Training safeguards

- Fit encoders, scalers, imputers, and calibration only within allowed training/calibration periods.
- Pad or truncate sequences using a documented lookback length.
- Use masks for padding and test masking behaviour.
- Ensure no sequence contains records later than the prediction timestamp.
- Use shared chronological folds and identical eligible cohorts where possible.
- Record sequence length, feature version, model parameters, seed, training time, parameter count, and hardware context.
- Stop training using development-era validation only. Do not use 2023 for early stopping.
- Keep TensorFlow checkpoints and model artefacts out of Git.

## Phase 6 metrics

Classification:

- PR-AUC as the primary metric.
- Precision, recall, F1, ROC-AUC.
- Precision@0.5%, 1%, 5%, and 10%.
- Brier score and reliability after validation-only calibration.
- Regime and missing-sequence subgroup performance.

Regression:

- MAE as the primary metric.
- RMSE, median absolute error, bias, and severe-subset MAE.
- Upper-tail error distribution.

## Phase 6 completion check

A reviewer can see:

- The exact flight-only sequence definition.
- Why the sequence is not an aircraft rotation.
- How point-in-time availability was enforced.
- Whether LSTM, GRU, or Transformer improved over LightGBM.
- Training stability, resource cost, calibration, and failure modes.
- A justified retain/reject decision for every advanced model.

A simpler LightGBM or linear model winning is a valid result.

# Phase 7A — Flight-only calibration, explainability, and error analysis

## Goal

Make the selected flight-only models interpretable, calibrated, and honest about failure modes.

## Calibration

- Compare uncalibrated, Platt/sigmoid, and isotonic probabilities where sample size supports them.
- Fit calibrators on a disjoint development-era calibration period or cross-fitting scheme.
- Never fit a calibrator on final-test labels.
- Report Brier score, reliability diagrams, calibration slope/intercept where appropriate, and mean predicted probability versus event prevalence.
- Evaluate calibration separately for 2019, 2020, 2021–2022, and partial 2023 only when the relevant evaluation protocol permits it.
- Treat 2023 as final evaluation; do not use it to select the calibration method.

## Tabular explainability

For the selected Logistic Regression, Ridge, or LightGBM model:

- Report coefficient or feature-effect summaries appropriate to the model.
- Use SHAP for LightGBM only after the model and feature version are locked.
- Produce global feature importance and local explanations for representative cases.
- Include true positives, false positives, false negatives, high-confidence errors, and low-risk cases.
- Describe features as associated with predictions, not causes of disruption.

## Sequence explainability

For LSTM, GRU, and Transformer models use:

- Feature-family ablation.
- Temporal masking.
- Lookback removal.
- Permutation or input perturbation.
- Performance change by sequence position.

Do not present Transformer attention weights as causal explanations.

## Error analysis

Analyse errors by:

- Operating regime.
- Carrier and airport.
- Route volume.
- Scheduled departure hour and day of week.
- Cancellation/severe-delay prevalence.
- Missingness and unseen categories.
- Sequence availability and sequence length.
- Prediction confidence.
- Upper-tail arrival-delay severity.

Create case-study tables for:

- Correct high-risk alerts.
- Missed severe events.
- False alarms.
- Low-risk correct predictions.
- Large regression errors.
- Airport episodes with fast and slow recovery.

## Phase 7A outputs

- `reports/evaluation/calibration_and_alert_policy_report.md`
- `reports/evaluation/explainability_report.md`
- `reports/evaluation/error_analysis_report.md`
- `reports/tables/calibration_summary.csv`
- `reports/tables/error_breakdown.csv`
- `reports/figures/calibration_curve.png`
- `reports/figures/pr_curve.png`
- `reports/figures/shap_summary.png`
- `reports/figures/local_flight_explanation.png`
- `reports/figures/training_curves.png`
- `configs/calibration.yaml`
- Tests for calibration split safety, metric correctness, explanation cohort selection, and no-final-test fitting.

## Phase 7A completion check

A reviewer can identify:

- Which probabilities are calibrated and on which period.
- Whether calibration generalises across regimes.
- Which features are associated with predictions.
- Where each selected model fails.
- Whether sequence information adds useful signal.
- Why explanations are not causal claims.

# Phase 7B — Flight-only operational prioritisation and final portfolio story

## Goal

Translate validated model outputs into transparent retrospective ranking scenarios and a final evidence-led portfolio report, without claiming airline authority or real costs.

## Ranking and alert scenarios

For cancellation and severe-delay predictions, evaluate analyst-review capacities at:

- Top 0.5%.
- Top 1%.
- Top 5%.
- Top 10%.

Report:

- Flights reviewed.
- Events captured.
- Precision@K.
- Recall@K.
- False alerts.
- Event prevalence.
- Results by regime, airport, carrier, and missingness subgroup.

Use calibrated probabilities for probability-based thresholds only after calibration is locked. Ranking metrics must state whether they use calibrated or uncalibrated scores.

## Illustrative utility scenarios

If scenario penalties are used:

- Define missed-event and false-alert weights explicitly.
- Label them as illustrative analytical assumptions.
- Do not present them as airline financial costs.
- Test sensitivity to alternative weights and analyst capacities.
- Separate measured metrics from assumed utility values.

## Final model-selection decision

The final decision must record:

- Target and eligible cohort.
- Prediction timestamp.
- Feature version.
- Sequence definition, if applicable.
- Training, calibration, validation, and final-test periods.
- Selected model and rejected challengers.
- PR-AUC, MAE, calibration, ranking, and robustness evidence.
- Regime-specific behaviour.
- Missingness and generalisation limitations.
- Why complexity was retained or rejected.

## Final portfolio deliverables

- Final technical report.
- Final model-selection report.
- Calibration and alert-policy report.
- Operational-value report.
- Network/resilience report.
- Survival/recovery report.
- Propagation report with airport/route-only terminology.
- Final README written after all analysis is complete.
- Curated final figures and small result tables.
- Optional read-only presentation using prepared outputs only.

## Phase 7B completion check

A reviewer can understand the complete flight-only study without running a live system:

- What was predicted.
- What information was available at prediction time.
- How airport and route disruption was analysed.
- Why aircraft-level propagation was not claimed.
- How LSTM, GRU, Transformer, LightGBM, Logistic Regression, and Ridge compared.
- Whether calibration improved reliability.
- Which events could be prioritised under illustrative capacities.
- Where the model fails.
- What the results do and do not support.

## Revised final narrative

```text
Kaggle flight data only
→ source, schema, timestamp, and data-quality audit
→ regime-shift analysis
→ leakage-safe targets and flight-only features
→ transparent baselines and tabular models
→ airport-route network analysis
→ airport disruption episodes and survival/recovery analysis
→ airport/route temporal context instead of aircraft rotations
→ TensorFlow LSTM/GRU sequence models
→ compact Transformer challenger
→ validation-only calibration
→ SHAP, temporal ablation, and error analysis
→ ranked analyst-prioritisation scenarios
→ final evidence-led model-selection decision
→ final technical report and README
```

The project continues through Phase 7B. The absence of NOAA and tail identifiers is a documented scope limitation, not a reason to stop. The final portfolio claim must remain proportional to the evidence: flight-level prediction, airport/route network resilience, airport-level recovery, and flight-only sequence-model comparison—not weather attribution, aircraft-tail propagation, or a production airline control system.

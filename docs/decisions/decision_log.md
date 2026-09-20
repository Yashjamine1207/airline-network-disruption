# Decision Log

## Purpose

This log records material project decisions that affect scope, data handling, reproducibility, modelling, validation, interpretation, or comparability of results.

A new entry must be added when a decision changes:

- The external data boundary.
- A target definition.
- The prediction timestamp.
- Timezone handling.
- Leakage controls.
- Rotation eligibility.
- Weather matching.
- Validation dates or split design.
- Model family or configuration.
- Calibration approach.
- Alert-ranking or threshold policy.
- Recovery definition.
- A major inclusion, exclusion, or data-quality rule.

## Decision status values

| Status | Meaning |
|---|---|
| Accepted | The decision is active and should be followed |
| Provisional | The decision is active for planning but awaits data-audit confirmation |
| Superseded | The decision has been replaced by a later documented decision |
| Rejected | The option was considered and not adopted |

---

## DEC-001 — Project identity and appropriate use

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Position the work as a retrospective aviation Data Science portfolio study |
| Rationale | The project is intended to demonstrate Data Science, applied science, forecasting, operations research, decision science, transportation, and aviation analytics skills |
| Consequence | The project must not be presented as live airline operating software, dispatch software, crew or aircraft optimisation, a booking system, or an autonomous decision engine |
| Evidence | `docs/decisions/limitations_and_appropriate_use.md` |

---

## DEC-002 — Official external data boundary

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Use exactly two external sources in the official pipeline |
| Sources | Kaggle Flight Delay and Cancellation Dataset; NOAA Global Historical Climatology Network hourly weather observations |
| Rationale | A fixed data boundary supports scope control, reproducibility, and clear provenance |
| Consequence | FAA, separate NOAA Storm Events, commercial aviation, passenger, airport API, airline API, and other datasets are excluded unless this protocol is formally revised |
| Evidence | `docs/data/source_metadata.md`, `configs/data_sources.yaml` |

---

## DEC-003 — Raw flight-data handling

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Preserve the downloaded flight CSV and HTML dictionary as immutable raw source materials |
| Local raw files | `data/raw/flights_kaggle/flights_sample_3m.csv`; `data/raw/flights_kaggle/dictionary.html` |
| Rationale | Raw immutability supports auditability and reproducible reconstruction |
| Consequence | Raw data, derived data, feature matrices, model artefacts, and local credentials remain outside Git |
| Evidence | `.gitignore`, `configs/ingestion.yaml`, `configs/data_sources.yaml` |

---

## DEC-004 — Primary severe-delay target

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Define primary severe arrival delay as final arrival delay greater than or equal to 120 minutes |
| Sensitivity thresholds | 60, 90, 120, and 180 minutes |
| Rationale | A 120-minute threshold defines a clearly material disruption event for this project |
| Consequence | This is a project-defined analytical target, not an official airline, airport, FAA, NOAA, or regulatory standard |
| Evidence | `docs/decisions/target_definitions.md`, `configs/targets.yaml` |

---

## DEC-005 — Prediction timestamp

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Provisional |
| Decision | Define the primary pre-flight prediction timestamp as two hours before scheduled departure, represented in UTC |
| Formula | `prediction_timestamp_utc = scheduled_departure_utc - 2 hours` |
| Rationale | A fixed horizon creates a clear point-in-time boundary for pre-flight disruption-risk analysis |
| Consequence | Only information available on or before this timestamp may enter a predictive feature |
| Confirmation requirement | Confirm that the downloaded source supports reliable scheduled-departure timestamp reconstruction during Phase 1 |
| Evidence | `docs/decisions/prediction_timestamp_policy.md`, `configs/base.yaml` |

---

## DEC-006 — Timezone methodology

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Preserve original local time fields and create auditable UTC-normalised timestamps |
| Rationale | Aircraft sequencing, weather matching, and cross-airport time comparison require valid timezone handling |
| Consequence | Local timestamps from different airports must never be directly subtracted or used for cross-airport sequencing without conversion |
| Evidence | `docs/data/timezone_methodology.md`, `configs/airports_timezones.yaml` |

---

## DEC-007 — Leakage prevention

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Restrict pre-flight model features to information available on or before the prediction timestamp |
| Rationale | Future-information leakage invalidates predictive performance claims |
| Consequence | Exclude actual outcomes from the predicted flight, delay-cause fields, future weather, future flights, future aircraft legs, and future network information |
| Evidence | `docs/data/leakage_audit.md`, `docs/decisions/prediction_timestamp_policy.md` |

---

## DEC-008 — Validation design

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Provisional |
| Decision | Use chronological validation, with intended 2019–2021 development, 2022 validation, and 2023 final testing |
| Rationale | Flight disruption data are time-dependent and must be evaluated on later periods |
| Consequence | Random train-test splitting is prohibited for headline results; the final test period remains untouched until selection decisions are locked |
| Confirmation requirement | Confirm actual source date coverage before finalising date boundaries |
| Evidence | `docs/decisions/validation_strategy.md`, `configs/validation.yaml` |

---

## DEC-009 — Model-comparison order

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Compare models in order: baselines, statistical models, LightGBM/XGBoost, then TensorFlow LSTM/GRU/compact Transformer challengers |
| Rationale | Complex models should demonstrate incremental value over transparent baselines and strong tabular methods |
| Consequence | Deep learning is an ablation challenger, not a required winning approach |
| Evidence | `configs/classification.yaml`, `configs/regression.yaml`, `configs/lightgbm.yaml`, `configs/xgboost.yaml`, `configs/lstm.yaml`, `configs/gru.yaml`, `configs/transformer.yaml` |

---

## DEC-010 — Primary metrics

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Use PR-AUC as the primary classification metric and MAE as the primary arrival-delay regression metric |
| Rationale | Severe delays and cancellations are expected to be rare; arrival-delay prediction error should be reported directly in minutes |
| Consequence | Accuracy is not a headline rare-event metric |
| Evidence | `docs/decisions/metric_definitions.md` |

---

## DEC-011 — Recovery definition

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Provisional |
| Decision | Define candidate normal operation as arrival delay below 15 minutes; recovery is the first later eligible operated flight in a validated rotation that returns below this threshold |
| Rationale | This supports consistent construction of recovery episodes and survival analysis |
| Consequence | Recovery results require documented right-censoring, rotation eligibility, and sensitivity analysis |
| Confirmation requirement | Confirm source outcome fields and rotation coverage during the Phase 1 audit |
| Evidence | `docs/decisions/recovery_definition.md`, `configs/survival.yaml` |

---

## DEC-012 — Weather matching

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Match weather using NOAA GHCNh observations at or before the flight prediction timestamp |
| Rationale | Weather observations after prediction time would create future-data leakage |
| Consequence | Preserve station identity, observation time, quality metadata, matching method, and time gap |
| Evidence | `configs/weather_matching.yaml` |

---

## DEC-013 — Ranked prioritisation scenarios

| Field | Value |
|---|---|
| Date | 2026-09-20 |
| Status | Accepted |
| Decision | Evaluate ranked flights at the top 0.5%, 1%, 5%, and 10% of eligible flights |
| Rationale | This shows how models capture disruption events under hypothetical review-capacity constraints |
| Consequence | Capacity and utility values are illustrative analytical assumptions, not airline policy, staffing plans, or financial estimates |
| Evidence | `docs/decisions/utility_scenarios.md`, `configs/decision_policy.yaml` |
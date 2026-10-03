# Phase 6 decision log drafts

Append the two entries below to `docs/decisions/decision_log.md`, change the
status if you disagree, then delete this file.

---

## DEC-014 - UTC embargo at split boundaries

| Field | Value |
|---|---|
| Date | 2026-09-29 |
| Status | Accepted |
| Decision | Phase 6 keeps the Phase 3 local-date splits and adds a UTC embargo. A row whose UTC prediction timestamp falls on the wrong side of its split's UTC boundary is labelled `embargoed_boundary` and is excluded from training and evaluation for every Phase 6 model. Final-test membership is never widened. |
| Rationale | Splits were assigned by local flight date, but prediction time is scheduled departure UTC minus two hours. The Phase 3 manifest therefore overlapped in UTC at both boundaries (development ended 2022-01-01 06:50 UTC, validation started 2022-01-01 03:59 UTC). |
| Consequence | 287 of 2,999,999 rows are embargoed: 137 development, 150 validation, 0 final test. Final-test rows are unchanged (463,484; 454,413 eligible for severe delay). Phase 4 results stay as reported and are not recomputed. Phase 6 row counts differ from Phase 4 by these 287 rows. |
| Evidence | `src/airline_disruption/validation/temporal_splits.py`, `scripts/build_phase6_cohort.py`, `tests/validation/test_temporal_splits.py`, `reports/tables/phase6_cohort_summary.csv` |

---

## DEC-015 - Phase 6 final-test protocol

| Field | Value |
|---|---|
| Date | 2026-09-29 |
| Status | Proposed |
| Decision | Feature sets, model families, hyperparameters, early stopping, calibration method and alert thresholds for Phase 6 are chosen using development and 2022 validation data only. The 2023 final test is evaluated once per locked candidate, after the locked choices are written to a dated record. |
| Rationale | 2023 outcomes have already been seen in Phase 4: two quarantined invalid runs and the corrected severe-delay run (PR-AUC 0.0574, event rate 0.0319). The 2023 period is therefore not pristine for LightGBM severe-delay prediction, and the reports must say so. |
| Consequence | The Phase 4 corrected 2023 result is reported as a Phase 4 result and is not used to choose anything in Phase 6. The Phase 6 benchmark uses one code path for validation and final test. Cancellation and arrival-delay 2023 results remain unreported until a Phase 6 run under this protocol produces them. |
| Evidence | `reports/evaluation/phase4_final_summary.md`, `docs/decisions/validation_strategy.md` |

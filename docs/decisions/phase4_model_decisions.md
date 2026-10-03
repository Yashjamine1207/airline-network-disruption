# Phase 4 Model Decisions — Airline Network Disruption Intelligence

**Date:** 2026-09-26  
**Scope:** Flight-only configuration using Kaggle flight data (2019–2023), with 2019–2021 for development and 2022 for validation. The 2023 period remains locked for final evaluation only.[1][6]

## Selected modelling choices (validation-based)

These decisions are based on 2022 validation evidence only. They must be re-evaluated after the single 2023 final-test run.

### Classification targets

- **Cancellation:**
  - Model family: **Logistic Regression** with base schedule/carrier/airport/route features.
  - Rationale: Simple, well-calibrated probabilities, and sufficient ranking performance for a rare event.
  - Class weighting: **Not used** in the selected configuration (unweighted model had better Brier score with similar PR-AUC).
- **Severe arrival delay (≥ 120 minutes):**
  - Model family: **LightGBM** with base features.
  - Rationale: Higher PR-AUC than Logistic Regression on validation.
  - Calibration: **Isotonic calibration** on validation probabilities to align mean predicted probability with the event rate and reduce Brier score.
  - Feature set: **Base features only**; the schedule-count and network ablation reduced PR-AUC and is not used.

### Regression target

- **Arrival-delay regression (completed flights):**
  - Model family: **Ridge Regression** with base features.
  - Rationale: Slightly lower MAE than Linear and Elastic Net on validation, with similar RMSE.
  - No additional feature families added beyond base schedule/carrier/airport/route in this flight-only configuration.

### Validation design

- Chronological split:
  - Development: 2019–2021
  - Validation: 2022
  - Final test: 2023 (partial year, untouched until decisions are locked)
- Metrics:
  - Classification: **PR-AUC** primary, supported by ROC-AUC, Brier score, precision/recall at fixed review capacities.
  - Regression: **MAE (minutes)** primary, supported by RMSE, median absolute error, and severe-subset MAE.
- No random train/test splits; no leakage from 2023 into feature engineering, model selection, or calibration.

### Excluded directions (for this configuration)

- NOAA weather features: not used.
- Aircraft-rotation features: not supported by the available Kaggle file (no tail identifier).
- Network-feature ablation for severe delay: tested and rejected on validation (lower PR-AUC than base LightGBM).
- SMOTE or other resampling across time: not used; class weighting and threshold/ranking evaluation preferred.

## Planned final evaluation (2023)

Once these decisions are accepted:

1. Freeze:
   - Target definitions.
   - Feature sets.
   - Model families and key hyperparameters.
   - Calibration strategy (for severe delay).
2. Train each selected model on **development + validation** (2019–2022).
3. Evaluate once on the **2023 locked test split**:
   - Compute PR-AUC, Brier score, and ranking metrics for classification.
   - Compute MAE, RMSE, and severe-subset MAE for regression.
4. Produce:
   - A final model-selection report.
   - A concise technical summary for recruiters, clearly separating validation and final-test results.

## Limitations and appropriate use

- This project is a **retrospective Data Science portfolio study**, not an airline operational system.[1][6]
- All findings are specific to this dataset, time period, and split design.
- Results describe associations and predictive performance, not causal effects.
- The locked 2023 evaluation provides one out-of-time test; it does not guarantee future performance under different operating conditions.
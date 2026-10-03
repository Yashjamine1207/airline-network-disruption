# Phase 4 Validation Report — Airline Network Disruption Intelligence

**Purpose:** Summarise validation-period evidence for delay, cancellation, and severe-delay prediction using only 2019–2021 training and 2022 validation. The locked 2023 test split is not used in this report.

**Data:** Kaggle Flight Delay and Cancellation Dataset (2019–2023), restricted here to 2019–2022 for modelling. No NOAA weather or aircraft-rotation features are used in this flight-only configuration.[1][6]

## Targets and cohorts

- **Cancellation:** All scheduled flights with an observed `cancelled_target` label.
- **Severe arrival delay:** Completed flights with `arrival_delay_minutes ≥ 120` (project-defined threshold).
- **Arrival-delay regression:** Completed flights with a finite `arrival_delay_minutes` outcome.

## Baselines

Constant and global baselines were fitted on development rows and evaluated on 2022 validation:

- **Cancellation prevalence baseline:** PR-AUC equal to the validation event rate (≈ 0.0268).
- **Severe-delay prevalence baseline:** PR-AUC equal to the validation event rate (≈ 0.0267).
- **Global median arrival delay:** MAE ≈ 22.85 minutes.

These establish the minimum useful performance for rare-event classification and regression.

## Model comparison (validation 2022)

Key results from `phase4_validation_summary.csv` and `phase4_model_comparison.png`:

- **Cancellation (unweighted Logistic Regression, base features):**
  - PR-AUC ≈ 0.0417
  - Brier score ≈ 0.0263
- **Severe delay (base LightGBM, base features):**
  - PR-AUC ≈ 0.0453
  - Brier score ≈ 0.0259
- **Severe delay (isotonic-calibrated LightGBM):**
  - Mean predicted probability aligned to the validation event rate (≈ 0.0267).
  - Brier score improved slightly relative to uncalibrated outputs.
- **Arrival delay (Ridge Regression, base features):**
  - MAE ≈ 22.85 minutes, slightly better than Linear (≈ 23.04) and Elastic Net (≈ 23.31).

Linear and regularised regression models perform similarly on MAE, with Ridge marginally best among them.

## Calibration

Isotonic calibration of severe-delay LightGBM probabilities:

- Shifted the mean predicted probability from ≈ 0.0221 to ≈ 0.0267, matching the validation severe-delay rate.
- Reduced the Brier score modestly, indicating better probability reliability without changing the ranking substantially.

Calibrated probabilities should be preferred for any threshold-based alert policy, but the locked 2023 split must be used to confirm that calibration generalises.

## Ranking capacity

The `phase4_severe_delay_ranking_capacity.csv` table reports precision, recall, and events captured at top 0.5%, 1%, 5%, and 10% of validation flights for both base and calibrated LightGBM.

Key pattern:

- At low review capacities (0.5–1%), both models concentrate severe-delay events well above the baseline event rate, with calibrated probabilities offering more reliable risk scores even if PR-AUC changes little.

These results support using the model as a **ranking and prioritisation tool**, not as a hard classifier with a single threshold.

## Limitations of this validation report

- All results are for **2022 validation only**. The 2023 final test period remains untouched and must be used before claiming final model performance.
- This configuration is **flight-only**: no weather, no aircraft-rotation, no network features beyond the tested schedule-count and network ablation (which did not improve severe-delay PR-AUC).
- Calibration was fitted on the same validation data used for early stopping; an independent test period is needed to assess overfitting in the calibrator.
- Results are descriptive of this dataset and split design; they are not operational guarantees for future periods or other airlines.

## Next steps

1. Lock the preferred model family, features, and calibration strategy using this validation evidence.
2. Evaluate the locked configuration once on the 2023 final test split.
3. Produce a final model-selection report and a concise technical summary for recruiters, clearly separating validation findings from final-test results.

**Files referenced in this report:**

- `reports/tables/phase4_validation_summary.csv`
- `reports/tables/phase4_severe_delay_ranking_capacity.csv`
- `reports/figures/phase4_model_comparison.png`
- `reports/tables/phase4_severe_delay_lightgbm_isotonic_calibration.csv`
- `reports/tables/phase4_ridge_regression_validation.csv`
- `reports/tables/phase4_cancellation_logistic_unweighted_validation.csv`
- `reports/tables/phase4_severe_delay_lightgbm_validation.csv`
# Phase 4 Final Summary — Airline Network Disruption Intelligence

**Scope:** Flight-only modelling using the Kaggle flight dataset (2019–2023). No NOAA weather or aircraft-rotation features are used in this configuration.

**Chronological design:**

- Development: 2019–2021  
- Validation: 2022  
- Final test: 2023 (partial year)

## Validation highlights (2022)

From `phase4_validation_summary.csv`, `phase4_model_comparison.png`, and `phase4_validation_report.md`:

- **Cancellation (Logistic Regression, base features, unweighted):**
  - PR-AUC ≈ 0.0417 vs prevalence baseline ≈ 0.0268.
  - Unweighted model had a much better Brier score than the class-weighted variant.
- **Severe arrival delay (LightGBM, base features):**
  - PR-AUC ≈ 0.0453 vs prevalence baseline ≈ 0.0267.
  - Isotonic calibration on validation probabilities aligned mean predicted probability with the event rate and slightly improved the Brier score.
- **Arrival-delay regression (Ridge, base features):**
  - MAE ≈ 22.85 minutes, marginally better than Linear and Elastic Net.

Full details are in `phase4_validation_report.md` and `phase4_model_decisions.md`.

## Final test (2023) — reported results

### Severe arrival delay (corrected evaluation)
File: `phase4_severe_delay_2023_corrected.csv`
- **Model:** LightGBM on full base features (numeric + categorical encoded as integers).
- **Training:** 2019–2021.
- **Calibration:** Isotonic regression fitted on 2022 predictions (disjoint from training).
- **Test:** 2023 only; no fitting or calibration used 2023 labels.
Key 2023 metrics:
- Test rows: **454,413**  
- Test event rate: **0.031874**  
- PR-AUC: **0.057415**  
- ROC-AUC: **0.640029**  
- Brier score: **0.030597**  
- Mean predicted probability: **0.026692**
This is the **only** severe-delay 2023 result that should be reported.

### Cancellation and arrival delay (2023)

Earlier scripts mistakenly fitted models or calibration using 2023 labels, violating the holdout principle. Those outputs have been quarantined as:

- `phase4_invalid_test_run_do_not_report.csv`
- `phase4_invalid_second_test_run_do_not_report.csv`

No valid cancellation or arrival-delay 2023 results are reported in this project. A future rerun could regenerate these with a strictly correct design, but for this portfolio they remain unreported.

## Limitations

- This is a **retrospective Data Science portfolio study**, not an airline operational system.
- The severe-delay 2023 evaluation uses a single out-of-time test; it does not guarantee future performance.
- Cancellation and arrival-delay final-test results are absent due to earlier implementation errors; only validation results are available for those targets.
- No weather, aircraft-rotation, or advanced network features are included in this flight-only configuration.

## Files

- Validation summary: `reports/tables/phase4_validation_summary.csv`  
- Model-comparison figure: `reports/figures/phase4_model_comparison.png`  
- Ranking capacity: `reports/tables/phase4_severe_delay_ranking_capacity.csv`  
- Validation report: `reports/evaluation/phase4_validation_report.md`  
- Model decisions: `docs/decisions/phase4_model_decisions.md`  
- Corrected severe-delay 2023 result: `reports/tables/phase4_severe_delay_2023_corrected.csv`  
- Invalid runs (do not report): `reports/tables/phase4_invalid_test_run_do_not_report.csv`, `phase4_invalid_second_test_run_do_not_report.csv`
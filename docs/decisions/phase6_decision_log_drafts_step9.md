# Phase 6 and 7A decision log drafts, Step 9

Three things to copy into `docs/decisions/decision_log.md`:

1. An **Outcome** block to append under DEC-021, with the numbers from your own Step 8 run.
2. The new entry **DEC-022** (compact Transformer), written before that model has been trained.
3. The new entry **DEC-023** (calibration protocol), written before the calibration script has been run on your data.

All stay "Proposed" until you have read them and changed the status yourself. DEC-022 and DEC-023 have no
Outcome block yet. Send me the run output and I will write them.

---

## DEC-021 outcome (append under DEC-021 after the Step 8 runs)

**Runs:** three fit-row variants, tuned LightGBM with seeds 42, 43 and 44; `mlp` and `lstm` with the default
configuration and seeds 42, 43 and 44. Same early-stopping rows (337,156) and same 2022 validation rows
(667,616) in every run. Nothing from 2023 was loaded.

Fit rows by UTC calendar year of the prediction time:

| Year | Rows | Events | Event rate | Share of fit rows |
| --- | --- | --- | --- | --- |
| 2018 | 1 | 0 | 0.0% | 0.0% |
| 2019 | 741,941 | 19,757 | 2.66% | 51.0% |
| 2020 | 449,920 | 4,866 | 1.08% | 30.9% |
| 2021 (to 30 June) | 262,490 | 4,712 | 1.80% | 18.0% |

The single 2018 row is a flight scheduled on 1 January 2019 local time whose UTC prediction time falls on 31
December 2018. It stays in the `full` and `matched` variants and does not matter.

`no2020` has 1,004,432 fit rows, and so does `matched`.

**LightGBM, PR-AUC on the second half of 2022** (three-seed mean, with the spread from the smallest to the
largest seed):

| Variant | Three-seed mean | Seed spread | Full-year mean |
| --- | --- | --- | --- |
| `full` | 0.04695 | 0.00036 | 0.04789 |
| `no2020` | 0.04662 | 0.00022 | 0.04715 |
| `matched` | 0.04614 | 0.00056 | 0.04742 |

Paired day-level bootstrap on seed 42, difference in PR-AUC with 95% interval:

| Comparison | Second half of 2022 | Full year 2022 |
| --- | --- | --- |
| `no2020` minus `full` | -0.00027 (-0.00100 to +0.00042) | -0.00081 (-0.00136 to -0.00019) |
| `no2020` minus `matched` | +0.00032 (-0.00048 to +0.00121) | -0.00017 (-0.00064 to +0.00058) |
| `matched` minus `full` | -0.00059 (-0.00116 to -0.00009) | -0.00064 (-0.00106 to -0.00030) |

**Rule result:** `no2020` does not beat `full` (interval includes zero on the second half; on the full year
`full` is better with the whole interval below zero). **2020 stays in the fit rows** for LightGBM.

**Neural networks, PR-AUC on the second half of 2022, seed 42, difference from the same architecture trained on
all fit rows:**

| Model | Variant | PR-AUC | Seed spread | Difference from full fit (95% interval) |
| --- | --- | --- | --- | --- |
| `mlp` | `no2020` | 0.04556 | 0.00089 | -0.00018 (-0.00107 to +0.00067) |
| `mlp` | `matched` | 0.04541 | 0.00092 | -0.00032 (-0.00102 to +0.00030) |
| `lstm` | `no2020` | 0.04538 | 0.00019 | -0.00001 (-0.00069 to +0.00082) |
| `lstm` | `matched` | 0.04420 | 0.00009 | -0.00119 (-0.00187 to -0.00045) |

Full-fit values for reference: `mlp` 0.04574, `lstm` 0.04539. The `no2020` versus `matched` comparison was
not run for the networks; for the LSTM the seed-42 point estimates are 0.04538 and 0.04420.

**Observations, not findings:**

* Nothing here says 2020 harms 2022. Removing it did not raise PR-AUC for any of the three models, and on the
  LightGBM full year it lowered it.
* Fewer rows do hurt. For LightGBM and the LSTM, `matched` is below `full` with the whole interval below zero.
  For the LSTM, removing the same number of random rows cost more (-0.00119) than removing 2020 (-0.00001).
  That fits the idea that the 2020 rows add little for the LSTM, but it is a comparison of point estimates, and
  the 2020 rows have a lower event rate (1.08% against 2.66% in 2019).
* Every network variant peaked at epoch 1 of 4 again. Removing 2020 did not change that. The early
  overfitting is not explained by 2020.
* The earlier suspicion from DEC-018 (dropping the calendar year raised 2022 PR-AUC) stands as a statement
  about the year column, not about 2020 rows.

**Limits of this result:** schedule-only features, one validation year, default network settings. The network
vocabularies and scalers were learned once from all fit rows, so an embedding for a level seen only in 2020
stays at its starting values in the `no2020` network runs (DEC-021 limits).

---

## DEC-022: Compact Transformer challenger

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-016, DEC-017, DEC-018, DEC-019, DEC-020, DEC-021

### Context

The scope names one compact Transformer encoder as the last challenger, after the LSTM and GRU. The LSTM and
GRU did not beat the MLP control or tuned LightGBM (DEC-020 outcome), and dropping 2020 did not change the
picture (DEC-021 outcome). A Transformer is added because the scope asks for it and because it reads the whole
48-hour lookback at once instead of step by step. It is not added because it is expected to win.

### Decision

The Transformer uses the same inputs, cohort, prediction time (scheduled departure UTC minus two hours),
splits, static branch, head, loss, optimiser, early stopping and comparison protocol as DEC-020. Only the
sequence layer changes.

| Part | Setting |
| --- | --- |
| Input | The same 16 three-hour bins with 3 channels and the same padding mask |
| Projection | One dense layer from 3 channels to `units` numbers per step (default 32) |
| Position | A learned position embedding for the 16 positions (0 is the oldest step, 15 the newest) |
| Block | One encoder block: 2-head self-attention, residual and LayerNorm, feed-forward layer of width `dense` (default 64), residual and LayerNorm |
| Masking | Padded steps are removed as attention keys, and excluded from the final average |
| Summary | Masked mean over the real steps. A row with no real step gets a summary of zero and is scored from the static inputs alone |
| Not used | No causal mask (every step is already in the past), no [CLS] token, no more than one block, no attention-weight explanations |
| Size | Default configuration adds 11,232 parameters to the MLP control (about 224,900 in total on your inputs, against 213,625 for the MLP and 220,281 for the LSTM) |

`units` is the model width and must divide by 2. Every value in the search space does. The Transformer adds no
tuning options of its own.

Runs: default configuration only, seeds 42, 43 and 44. No random draws. DEC-020 allowed up to five and left
them optional; the results so far make it unlikely they would change the verdict, and a Transformer gets no
larger budget than the models it is compared with.

### Decision rule, same as DEC-020 and fixed before the run

The Transformer is kept only if BOTH comparisons on the second half of 2022 have the whole 95% paired-bootstrap
interval above zero, and the gain is larger than the seed spread:

1. Transformer minus the MLP control.
2. Transformer minus tuned LightGBM.

The comparison with the LSTM (the recurrent model with the higher three-seed mean on the second half) is
reported because the scope asks for it. It cannot keep the Transformer alive on its own. A Transformer that
beats the LSTM but not the control has only beaten a model that was itself rejected.

### What I expect

My expectation is the same band as the other three families (within about 0.0015 PR-AUC of LightGBM). If it
lands there, that supports the reading that the schedule-only features set the ceiling, not that the
Transformer is badly built. That is an expectation, not a result.

### Limits

* Tests check that padded steps and their position embeddings cannot change a score, that a fully masked row
  scores finitely and does not lose its static inputs, and that a saved model reloads to the same scores.
  They do not show the design is optimal.
* One block and two heads is a compact design chosen in advance. A larger Transformer is not part of the scope.
* Same limits as DEC-020: validation year only, one sequence definition (origin airport schedule).

### Not decided here

* SHAP, feature-family ablation or masking studies for the networks (Phase 7A explainability).
* Whether any network is retained. That is decided by the rule above, then recorded in the final selection.

---

## DEC-023: Calibration protocol for the selected classifier

**Status:** Proposed
**Phase:** 7A
**Depends on:** DEC-014, DEC-015, DEC-018, DEC-019, DEC-020

### Context

The scope requires probabilities that can be trusted as probabilities, not only scores that rank flights. Two
facts make this matter here.

* The fit rows have a severe-delay rate of about 2.0% (29,335 events in 1,454,352 rows) and later years are
  higher (the prevalence table from Step 4 gives about 2.7% for 2022). A model trained on lower-rate years tends
  to under-predict later ones. That is calibration-in-the-large and it can be fixed after training.
* PR-AUC is about 0.047 (lift roughly 1.8 times the base rate), so most predicted probabilities are small and a
  level error is a large share of the value.

Nothing has been calibrated yet and no calibrated number exists.

### Decision

**Model.** The tuned LightGBM on `base_no_year` (DEC-018, DEC-019), refitted with seed 42. The script refits it
and stops if the 2022 full-year PR-AUC differs from the stored validation scores by more than 0.0005, so the
model being calibrated is the model already reported. If final selection picks a different model, calibration
is run again for that model.

**Methods.** Uncalibrated (the reference), Platt (logistic on the logit of the score) and isotonic.

**Experiment A (decides the method).** Fit each calibrator on the early-stopping rows (development, prediction
time from 2021-07-01 to 2021-12-31 UTC). They come after every fit row and before every 2022 row. Score 2022.
The first half of 2022 (before 2022-07-01 UTC) chooses candidates. The second half confirms. Quarter-by-quarter
results for 2022 are descriptive.

**Experiment B (diagnostic, never used to choose the method).** Fit on the first half of 2022 and score the
second half. It answers one question: is calibration data from 2021 too old?

**Rule 1, which method.** A calibrator is a candidate only if its Brier score on the first half of 2022 is lower
than the uncalibrated Brier score. A candidate is adopted only if the paired day-level bootstrap of
(Brier of the method minus Brier of uncalibrated) on the second half has its whole 95% interval below zero. If
both pass, isotonic is chosen only if it also beats Platt with the whole interval below zero. Otherwise Platt is
chosen. If nothing passes, the scores stay uncalibrated. Simple beats flexible unless the evidence is clear.

**Rule 2, how the final-test calibrator will be fitted.** If the chosen method fitted on the first half of 2022
beats the same method fitted on the 2021 rows, on the second half of 2022, with the whole 95% interval below
zero, recency matters. The final-test calibrator is then fitted on 2022 (the most recent labelled year before the
test). Otherwise it is fitted on the same kind of rows as in Experiment A. **Check this against DEC-015:** using
2022 as calibration data spends the validation year after the method has been chosen, which is only acceptable
if DEC-015 allows the validation year to be reused once model and method are locked.

**Metrics.** Brier score (primary), Brier skill against a constant equal to the event rate of the calibration
rows, calibration slope, calibration-in-the-large (the intercept with the slope fixed at 1), expected
calibration error in 10 equal-count bins, mean predicted probability against observed prevalence and their
ratio, reliability tables and figure, PR-AUC and ROC-AUC, and the number of distinct scores. The bootstrap
resamples days (1,000 resamples, seed 42).

**Ranking.** Rankings in the alert-policy step use the uncalibrated LightGBM score unless a table says otherwise.
Platt is strictly increasing, so it gives the same ranking and the same PR-AUC by construction, which is why
PR-AUC cannot choose between uncalibrated and Platt. Isotonic merges scores into ties and can lower PR-AUC.

**Guards (tested).** No row from 2023 is used. Calibration rows end strictly before evaluation rows start.
No row is in both sets. The config file cannot allow the final test.

### Limits

* 2019 and 2020 are training years for this model, so they cannot be scored on fresh data here. The regime
  evidence is 2022 by quarter. Checking calibration in the 2019 and 2020 regimes would need models refitted on
  earlier windows. That is not attempted.
* The early-stopping rows that calibrate also chose the number of trees. Their scores are slightly optimistic.
  This is a small overlap between calibration and model selection, not a use of validation or test labels.
* Good calibration in 2022 does not guarantee it in 2023. The final test reports calibration once, with the
  method locked here.
* Calibration changes probabilities, not the ranking of flights (Platt) or only slightly (isotonic ties). It
  cannot add signal. PR-AUC and top-K recall stay where the model put them.

### Not decided here

* Alert thresholds or review capacities (Phase 7B).
* SHAP and error analysis.
* Whether the calibrated model is the final selection.

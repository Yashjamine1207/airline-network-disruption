# Decision log drafts, Step 10 (copy into docs/decisions/decision_log.md)

Four blocks. The first two record the results of runs you already did (Step 9). The last two are new and stay
"Proposed" until you have read them. Nothing here uses 2023.

---

## DEC-022: Outcome (append to the DEC-022 entry)

**Status change:** Proposed to Accepted (the rule was fixed before the run and the run followed it).

Run: default configuration, seeds 42, 43, 44, no random draws, 224,857 parameters, about 245 s per seed
(about 60 s per epoch, two to three times the cost of an LSTM epoch on the same machine).

| Item | Result |
| --- | --- |
| Transformer PR-AUC, second half of 2022, seeds 42 / 43 / 44 | 0.04489 / 0.04524 / 0.04582 (spread 0.00093, mean 0.04532) |
| Tuned LightGBM, same rows | 0.04680 |
| Transformer minus LightGBM, second half | -0.00190, 95% interval [-0.00317, -0.00059] |
| Transformer minus MLP control, second half | -0.00084, 95% interval [-0.00185, -0.00004] |
| Transformer minus LSTM, second half | -0.00050, 95% interval [-0.00108, +0.00001] |
| Transformer minus LightGBM, full year 2022 | -0.00128, 95% interval [-0.00212, -0.00023] |
| Transformer minus MLP control, full year 2022 | -0.00026, 95% interval [-0.00085, +0.00047] |
| Brier score, second half | 0.02524 (LightGBM 0.02522) |
| ROC-AUC, second half | 0.63221 (LightGBM 0.63971) |

**Verdict: rejected.** The rule needed both required comparisons (against the MLP control and against tuned
LightGBM) to have the whole interval above zero. Both are wholly below zero on the clean second half. All three
seeds sit below LightGBM. The best epoch was epoch 1 of 4 in every run, and the fit-row PR-AUC climbed from about
0.041 to 0.067 while the early-stopping PR-AUC fell from 0.0497 to 0.0446, so the network overfits from the
first epoch. The comparison with the LSTM could not keep it alive and did not need to.

**What this does and does not show.** On this input (16 three-hour airport bins, three count channels, and the
static schedule features) a compact Transformer is no better than a plain MLP or an LSTM and costs the most to
train. It does not show that Transformers cannot help on other data, or with a longer or richer sequence.

---

## DEC-023: Outcome (append to the DEC-023 entry)

**Status change:** Proposed to Accepted for Rule 1. Rule 2 is discussed in DEC-024.

The refitted LightGBM (53 trees) reproduced the stored scores exactly (PR-AUC difference 0.000000, largest score
difference 0). Calibration rows: 337,156 rows, 8,154 events, 2.42% rate. Validation: 667,616 rows, 17,820
events, 2.67% (first half 2.74%, second half 2.60%).

| Second half of 2022 | Mean predicted | Predicted / observed | Brier | Calibration in the large | Slope | ECE | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uncalibrated | 0.01970 | 0.756 | 0.02522 | 0.288 | 0.938 | 0.00635 | 0.04680 |
| Platt (fitted on 2021 H2 rows) | 0.02428 | 0.932 | 0.02518 | 0.073 | 0.870 | 0.00232 | 0.04680 |
| Isotonic (fitted on 2021 H2 rows) | 0.02427 | 0.932 | 0.02519 | 0.073 | 0.839 | 0.00255 | 0.04509 |
| Platt fitted on 2022 H1 (diagnostic) | 0.02633 | 1.011 | 0.02518 | -0.011 | 0.908 | 0.00164 | 0.04680 |
| Isotonic fitted on 2022 H1 (diagnostic) | 0.02623 | 1.007 | 0.02518 | -0.007 | 0.862 | 0.00243 | 0.04538 |

Observed rate in the second half: 0.02604.

Rule 1 (paired day-level bootstrap, Brier of the method minus Brier of uncalibrated, second half):

| Comparison | Difference | 95% interval |
| --- | --- | --- |
| Platt vs uncalibrated | -0.000040 | [-0.000077, -0.000006] |
| Isotonic vs uncalibrated | -0.000030 | [-0.000068, +0.000005] |
| Isotonic vs Platt | +0.000010 | [+0.000002, +0.000018] |

**Verdict: Platt.** It improved the first half and the improvement was confirmed on the second half. Isotonic
did not clear the interval rule, was slightly worse than Platt with the whole interval above zero, collapsed
the 215,767 distinct scores of the full year to 124 and lowered PR-AUC by 0.0017 (second half).

**Read the size of the win.** The Brier improvement is 0.00004 on a base of 0.0252, about 0.16%. It is real
(the interval excludes zero) and practically negligible. What calibration actually fixed is the level: the
uncalibrated model predicted 76% of the events that happened, Platt 93%. The ranking is unchanged, so PR-AUC,
Precision@K and Recall@K are the same as before. Calibration does not add signal.

Rule 2 result as run: SAME_AS_HERE (recent rows not clearly better on Brier). See DEC-024 before you rely on it.

**Limits carried over.** The slope stayed below 1 after calibration (0.87 for Platt), so the spread of the
probabilities is somewhat too wide in 2022, and it differed between the calibration rows (Platt slope 1.078 on
2021 H2) and 2022. The model's spread is not stable across periods. 2019 and 2020 could not be checked because
they are training years for this model.

---

## DEC-024: Level shift and the final-test calibrator (post-hoc amendment to DEC-023 Rule 2)

**Status:** Proposed
**Phase:** 7A
**Depends on:** DEC-015, DEC-023

### Why this exists

Rule 2 was written before the run and said: fit the final-test calibrator on 2022 only if recent rows beat
older rows on Brier with the whole interval below zero. The interval was [-0.000014, +0.000010] for Platt, so
Rule 2 says "keep the same kind of rows". Followed literally, the final calibrator would be fitted on rows like
the 2021 second half.

Rule 2 could not have detected this, and that is a design flaw of mine, not a finding. A level error of 7% on a
2.6% event rate costs about (0.07 x 0.026)^2, roughly 0.000003 of Brier. The bootstrap interval for the
difference is about 0.000012 wide on each side. A gain that small is invisible to a Brier test. The level
itself is measured directly, and the direct measures all point one way:

| Calibrator fitted on | Predicted / observed | Calibration in the large | ECE |
| --- | --- | --- | --- |
| 2021 H2 rows | 0.932 | +0.073 | 0.00232 |
| 2022 H1 rows | 1.011 | -0.011 | 0.00164 |

The event rate has risen with time (2.42% in the calibration rows, 2.67% in 2022), and a calibrator fitted on
older rows carries the older rate.

### Proposal

Fit the final-test calibrator (Platt, method fixed by Rule 1) on all of 2022 (validation year, prediction time
2022-01-01 to 2022-12-31 UTC) with the model kept as trained on 2019 to 2021. Nothing from 2023 is used.

### This is post-hoc

I am proposing it after seeing the Step 9 numbers. That has a cost you should accept knowingly:

* There is no clean data left to test it. 2022 is spent on fitting, so the first test of the recency-fitted
  calibrator is the 2023 final evaluation, reported once, with the rule stated here in advance.
* The evidence for recency comes from a half-year gap inside 2022 (H1 to H2). The final test is a gap of one
  full year or more. Level drift over that gap is not tested.
* The improvement in Brier is not distinguishable from zero. The case rests on calibration in the large and the
  predicted-to-observed ratio, which are direct measures of the quantity being fixed but were not the
  pre-registered criterion.

### Open question that decides the details

Is the final model itself refitted on 2019 to 2022 before the 2023 test, or kept as trained on 2019 to 2021?
The scope keeps 2022 for validation and model selection. If the model is refitted on 2019 to 2022, the
calibrator needs rows the refitted model did not see, and the split has to be redesigned (for example a
calibration window in the second half of 2022). This draft assumes the model is NOT refitted. The final
protocol decision is still to be made and must be checked against DEC-015, whose exact wording I cannot see from
here. If DEC-015 forbids reusing the validation year for fitting anything after model selection, keep Rule 2 as
written and report the 0.93 ratio as a known limit.

### If you decline this

Keep Rule 2 as run. The final calibrator is fitted on rows like the 2021 second half, and the report says the
level correction is expected to be incomplete because the event rate drifts upward.

### Not decided here

Whether the final model is refitted; alert thresholds (Phase 7B).

---

## DEC-025: Explainability, error analysis and sequence perturbation protocol (Step 10)

**Status:** Proposed
**Phase:** 7A
**Depends on:** DEC-014, DEC-018, DEC-019, DEC-020, DEC-023

### Decision

**Model explained.** The tuned LightGBM on `base_no_year`, seed 42, refitted by the script and checked against
the stored scores (the run stops if the PR-AUC differs by more than 0.0005). The model and feature version are
locked for this purpose. If the final selection picks another model, the explanation is redone for it.

**SHAP.** LightGBM's own TreeSHAP (`pred_contrib`), in log-odds, on the second half of 2022 (the rows nothing
was tuned on). The contributions are checked to add up to the model's raw output (tolerance in the code, gap
reported). Reported: mean absolute SHAP by feature and by feature family, direction tables for numeric features
(by quintile) and for categorical levels, a beeswarm figure on a random 4,000 rows. SHAP values describe how the
model reaches its score. They are associations learned from 2019 to 2021, not causes, and say nothing about
weather, aircraft or crews.

**Case studies.** Picked by a fixed rule from score and label only, never by looking at the explanation: true
positives (top 1% and severe), false positives (top 1% and not severe), false negatives (severe, outside the
top 10%), low-risk correct (not severe, lower-half score), and the most confident mistakes of each kind. The four main
categories take the cases at the 25th, 50th and 75th percentile of score inside the category. The two
high-confidence categories take the three most extreme cases (the highest-scoring non-events and the
lowest-scoring events). The local explanation figure draws the median case of four categories. The complete table is written so nobody has
to trust the drawn cases. The realised arrival delay is added to the table for description after the pick.

**Error breakdown.** All of 2022, using ONE global cutoff for the top 1% and for the top 10% (never a cutoff
per group). Dimensions: quarter, carrier, origin airport (top 25 by volume plus the rest), route volume in the
fit period, scheduled departure hour band, day of week, missing or unseen values, score decile, and, for events
only, the size of the delay. PR-AUC is reported only for a group with at least 30 events and 30 non-events. The
first half of 2022 chose the LightGBM settings, so the full-year table slightly favours the model.

**Sequence perturbation (LSTM first, GRU optional).** The saved seed-42 network scores the second half of 2022
once as trained and once after each change to the lookback: lookback removed, lookbacks shuffled between flights
in the same batch, only the newest 1, 4 or 8 bins kept, and each quarter of the lookback hidden in turn. Every
change is compared with the unchanged run by a paired day-level bootstrap on PR-AUC. The unchanged run must
reproduce the stored predictions. The whole interval has to sit on one side of zero for a statement of reliance,
and a change below 0.0005 PR-AUC is called tiny even if the interval excludes zero (a display rule, not a
selection rule).

### Limits

* A perturbation drop describes the trained network, not the airline system. An empty or shuffled lookback is
  something the network never saw in training, so a drop is an upper bound on what the lookback is worth.
* One saved seed. The seed spread for the networks is about 0.001 PR-AUC, so changes of that size are noise.
* 2019 and 2020 regimes cannot be scored for a model trained on them, so regime-level error analysis covers
  2022 by quarter only.
* Sequence availability cannot be compared in 2022 because every 2022 flight has a full lookback. The table
  says so.
* Regression case studies and airport recovery episodes need the regression model and the Phase 5 episode
  table. They are not part of this step and are listed as open.
* Attention weights are not used or reported.

### Not decided here

Alert policy and utility scenarios (Phase 7B); the final model choice.

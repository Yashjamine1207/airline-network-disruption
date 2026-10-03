# Decision log drafts, Step 11 (copy into docs/decisions/decision_log.md)

Three blocks. The first records the real Step 10 results, the second adds evidence to DEC-024, the third is the
new protocol for the review scenarios. Nothing here uses 2023.

---

## DEC-025: Outcome (append to the DEC-025 entry)

**Status change:** Proposed to Accepted.

Run on the tuned LightGBM (53 trees, reproduced the stored scores exactly) and the saved seed-42 LSTM.

### What the tabular model leans on (SHAP, second half of 2022, 341,232 rows, additivity gap 8.9e-15)

| Feature | Mean absolute SHAP (log-odds) | Share |
| --- | --- | --- |
| carrier_identifier | 0.2138 | 24.8% |
| scheduled_departure_hour_local | 0.1516 | 17.6% |
| scheduled_departure_month | 0.1449 | 16.8% |
| destination_airport | 0.1189 | 13.8% |
| origin_airport | 0.0957 | 11.1% |
| scheduled_arrival_hour_local | 0.0957 | 11.1% |
| route | 0.0402 | 4.7% |
| scheduled_departure_day_of_week | 0.0025 | 0.3% |
| scheduled_departure_minute_local, distance_miles | 0.0000 | 0.0% |

By family: calendar and clock 45.7%, airport 24.9%, carrier 24.8%, route 4.6%, trip length 0%.

These are associations the model learned from 2019 to 2021. They are not causes of delay.

**Limitation to carry forward.** Month is the third most important feature. The model learned it from three
years, two of which (2020 and 2021) were abnormal, so each month level rests on three observations confounded
with the operating regime. It is the feature least likely to transfer. The effect is visible in 2022: with one
global cutoff, recall at the top 1% is 0.8% in Q1, 4.5% in Q2, 6.5% in Q3 and 0.9% in Q4, and recall at the
top 10% is 13.6%, 28.7%, 33.4% and 9.5%. A global top-K list is mostly a list of summer flights. The
scenarios in DEC-026 therefore also report capacity within each month and each day.

### Error analysis, all of 2022 (first half chose the LightGBM settings, so this slightly favours the model)

| Cut | Result |
| --- | --- |
| Quarter, PR-AUC | Q1 0.0420, Q2 0.0544, Q3 0.0527, Q4 0.0403 (prevalence 2.55%, 2.90%, 2.64%, 2.57%) |
| Quarter, calibrated predicted / observed (Platt fitted on 2021 H2) | 0.915, 0.929, 1.032, 0.825 |
| Score decile, observed rate | 0.89% in the lowest, 5.85% in the highest (overall 2.67%) |
| Highest decile | holds 3,905 of the 17,820 events (21.9%) |
| Route volume in the fit period | PR-AUC 0.0451 for routes with 1,000 or more flights, 0.0484 for 100 to 999, 0.0514 for 1 to 99, 0.0508 for unseen routes (6,110 flights). No loss on unseen routes |
| Delay size, events caught by the global top 10% | 22.2% (120 to 179 min), 22.4% (180 to 299), 17.0% (300 to 599), 25.7% (600 or more) |
| Delay size, events caught by the global top 1% | 3.3%, 3.5%, 2.4%, 3.7% |

The model does not find the worst delays better than the mild ones. With schedule-only features it has no way
to know which delays will be long.

Calibration is not stable across seasons. The predicted-to-observed ratio moves between 0.83 and 1.03 by
quarter, so a single year-level calibrator leaves a seasonal miscalibration of roughly 10 to 15%.

### Sequence perturbation, LSTM seed 42, second half of 2022 (8,887 events, 100 bootstrap resamples)

The reloaded network reproduced its stored predictions (PR-AUC 0.045390, largest score difference 7e-9).

| Change to the lookback | PR-AUC | Change | 95% interval | Reading |
| --- | --- | --- | --- | --- |
| none (baseline) | 0.04539 | | | |
| lookback removed | 0.04425 | -0.00114 | [-0.00255, -0.00014] | relies on it |
| lookbacks shuffled between flights | 0.04174 | -0.00365 | [-0.00569, -0.00206] | relies on it |
| newest 1 bin only | 0.04515 | -0.00024 | [-0.00101, +0.00057] | no clear change |
| newest 4 bins only | 0.04504 | -0.00035 | [-0.00108, +0.00027] | no clear change |
| newest 8 bins only | 0.04552 | +0.00013 | [-0.00011, +0.00037] | no clear change |
| oldest quarter (bins 0 to 3) hidden | 0.04538 | -0.00001 | [-0.00012, +0.00009] | no clear change |
| second-oldest quarter hidden | 0.04544 | +0.00005 | [-0.00024, +0.00039] | no clear change |
| second-newest quarter hidden | 0.04475 | -0.00064 | [-0.00156, +0.00001] | no clear change |
| newest quarter (bins 12 to 15) hidden | 0.04407 | -0.00132 | [-0.00219, -0.00050] | relies on it |

Reading: the LSTM does use its lookback, and what it uses is the most recent stretch (the newest 12 hours). The
older half (more than 24 hours back) carries nothing it uses: keeping only the newest 8 of 16 bins loses
nothing. The shuffle result is larger than the removal result partly because a shuffled lookback comes from a
different airport and can be on the wrong size scale for the flight's own airport, so it is not clean
evidence of temporal information.

**Reliance is not value.** The network depends on the lookback, yet its PR-AUC on this half (0.04539) is no
better than the MLP control that never sees it (0.04574). The information it uses is either already carried by
the static features or not usable beyond them. DEC-020's comparison against the control is the test of
value, and it was not passed.

Every 2022 flight has a full 16-bin lookback, so performance by sequence availability cannot be compared.

Still open: regression case studies and airport recovery episodes (need the regression model and the Phase 5
episode table). The carrier, airport and hour tables are in `error_breakdown.csv` and go into the error
analysis report.

---

## DEC-024: Addendum (append to the DEC-024 entry)

The Step 10 run adds one fact that matters for the proposal. Fitting the calibrator on all of 2022 fixes the
annual level but not the seasonal one: with a calibrator fitted on 2021 H2 the ratios were 0.915, 0.929, 1.032
and 0.825 by quarter. Refitting on the whole of 2022 would scale all of them up by about 8%, giving roughly
0.99, 1.00, 1.11 and 0.89. The final test covers January to August 2023 (Q1, Q2, and July and August), so the
Q4 shortfall does not apply to it, and the summer months are the ones a 2022-wide calibrator would over-predict.
Expect the final calibration to be within about 10 to 15% by month whichever way DEC-024 is decided. That is a
property of a model whose seasonal pattern was learned from three unusual years, not something calibration can
repair.

---

## DEC-026: Retrospective ranked analyst-review scenarios (Step 11)

**Status:** Proposed
**Phase:** 7B
**Depends on:** DEC-014, DEC-015, DEC-018, DEC-019, DEC-023, DEC-024, DEC-025

### Decision

**Scope of the claim.** These are retrospective readings of a ranking: if someone could look at the top K% of
flights, how many events would be among them. They are not a procedure for running an airline, they model no
action taken after a review, and they use no cost of a real review or a real missed event.

**Targets.** Severe delay (arrival delay of 120 minutes or more, the locked model and cohort). Cancellation
as a second target with the same LightGBM settings and feature set. Those settings were tuned for severe
delay, so the cancellation results are labelled a baseline, not a tuned model. Neither target has been through a
cancellation-specific tuning step and none is added.

**Ranking score.** The uncalibrated LightGBM probability. Platt calibration is monotone and gives the same
ranking. No probability threshold is used in this step. Thresholds and confusion matrices belong to the final
evaluation and will use the calibration locked by DEC-023 and DEC-024.

**Capacities.** Top 0.5%, 1%, 5% and 10%. The number reviewed in a group of n flights is ceil(K x n), at least
1, so the reviewed total is slightly more than K% when groups are small. Ties break by row order.

**Three ways to apply a capacity.** Whole year (one ranking over the year), within each month, within each
UTC prediction day. Comparing them separates ranking skill from seasonal level: a score that only knows which
month is busy looks good over the whole year and is no better than chance within a day (tested).

**Regimes by expanding-window backtest.** One LightGBM per evaluation year, each scored on a year it never
saw: 2020 (fit 2019-01 to 2019-06, early stopping 2019-07 to 2019-12), 2021 (fit 2019-01 to 2020-06, early
stopping 2020-07 to 2020-12) and 2022 (the locked model, must reproduce the stored scores). 2019 cannot be
evaluated (no earlier data). 2023 cannot be an evaluation year (locked). Every model uses the same settings,
seed and feature set. The three numbers are three different models, not one model tested three times.

**Subgroups (2022).** Quarter, carrier, origin airport (25 largest by volume, rest pooled), scheduled local
departure hour band, missing or unseen values. One global cutoff per capacity over the whole year. For every
group the table shows the share of its flights that would be reviewed (alert rate), precision and recall, so
concentration is visible.

**Uncertainty.** 95% percentile intervals for precision and recall from a bootstrap that resamples whole UTC
prediction days (500 resamples, seed 42), with the flagged set held fixed.

**Illustrative penalty table.** A missed event is assumed to cost r false alerts, for r = 1, 2, 5, 10, 20, 50
and 100. Net reduction against no review is r x captured minus false alerts, and is compared with a random
review of the same size. These weights are assumptions. They are never reported as airline costs and the
measured counts stay in their own columns. The break-even ratio, false alerts divided by events captured, is
computed from the measured counts with no assumption: below that ratio the review loses.

### Limits

* 2022 chose the LightGBM settings and the calibration method, so it is validation evidence, not a clean test.
* Backtest years use models trained on less data for the earlier years and are not comparable as if one model.
* The month feature was learned from three unusual years (DEC-025). The whole-year scheme benefits from it and
  the by-day scheme does not, which is the reason both are reported.
* The alert list is concentrated by carrier and evening departure (DEC-025 and the group tables). A review
  capacity that looks fair by volume is not spread evenly across carriers or airports.
* Nothing here says a flagged flight could be acted on. No action or cost of acting is modelled.

### Not decided here

Probability thresholds and confusion matrices, the final model selection, and the single 2023 evaluation.

# Decision log drafts, Step 12 (copy into docs/decisions/decision_log.md)

Two blocks. The first records the real Step 11 results. The second is the protocol for the one-time final test and
is enforced by the lock file, not just written down. Nothing here uses 2023 results; no 2023 result exists yet.

---

## DEC-026: Outcome (append to the DEC-026 entry)

**Status change:** Proposed to Accepted. The protocol was followed and no choice was made from the results.

Each evaluation year is a different model trained on less data for the earlier years. They are not one model
tested three times. The 2022 model is the locked one and reproduced the stored scores exactly.

### Severe delay (arrival delay of 120 minutes or more)

| Year | Trees | Event rate | PR-AUC | Lift | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| 2020 (fit 2019-01 to 2019-06) | 12 | 1.08% | 0.01375 | 1.27 | 0.589 |
| 2021 (fit 2019-01 to 2020-06) | 123 | 2.15% | 0.04049 | 1.89 | 0.646 |
| 2022 (locked model) | 53 | 2.67% | 0.04793 | 1.80 | 0.643 |

Review capacity in 2022, ranking by the uncalibrated score, 95% day-bootstrap intervals in brackets:

| Capacity | Scheme | Reviewed | Events caught | Precision | Recall | Lift | Break-even ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5% | whole year | 3,339 | 318 | 9.5% [7.9, 11.3] | 1.8% | 3.57 | 9.5 |
| 1% | whole year | 6,677 | 588 | 8.8% [7.5, 10.2] | 3.3% | 3.30 | 10.4 |
| 5% | whole year | 33,381 | 2,262 | 6.8% [6.1, 7.5] | 12.7% | 2.54 | 13.8 |
| 10% | whole year | 66,762 | 3,905 | 5.9% [5.4, 6.4] | 21.9% | 2.19 | 16.1 |
| 1% | within each day | 6,857 | 543 | 7.9% [6.9, 9.0] | 3.1% | 2.97 | 11.6 |
| 10% | within each day | 66,927 | 3,803 | 5.7% [5.2, 6.2] | 21.3% | 2.13 | 16.6 |

The break-even ratio is measured, with no assumption: false alerts divided by events caught. A missed event has to be
worth that many false alerts for the review to pay off.

**What this says.** The seasonal part of the ranking is small. Reviewing within each day keeps nearly all of the
lift (2.97 against 3.30 at 1%, 2.13 against 2.19 at 10%), so the model ranks flights inside a day and is not only a
month detector. A review of the top 10% still catches about a fifth of the severe events, so four fifths are missed.
In 2020 the model is weak: PR-AUC lift 1.27, top-K lift 1.1 to 1.4 over the whole year and 1.5 to 2.1 inside a month or
a day, and break-even ratios of 44 to 82 to 1. That is a distribution-shift result, not a bug: it was trained on six
months of 2019.

**Where the alerts land (2022, one global cutoff).**
* By time of day. Flights departing 14:00 to 23:59 get 19% of their flights flagged at the top 10%. Flights before
  10:00 get 0.4% to 0.7%. Flights departing 06:00 to 09:59 hold 18% of the events (3,268 of 17,820) and the
  top 10% review catches 1.7% of them. The model is a late-in-the-day detector.
* By quarter. The top 1% flags 0.28% of Q1 flights and 0.31% of Q4 flights, against 1.43% in Q2 and 1.89% in Q3.
* By airport. EWR has 8.7% of its flights flagged at the top 1% (recall 19.6%); LAS, LAX, PHX, SEA, SFO and SLC have none.
* By carrier. DOT_20409 has 7.1% of its flights flagged at the top 1%; 5 of 17 carriers have none.
Capacity is not spread by volume. A reader who is told "1% of flights" has to be told these concentrations as well.

**Illustrative penalty table (assumption only).** A missed event is assumed to cost r false alerts. At r = 10 only the
0.5% review is above zero (+159 units). At r = 20 all four are (+3,339, +5,671, +14,121, +15,243). These weights are
not airline costs.

### Cancellation (same settings, NOT tuned for this target)

| Year | Event rate | PR-AUC | Lift | ROC-AUC |
| --- | --- | --- | --- | --- |
| 2020 | 6.00% | 0.0687 | 1.14 | 0.535 |
| 2021 | 1.72% | 0.0171 | 0.99 | 0.528 |
| 2022 | 2.68% | 0.0300 | 1.12 | 0.566 |

In 2022 a whole-year review of the top K% catches fewer cancellations than a random review (precision 1.8% to 2.1%
against an event rate of 2.68%, lift 0.68 to 0.78). Inside each month it does better than random (lift 2.75, 2.54,
2.04, 1.80 at 0.5%, 1%, 5%, 10%). The cause is visible in the table of alerts: Q1 2022 had a cancellation rate of
4.06% against 1.9% to 2.5% in the other quarters, and the model, which learned its month pattern from 2019 to
mid-2021, flagged almost nothing in Q1 (0.05% of flights at the top 1%) and put its alerts in Q2 (3.9%). The likely
cause of the Q1 rate is the winter 2022 operating conditions; that was not tested.

**Verdict.** There is no cancellation model here, only a labelled baseline with no demonstrated whole-year skill. It
goes through the final test as a baseline, ranking only, and no probability or threshold claim is made for it.

### Limits

* 2022 chose the LightGBM settings and the calibration method, so it is validation, not a clean test.
* The month feature was learned from three unusual years. For severe delay it happened to point the right way in
  2022. For cancellation it pointed the wrong way. It is the feature least likely to transfer.
* No cost of reviewing a flight, no cost of a missed event and no action after a review is modelled.

---

## DEC-027: Final-test protocol and lock (Step 12)

**Status:** Proposed
**Phase:** 7B
**Depends on:** DEC-014, DEC-015, DEC-018, DEC-019, DEC-022, DEC-023, DEC-024, DEC-025, DEC-026

### Decision

The final test (2023-01-01 to 2023-08-31) is run once, from a lock file written before any 2023 row is read
(`scripts/run_phase7b_final_lock.py`, then `scripts/run_phase7b_final_test.py --confirm USE-2023-ONCE`). The lock is
sealed with a SHA-256 over its content, records a fingerprint of every file the result depends on, and is committed
to Git before the test is run.

**Model.** The tuned LightGBM, `base_no_year`, seed 42, as validated: fit 2019-01 to 2021-06, early stopping 2021-07
to 2021-12. It is **not refitted on 2022.** The model that was validated is the model that is tested. A refit on
2019 to 2022 would test a model whose tree count and calibration were never validated. The cost is a gap: the model
has seen nothing after 2021-06, so the test is 18 to 26 months after its training data. That gap goes in the report
as a limitation.

**Calibration.** The method is the one DEC-023 chose (Platt). The final calibrator is fitted on the early-stopping
rows (2021-07 to 2021-12), which is DEC-023 Rule 2 as it ran, unless the lock was written with
`--calibration-fit validation_2022`, which is DEC-024. Whichever is the primary is fixed in the lock. The other one is
reported beside it and labelled a diagnostic. Neither is chosen after the test. *Edit this paragraph to match the
flag you used.*

**Known before the test.** The Phase 3 audit recorded the 2023 severe-delay rate as about 3.2% (it is stated in the
docstring of `evaluation/classification.py`; check it against the audit). That is above 2022 (2.67%) and above the
calibration rows (2.42%). Expect the calibrated probabilities to run low under either calibrator. This is stated
here so nobody can later call it a surprise or a finding.

**Targets.** Severe delay is the selected model. Cancellation is an untuned baseline (DEC-026): ranking only, no
calibration, no thresholds.

**Operating points, learned on 2022 and applied unchanged to 2023.** The score cutoffs for the top 0.5%, 1%, 5% and
10% of 2022 flights, and the F1-optimal cutoff on 2022. The 2023 alert rate at each cutoff is a result. If the
F1-optimal cutoff flags more than half of all flights on 2022, the report says it is not a usable alert rule.

**Review scenarios.** Capacities 0.5%, 1%, 5%, 10% as in DEC-026: over the whole period, within each month and within
each day. Rankings use the uncalibrated score. The final period is eight months, so "whole year" means whole period.

**Challengers.** The LSTM, GRU, Transformer and MLP control are not scored on 2023. They were rejected on 2022.
Scoring them on the final test would turn it into a second selection round.

**The only pass/fail claims.** Judged the same way for both targets, with 1,000 day-level bootstrap resamples (seed 42):
* C1: the lower end of the 95% interval of PR-AUC lift is above 1.0.
* C2: the whole 95% paired interval of PR-AUC(LightGBM) minus PR-AUC(best history baseline) is above 0. The baseline
  is whichever of carrier, origin, destination and route rate has the highest 2022 PR-AUC, fixed in the lock.
* C3: the lower end of the 95% interval of Precision@10% with a fixed capacity inside every day is above the 2023
  event rate.
Everything else in the final report is descriptive.

**Run rules.** The script refuses if the lock was edited, came from a smoke run, or any fingerprinted file changed.
It refits the model and stops unless it has the locked tree count, 2022 PR-AUC and sum of 2022 scores, before it reads
2023. The 2023 scores are saved on the first run. A later run is a reporting rerun: it must reproduce those scores or
it stops, so a rerun can fix reporting but cannot change the model. This is a discipline tool, not security.

### After the test

Nothing changes. If a claim fails the report says it failed. No model, feature, setting, calibrator or threshold is
adjusted after 2023 has been seen. A new model built after seeing 2023 cannot use 2023 as its test.

### What the test cannot show

* One regime, one period (January to August 2023). No regime comparison inside the final test.
* Schedule-only features: no same-day airport state, no weather, no aircraft identity.
* No regression result and no airport recovery result. No locked regression model is part of this lock.
* No cost, no action and no operating authority. The penalty table is an assumption.

### Not decided here

Whether a regression model is evaluated on the final test (it would need its own lock). The text of the final
model-selection record, which is written after the test from its results.

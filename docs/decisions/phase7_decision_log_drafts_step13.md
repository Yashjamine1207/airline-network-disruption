# Decision log drafts, Step 13 (copy into docs/decisions/decision_log.md)

Four blocks, written after the one-time final test. DEC-027 gets its outcome. DEC-028 to DEC-030 are new.
Items in square brackets marked FILL are values only your machine has. Fill them in before you commit.

---

## DEC-027: Outcome (append to the DEC-027 entry)

**Status change:** Proposed to Accepted. The protocol in the lock was followed. Nothing was chosen or changed after
2023 was read.

**Evidence that the lock came first.** Lock `e9d60fd5184a9e38...` was written 2026-10-01T08:57:11+00:00 and committed
as `7ee01c9` at [FILL: output of `git log -1 --format=%cI 7ee01c9`]. The consumed marker
`reports/final/final_test_consumed.json` records the first scoring of 2023 at [FILL: `first_run_utc` from that file].
The commit time is earlier than the scoring time.

**Model that was tested.** Tuned LightGBM, feature set `base_no_year`, seed 42, fitted on 2019-01 to 2021-06, early
stopping on 2021-07 to 2021-12, not refitted on 2022. The refit reproduced the stored 2022 scores with a difference of
0.000000 before any 2023 row was read. Platt calibrator fitted on the early-stopping rows (DEC-023 Rule 2 as run).

### Severe delay (arrival delay of 120 minutes or more), 2023-01-01 to 2023-08-31

454,413 flights, 14,484 severe delays (3.187%). 95% day-bootstrap intervals in brackets.

| Period | PR-AUC | Lift | ROC-AUC | Carrier-rate baseline PR-AUC | Difference |
| --- | --- | --- | --- | --- | --- |
| 2022 validation | 0.0479 [0.0437, 0.0524] | 1.80 | 0.643 | 0.0367 | +0.0112 [0.0084, 0.0143] |
| 2023 final test | 0.0591 [0.0540, 0.0652] | 1.85 [1.73, 1.98] | 0.656 | 0.0434 | +0.0157 [0.0121, 0.0197] |

The other history baselines on 2023: origin 0.0378, destination 0.0379, route 0.0386, prevalence 0.0319.

**Pre-registered claims (severe delay): C1, C2 and C3 are all supported.**
C1: the lower end of the lift interval is 1.73, above 1.0. C2: the whole interval of the difference against the carrier
baseline is above zero (+0.0121 to +0.0197). C3: precision at the top 10% inside each day is 6.7% (lower end 6.1%),
above the 2023 event rate of 3.19%.

**Read lift, not PR-AUC, across years.** PR-AUC rose from 0.0479 to 0.0591 mostly because the event rate rose from
2.67% to 3.19%. Lift is 1.80 and 1.85, and the intervals overlap. The model did not decay and did not improve.

Review capacity on 2023, ranking by the uncalibrated score:

| Capacity | Scheme | Reviewed | Events caught | Precision | Recall | Lift | False alerts per event caught |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5% | whole period | 2,273 | 275 | 12.1% | 1.9% | 3.80 | 7.3 |
| 1% | whole period | 4,545 | 476 | 10.5% | 3.3% | 3.29 | 8.5 |
| 5% | whole period | 22,721 | 1,842 | 8.1% | 12.7% | 2.54 | 11.3 |
| 10% | whole period | 45,442 | 3,318 | 7.3% | 22.9% | 2.29 | 12.7 |
| 1% | within each day | 4,668 | 412 | 8.8% | 2.8% | 2.77 | 10.3 |
| 10% | within each day | 45,547 | 3,037 | 6.7% | 21.0% | 2.09 | 14.0 |

**Quote the within-day figures for ranking skill.** The whole-period figures are flattered by season: June to August
hold about 82% of the whole-period top 1% alerts while being 39% of the flights. Inside each day the lift at the top
1% and 10% is 2.77 and 2.09, against 2.97 and 2.13 on 2022. The ranking held.

**What the model is.** Event rate rises from 0.81% in the lowest score decile to 7.30% in the highest, with no dip
between deciles. It is still a late-in-the-day detector. Departures from 14:00 hold 62% of the severe events and get
nearly all the alerts. Departures from 06:00 to 09:59 are 26% of the flights and 16% of the events, and a top 10%
review catches about 1% of those events. The largest carrier (DOT_19393, 21% of flights) gets no top 1% alert.
Recall at the top 10% shows no trend with the size of the delay (22.5% for 2 to 3 hours, 26.0% above 10 hours), so the
model is not finding the worst delays. EWR has 10.7% of its flights in the top 1%; LAS, LAX, MSP and PHX have none.

### Limits

* One partial year, one regime. No regime comparison inside the test.
* The model has seen nothing after 2021-06. The result is how a model frozen in mid-2021 did on 2023.
* The model has no state of the airport on the day. DEC-016 (outcome-aware features) was never built.
* The 2023 event rate of about 3.2% was seen in earlier descriptive work, disclosed in DEC-027 before the test.

---

## DEC-028: Cancellation is a negative result, and 2023 is spent

**Status:** Proposed.

**Decision.** Cancellation is reported as it came out: no demonstrated skill on 2023. No tuning, feature change or
model change for cancellation will be evaluated on 2023 data.

**Evidence.** 463,484 flights, 7,809 cancellations (1.685%, down from 2.68% in 2022).

| Scorer | 2023 PR-AUC | 95% interval | Lift |
| --- | --- | --- | --- |
| LightGBM (untuned baseline) | 0.0176 | 0.0151 to 0.0203 | 1.05 [0.97, 1.14] |
| Route history rate | 0.0208 | 0.0177 to 0.0244 | 1.24 |
| Origin history rate | 0.0196 | 0.0165 to 0.0233 | 1.17 |
| Prevalence | 0.0168 | 0.0145 to 0.0195 | 1.00 |

C1 not supported (lower end of lift 0.97). C2 not supported: the model minus the route baseline is -0.0032, with the
whole interval below zero (-0.0051 to -0.0014). C3 supported (precision at the top 10% inside each day 2.9%, lower end
2.3%, against 1.69%). The model already lost to route history on 2022 (0.0300 against 0.0343), so the lock predicted
this.

**What fails.** Whole-period lift is 0.97, 1.18, 0.81 and 0.88 at the top 0.5%, 1%, 5% and 10%. Inside each day it is
3.06, 2.76, 2.17 and 1.74. All top 1% alerts go to April (7.9% of April flights) and none to January, February, June,
July or August. The month pattern learned from 2019 to mid-2021 does not match where cancellations fell in 2023, so
the whole-period ranking has no skill. C3 holds because a fixed daily capacity removes the month pattern, which means
some within-day signal exists. It does not make an alert list that is better than reading route history.

**Rule from now on.** 2023 is spent. Any later modelling that is judged on 2023 is exploratory and must be labelled
as post-test. Nothing judged on 2023 may be reported as a held-out result or used to choose a model.

**Not done, deliberately.** No calibration, threshold, utility table or alert policy was built for cancellation.

---

## DEC-029: Probabilities are level-biased, so the review policy is capacity-based

**Status:** Proposed.

**Decision.** The alert policy uses a fixed review capacity (the top K% of each day), ranked by the uncalibrated
score. No probability threshold is part of the policy. Calibrated probabilities are reported with their error and are
not used to trigger anything.

**Evidence (severe delay).**

| Period | Variant | Mean predicted | Observed | Ratio | Brier | Slope | Intercept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | locked Platt (early-stop fit) | 2.48% | 2.67% | 0.928 | 0.02578 | 0.912 | 0.078 |
| 2023 | uncalibrated | 2.13% | 3.19% | 0.667 | 0.03068 | 1.041 | 0.420 |
| 2023 | locked Platt (early-stop fit) | 2.63% | 3.19% | 0.826 | 0.03056 | 0.966 | 0.199 |
| 2023 | Platt fitted on 2022 (diagnostic) | 2.82% | 3.19% | 0.885 | 0.03055 | 1.059 | 0.127 |

Platt beats uncalibrated on Brier on 2023: difference -0.000115, 95% interval -0.000151 to -0.000081. The gain is
tiny. The diagnostic calibrator is better than the locked one by 0.000016 (0.000008 to 0.000025), also tiny, and it
was not the locked choice. The event rate was 2.42% in the calibration rows, 2.67% in 2022 and 3.19% in 2023. The
slope stays near 1, so the shape of the probabilities holds. The level does not: they are about 17% too low in 2023.

**Operating points.** Cutoffs learned on 2022 flagged 0.69%, 1.33%, 6.61% and 12.82% of 2023 flights where 0.5%, 1%, 5%
and 10% were planned, and the F1-optimal cutoff flagged 11.7% instead of 9.1%. A fixed cutoff does not hold a fixed
workload. A capacity rule does. On 2023 the F1-optimal cutoff gives precision 7.1%, recall 26.1% and F1 0.111.

**Consequence.** Do not refit the calibrator on 2023. Do not describe the probabilities as reliable for the level of
risk. Say they are well ordered and about a sixth too low when the event rate is above the calibration period.

---

## DEC-030: Commit record for the final test

**Status:** Proposed. This is a disclosure, not a design decision.

Commit `7ee01c9`, titled "Lock the final-test protocol", holds 44 files, not one. The lock file was added on purpose.
The other 43 files were already staged from earlier phases (the Phase 1 and 2 audit, field mapping, timestamp code and
their tests), and the commit command took them as well. The commit message undersells what is in it.

At that time the Phase 3 to 7 code (`src/airline_disruption/final`, `models`, `calibration`, `explain`, `policy`,
`deep`, the Phase 7 scripts and their tests) was not in any commit. It was committed afterwards in [FILL: hash of the
second commit], whose message says it was committed after the final test. The lock's own SHA-256 seal and its file
fingerprints, not the commit contents, are the evidence that the protocol was fixed before 2023 was read.

History was not rewritten.

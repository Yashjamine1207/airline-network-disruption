# Phase 6 decision log drafts, Step 8

Two things to copy into `docs/decisions/decision_log.md`:

1. An **Outcome** block to append under DEC-020, with the numbers from your own Step 7 run.
2. The new entry **DEC-021**, after DEC-020.

Both stay "Proposed" until you have read them and changed the status yourself.

---

## DEC-020 outcome (append under DEC-020 after the Step 7 runs)

**Runs:** default configuration only, no random draws, seeds 42, 43, 44. Fit 1,454,352 rows,
early stopping 337,156 rows, validation 667,616 rows (2022). Nothing from 2023 was loaded.

Second half of 2022 (confirmation, 341,232 flights). PR-AUC, seed 42, with the seed spread
(largest minus smallest of the three seeds) and the three-seed mean:

| Model | Seed 42 | Seed spread | Three-seed mean | Best epoch |
| --- | --- | --- | --- | --- |
| LightGBM, `base_no_year`, tuned | 0.04680 | n/a | n/a | n/a |
| `mlp` (control, no sequence) | 0.04574 | 0.00124 | 0.04655 | 1 of 4 |
| `lstm` | 0.04539 | 0.00078 | 0.04581 | 1 of 4 |
| `gru` | 0.04538 | 0.00082 | 0.04575 | 2 of 5 |

Paired day-level bootstrap, second half of 2022, difference in PR-AUC with 95% interval:

| Comparison | Difference | 95% interval |
| --- | --- | --- |
| `lstm` minus LightGBM | -0.00141 | -0.00276 to +0.00009 |
| `gru` minus LightGBM | -0.00142 | -0.00300 to +0.00008 |
| `mlp` minus LightGBM | -0.00106 | -0.00236 to +0.00048 |
| `lstm` minus `mlp` | -0.00035 | -0.00118 to +0.00038 |
| `gru` minus `mlp` | -0.00035 | -0.00159 to +0.00085 |

On the full year, `gru` minus LightGBM is -0.00108 with an interval of -0.00190 to -0.00001, the only
interval that excludes zero. The full year and the first half were used to choose LightGBM's settings, so
they favour LightGBM slightly.

**Rule result:** neither comparison 1 (against the control) nor comparison 2 (against LightGBM) has an
interval wholly above zero for either recurrent model. By DEC-020 the added complexity is rejected. The
sequence carries no measurable value over the control, and the neural networks do not beat tuned LightGBM.

**Observations, not findings:**

* For all three networks, PR-AUC on the fit rows rose every epoch while PR-AUC on the early-stop rows fell
  after epoch 1 or 2. The networks learn patterns from the fit period that do not carry into 2021 H2.
* LightGBM, the MLP, the LSTM and the GRU land within 0.0015 PR-AUC of each other, about one seed spread
  (0.0008 to 0.0012), with ROC-AUC between 0.631 and 0.640. This looks like a ceiling set by what the
  schedule-only features contain, not by model capacity. That reading is an inference from the pattern.
* Both recurrent models score slightly below the control on average (three-seed means 0.04581 and 0.04575
  against 0.04655). The difference is inside seed noise. It gives no support to the idea that the sequence
  helps.

**Limits of this result:** default configuration only. The five random draws that DEC-020 allows were not
run at the time of this entry. If they are run, the same rule applies and the result is written here.

---

## DEC-021: The 2020 ablation, a size-matched control and the drop rule

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-015, DEC-018, DEC-019, DEC-020

### Context

The scope asks whether including the 2020 shock improves or harms later generalisation, and says 2020
is not removed merely because it is unusual. Two earlier results make a real 2020 effect plausible but
do not show one: dropping the calendar year raised 2022 PR-AUC (DEC-018), and every network peaked at
epoch 1 and then got worse on later rows while getting better on its own fit rows (DEC-020 outcome).

A plain comparison of "with 2020" and "without 2020" would mix two effects. Removing 2020 removes about
a third of the fit rows, and fewer rows can lower or raise a score for reasons unrelated to 2020.

### Decision

Three fit-row variants, all with the same early-stopping rows (2021-07-01 UTC onward), the same
validation rows (2022) and the same model settings:

| Variant | Fit rows |
| --- | --- |
| `full` | Every fit row (development before 2021-07-01 UTC) |
| `no2020` | The same rows without calendar year 2020, taken from the UTC prediction time |
| `matched` | A random subset of the fit rows with exactly as many rows as `no2020` has, drawn from all years |

`matched` removes the same number of rows as `no2020` but leaves 2020 in. The difference between `no2020`
and `matched` is the effect of removing 2020 with the amount of data held equal. The difference between
`matched` and `full` is the effect of having less data.

Models: tuned LightGBM on `base_no_year` (settings from DEC-019), then the `mlp` control and the `lstm`,
each with three seeds. The GRU is not repeated, because the LSTM stands for the recurrent models and the
GRU result matched it in Step 7. Comparisons use seed 42. Seed spread is reported next to every result.

Comparison: paired day-level bootstrap of PR-AUC on the **second half of 2022**. The full year is shown
for context. 2023 is not used.

### Decision rule, fixed before the ablation is run

Training without 2020 is adopted for a model only if `no2020` beats **both** `full` and `matched`, each
with the whole 95% interval above zero on the second half of 2022. In every other case 2020 stays in the
fit rows.

| Result | Reading |
| --- | --- |
| `no2020` beats `full` and `matched` | 2020 harms later generalisation for this feature set. Drop it |
| `no2020` beats `full` but not `matched` | The gain is about data quantity or noise, not about 2020. Keep 2020 |
| `no2020` does not beat `full` | 2020 neither helps nor clearly hurts. Keep 2020 (simplest) |
| `matched` beats `full` | Fewer random rows scored higher. That points to noise or overfitting, not to 2020. Keep 2020 and investigate |
| `full` beats `no2020` and `matched` | The 2020 rows are useful, or the amount of data matters. Keep 2020 |

I do not have a confident expectation for the direction. The `matched` control exists because the naive
comparison would be confounded.

### Limits

* "2020" here is the calendar year of the UTC prediction time. It is a proxy for the COVID regime, not a
  measurement of it, and the result is not a causal statement about COVID.
* Removing 2020 leaves 2019 and the first half of 2021, with a gap year between them.
* LightGBM category levels are recomputed from each variant's own fit rows. The neural-network inputs
  are not: vocabularies, medians and scalers were learned once from all fit rows. An embedding for a level
  seen only in 2020 therefore stays at its starting values instead of mapping to "unknown". Route levels
  seen only in 2020 are a small share of rows, so the effect should be small, but it is not zero.
* The result applies to schedule-only features and to the 2022 validation year. It does not say why 2020
  matters or does not.
* If 2020 is dropped, the decision applies to how the final model is trained, and the final-test protocol
  (DEC-015) still governs when 2023 is touched.

### Not decided here

* The Transformer's design.
* Whether the five random draws of DEC-020 are run.
* The outcome-aware ablation of DEC-016.

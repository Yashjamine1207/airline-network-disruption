# Phase 6 decision log drafts, Step 7

Copy the entry below into `docs/decisions/decision_log.md` after DEC-019. Status stays
"Proposed" until you have read it and changed it to "Accepted" yourself. Also change
DEC-017 to "Accepted" if you agree with `reports/tables/phase6_sequence_density.csv`
from your own data (the density rule was met).

---

## DEC-020: Neural-network design, control model, budget and decision rule

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-014, DEC-015, DEC-016, DEC-017, DEC-018, DEC-019
**Input version:** `phase6_deep_inputs_v1`   **Feature version:** `phase6_features_v1`

### Context

DEC-017 fixed the sequence (origin-airport schedule, 3-hour bins, 48-hour lookback).
DEC-019 fixed how LightGBM is tuned and confirmed. The neural networks need the same
discipline, or a network that was tried five ways would be compared with a LightGBM
that was tried once.

### Decision

**Models.** Three architectures, one code path (`airline_disruption/deep/keras_models.py`):

| Name | Static branch | Sequence branch | Role |
| --- | --- | --- | --- |
| `mlp` | yes | none | Control. Same inputs and head, no sequence |
| `lstm` | yes | LSTM over 16 bins, mask passed in | Candidate |
| `gru` | yes | GRU over 16 bins, mask passed in | Candidate |

The compact Transformer is trained after these two, under the same protocol, because the
scope requires the comparison. Its design is not fixed here.

**Static inputs.** Exactly the `base_no_year` columns (DEC-018), checked by a test.
Four categorical columns become embeddings (width `1.6 * n^0.56`, between 4 and 32).
A level seen in fewer than 20 fit rows, or never seen, shares one "unknown" embedding.
Month, day of week and the departure and arrival clock times enter as sine and cosine.
Distance and scheduled elapsed time are `log1p` then standardised. Missing numbers get
the fit-row median. There is no year.

**Learned from fit rows only:** vocabularies, medians, means and standard deviations,
and the sequence scaler (sample of 200,000 fit rows, real positions only).

**Layers.** Static: Dense, Dropout. Sequence: LSTM or GRU with the padding mask. Head:
concatenation, Dense, Dropout, one sigmoid unit whose bias starts at the fit-row
log-odds. Loss is unweighted binary cross-entropy. Optimiser is Adam. Batch size 2048.
At most 15 epochs. Early stopping watches PR-AUC on the early-stop rows (development
from 2021-07-01 UTC), patience 3, best weights restored. The 2022 rows are not seen
during training or stopping. No class weights and no resampling.

**Budget.** The default configuration is trained first (32 recurrent units, dense width
64, dropout 0.2, learning rate 0.001). Up to 5 random draws may follow, from 3 recurrent
sizes, 3 dense widths, 3 dropout values and 2 learning rates. The challenger is the draw
with the best PR-AUC on the first half of 2022. It is adopted only if the paired
day-level bootstrap on the second half puts the whole 95% interval above zero. Then the
chosen configuration is trained with two more seeds, to show how much a score moves with
the seed alone. The number of draws is written beside every result.

**Which numbers count.** The second half of 2022 is the clean comparison, because no
setting of any model was chosen on it. LightGBM's settings were chosen on the first
half (DEC-019), so the first half and the full year favour LightGBM. They favour a
network too if draws were tried. Tables show all three, and the second half is the one
quoted.

### Decision rule, fixed before any network has been trained

Three paired comparisons on the second half of 2022, all by day-level bootstrap:

1. **Sequence value:** `lstm` or `gru` against `mlp`.
2. **Against the benchmark:** `lstm` or `gru` against the tuned LightGBM on `base_no_year`.
3. **Network cost:** `mlp` against the tuned LightGBM.

A recurrent model is kept as a serious candidate only if comparison 1 AND comparison 2
both have a 95% interval wholly above zero, and the point estimate of comparison 2
is larger than the model's own seed spread (largest minus smallest second-half PR-AUC
across its three seeds). A gain smaller than what a different seed produces is not a
finding.

Outcomes and what they will be called:

* Only comparison 1 is above zero: the sequence carries some signal, but the network
  family does not beat LightGBM here. LightGBM stays the selected model.
* Only comparison 2 is above zero, comparison 1 is not: the network beats LightGBM, but
  the sequence is not the reason. That is a finding about the network family, not about
  sequences.
* Neither: the added complexity is rejected. This is the outcome expected in advance.

### What is expected

Schedule-only sequences carry little that the hour, day, airport and route features do
not already carry (DEC-016, DEC-017). The most likely result is no gain over LightGBM
and no gain of the sequence models over the control.

### Watch item: traffic level can act as a regime proxy

The median number of scheduled origin departures in the 48-hour lookback rose from 54
(development) to 62 (2022 validation). The sequence channels carry traffic level. In
2020 traffic collapsed together with disruption patterns that were unlike 2019 and 2022.
A sequence model could learn "low traffic means low risk" from 2020 and misapply it to
later years, the same way the year column did (DEC-018). This is a hypothesis. The
planned "train with 2020 versus without 2020" ablation is the test. Until it is run,
any sequence result is reported with this caveat.

### Repeatability

Two runs of the same model with the same seed gave bit-identical predictions on a
2-core CPU. On a machine with more threads, floating point additions can happen in a
different order and small differences are possible. Seeds, thread count, TensorFlow
version and hardware are written into every decision file. The seed check is part of
the protocol for this reason.

### Not decided here

* The Transformer's size and settings.
* Lookback and bin-width ablations (24 hours and 96 hours are allowed on 2022 rows,
  all variants reported, none presented as chosen in advance).
* Whether the outcome-aware ablation of DEC-016 is built.
* Calibration, explanations and error analysis (Phase 7A).

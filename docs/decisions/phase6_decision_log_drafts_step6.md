# Phase 6 decision log drafts, Step 6

Copy the entry below into `docs/decisions/decision_log.md` after DEC-018. It stays
"Proposed" until you have read it.

If you have not yet done so, also change DEC-017 to "Accepted". Its density rule
was fixed in advance and your data met it: the route sequence had a median of 1
prior flight in 48 hours (94% of flights had fewer than 5), and the carrier-at-airport
sequence had a median of 9 (36.5% fewer than 5). The airport sequence had a median of 54.

---

## DEC-019: LightGBM tuning protocol, and the same protocol for the deep models

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-014, DEC-015, DEC-018
**Code:** `src/airline_disruption/models/lightgbm_tuning.py`, `scripts/tune_phase6_lightgbm.py`

### Context

Step 4 ran LightGBM with Phase 4's settings and no tuning. Early stopping fired after
24 to 78 trees, and fitting on 1.45M rows instead of 300k raised PR-AUC by about 0.005,
so the model was not near its limit. Comparing tuned deep models against an untuned
LightGBM would prove nothing, and would favour the deep models.

### Decision

1. **Search space** (fixed in advance): learning rate, number of leaves, minimum child
   samples, L2 regularisation, column and row sampling, and four settings that control
   splits on high-cardinality categories (`cat_smooth`, `min_data_per_group`, `cat_l2`,
   `max_cat_threshold`). Nothing else is tuned. Every list contains the Phase 4 default.
2. **Budget:** the default settings plus 30 random draws, seed 42.
3. **Same rows as Step 4:** fit on development before 2021-07-01 UTC, early stopping on
   development from 2021-07-01 UTC.
4. **Select on the first half of 2022** (prediction time before 2022-07-01 UTC). The
   configuration with the highest PR-AUC there is the challenger.
5. **Confirm on the second half of 2022.** The challenger replaces the default only if a
   paired day-level bootstrap of the PR-AUC difference (200 resamples) has its 95%
   interval wholly above zero. Otherwise the default stays and the benchmark is called
   untuned in every report.
6. **The confirmation half is never used to choose.** The gain it shows is therefore
   not inflated by picking the best of 31. The full-year 2022 figures are partly
   selection-contaminated and are labelled as such.
7. **2023 is never loaded.**
8. **Deep models follow the same rule** (select on H1, confirm on H2 against the
   LightGBM benchmark). Their budget is smaller because each fit is far slower. The
   number of configurations tried is recorded beside every deep-model result, and the
   difference in budget is stated wherever the models are compared.
9. **Feature set:** `base_no_year` (DEC-018).

### What a result will and will not show

* An adopted challenger shows that these settings beat the Phase 4 settings on
  2022 H2 by more than day-to-day noise. It does not show they are optimal.
* "Default kept" shows that this search space did not contain a confirmed improvement
  at this budget. It does not show LightGBM cannot be improved.
* The ablation ladder (`base_no_year_history`, `_network`) will be re-run with the
  selected settings, because the ladder finding after Step 5 ("history and network features add
  nothing measurable") was obtained with untuned settings.

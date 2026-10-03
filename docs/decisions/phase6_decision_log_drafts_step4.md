# Phase 6 decision log drafts, Step 4

Copy the entry below into `docs/decisions/decision_log.md` after DEC-015. Status stays
"Proposed" until you have read it and changed it to "Accepted" yourself.

---

## DEC-016: Sequence models use schedule-only inputs; one outcome-aware ablation, matched

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-014 (UTC embargo), DEC-015 (final-test protocol), Phase 3 feature approval

### Context

Phase 3 approved pre-flight predictors that come from the schedule only: calendar
fields, distance, carrier, airport, route, prior-month scheduled-flight counts and
prior-month schedule-only network degrees. It did not approve historical outcome
rates (delay, cancellation or severe-delay rates) as pre-flight predictors, because
the source file does not say when an earlier flight's outcome became known.

A sequence model can only show value beyond LightGBM if it sees information that
LightGBM does not. Schedule-only sequences carry little of that: the same counts
are already in the tabular features. Outcome-aware sequences (recent airport delay
pressure) are the only version with a realistic chance of adding signal, and they
are also the version with a leakage risk.

### Decision

1. **Primary sequence models are schedule-only.** Inputs are ordered past
   schedule records for one entity, using only fields Phase 3 already approved.
   This keeps the main Phase 6 result inside the approved leakage boundary.
2. **One outcome-aware ablation is allowed.** It is labelled "outcome-aware
   ablation" in every table and figure. It is never the headline result and is
   never described as the selected pre-flight model unless DEC-017 (to be written
   after the results) says so explicitly.
3. **Every outcome-aware sequence model has a matched tabular baseline.** A LightGBM
   model receives the same aggregate information as the sequence model (same
   windows, same source rows, same availability rule), so the comparison isolates
   the value of sequence structure, not the value of extra information.
4. **The outcome-known rule is fixed before any outcome-aware feature is built.**
   Proposed rule, to be confirmed in Step 5 before code is written:
   * A prior flight's arrival delay counts only if its actual arrival time, converted
     to UTC, is at or before the prediction timestamp of the flight being predicted.
   * A prior flight's cancellation flag counts only if its scheduled departure time
     in UTC is at or before that prediction timestamp.
   * Flights with missing or invalid actual times do not count.
   * This rule truncates toward flights that landed early, which is what an analyst
     would actually have seen at that moment. The bias is part of the setup, not a
     bug, and is documented in the ablation table.
5. **The Phase 3 block on historical outcome rates stays in force** for the main
   pipeline, the LightGBM benchmark and the schedule-only sequence models.
6. **The historical-rate baselines in Step 4 are comparators only.** They are
   lookup tables learned from the fit period and applied to later rows. They are not
   features and do not enter any model.
7. **No aircraft identity is inferred**, by any route, carrier, flight-number or
   schedule-similarity rule. Sequences are per airport, per route or per
   carrier-airport context. The word "rotation" is not used.
8. **2023 plays no part** in choosing the outcome-known rule, the window lengths,
   the model family or any threshold.

### Consequences

* If schedule-only sequence models do not beat LightGBM on 2022 validation, that is
  a valid result and is reported as such.
* If the outcome-aware ablation beats its matched baseline, the report must say the
  gain comes from sequence structure only after the matched baseline has been
  given identical aggregates. If the matched baseline closes the gap, sequence
  structure did not help.
* If the outcome-aware ablation and its matched baseline both beat the schedule-only
  models, that shows outcome information helps. It does not show the information is
  available in a live setting, because the source has no record of when outcomes
  were published.

### Not decided here

* Which single entity (airport, route or flight-context) is the primary sequence.
  That is Step 5.
* Lookback length and window size. Step 5, chosen on development data only.
* Whether the outcome-aware ablation is run at all. It is run only if the
  schedule-only sequence models are stable first.

# Phase 6 decision log drafts, Step 5

Copy the entries below into `docs/decisions/decision_log.md` after DEC-016. Both stay
"Proposed" until you have read them and, for DEC-017, looked at
`reports/tables/phase6_sequence_density.csv` from your own data.

---

## DEC-017: The primary sequence is the origin-airport schedule sequence

**Status:** Proposed
**Phase:** 6
**Depends on:** DEC-014, DEC-015, DEC-016, Phase 3 feature approval
**Sequence version:** `phase6_sequences_v1`

### Context

The source has no aircraft identifier, so a sequence cannot follow an aircraft. The
scope allows three entities: an airport, a route, or a fixed number of preceding
schedule records for an airport, route or carrier. One primary definition has to be
chosen and defended.

The source is a sample of about three million flights over 56 months. Counts taken
from it are counts within the sample, not real traffic. The Phase 3 prior-month
features have the same limit.

### Decision

The primary sequence is the **origin airport's schedule**, defined as follows.

| Item | Value |
| --- | --- |
| Entity | Origin airport of the predicted flight |
| Time axis | Fixed UTC bins, 3 hours wide, aligned to UTC midnight |
| Lookback | 16 bins (48 hours) |
| Newest position | The last bin that closed at or before the prediction time |
| Prediction time | Scheduled departure (UTC) minus 2 hours, as in DEC-014 |
| Channels | scheduled departures, scheduled arrivals, distinct destinations |
| Source fields | origin, destination, scheduled departure UTC, scheduled elapsed time |
| Arrival time | Scheduled departure UTC plus scheduled elapsed time |
| Burn-in | Bins in the first 24 hours of the file are unavailable |
| Padding | Values 0, mask False, always at the oldest end |
| Static inputs | The `base_no_year` tabular features, so "base + airport sequence" adds only the sequence |
| Scaling | `log1p`, then standardise. Mean and standard deviation from FIT rows only |
| Not used | Outcomes, calendar year, regime labels, the open bin, later bins |

The bin width and lookback are fixed here, before any model is trained, and are not
tuned on 2023. A sensitivity ablation on the 2022 validation rows (24 h and 96 h
lookbacks) is allowed. All variants are reported. None is later presented as the
choice made in advance.

Cancelled and diverted flights count as scheduled flights. They were on the schedule.
No outcome column is read by the builder, and a test checks that.

### Why the airport, not the route or carrier

Airports have the densest history. The build script measures this on development rows
only and writes `phase6_sequence_density.csv`: the median number of scheduled flights
of the same entity in the 48 hours before the prediction time, for an origin airport,
a route, and a carrier at an origin airport.

Rule fixed in advance: a route sequence becomes a candidate primary sequence only if
its median is at least 10 and fewer than 25% of flights have fewer than 5 prior flights.
Otherwise the route and carrier-at-airport sequences stay ablations. If the table from
the real data contradicts the airport choice, this entry is revised before any sequence
model is trained.

The destination airport index is stored beside the origin index so a destination
sequence can be tested as an ablation without rebuilding anything.

### Why no year or regime input

In Step 4, dropping the calendar year raised 2022 validation PR-AUC (see DEC-018).
Giving a sequence model a year or regime flag repeats the same problem: it would learn
2019 to 2021 regimes and meet a regime it has never seen in 2022. Regimes are studied
in the evaluation by period, not fed in as inputs.

### What is expected

Stated now so it cannot be adjusted later: schedule-only sequences carry little that
the prior-month counts and the hour-of-day and day-of-week features do not already
carry. The most likely result is no gain over the tabular benchmark. If so, that is
reported as the result.

### Not decided here

* Model architectures and training settings (Step 6 onward).
* Whether the outcome-aware ablation of DEC-016 is built (only after the schedule-only
  models are stable).

---

## DEC-018: Drop `scheduled_departure_year` from the benchmark features

**Status:** Proposed
**Phase:** 6
**Evidence:** Step 4 run, protocol `phase6`, 2022 validation rows, 667,616 flights

| Model | PR-AUC | Lift | ROC-AUC | Best iteration |
| --- | --- | --- | --- | --- |
| `base` (with year) | 0.04146 | 1.55 | 0.6234 | 78 |
| `base_no_year` | 0.04556 | 1.71 | 0.6361 | 32 |

Paired day-level bootstrap of the difference (100 resamples): +0.0041, 95% interval
0.0026 to 0.0056. The no-year model was better in every resample.

### Decision

The Phase 6 ablation ladder is built on `base_no_year`
(`base_no_year_history`, `base_no_year_network`, `base_no_year_history_network`).
The with-year sets remain in the registry for comparison with Phase 4 only.

### What this does and does not show

* It shows the year column hurts on 2022 under this protocol.
* The mechanism is a hypothesis, not a finding: a tree can use the year to fit 2020 and
  2021 behaviour separately, and 2022 falls in the "2021 or later" branch. The
  "train with 2020 versus without 2020" ablation is the test of that idea.
* This choice was made on 2022 validation, which is the period reserved for model
  selection. 2023 played no part.
* The Phase 4 figure of 0.0453 is reproduced by `phase4_reproduce` (0.04530). The
  `phase6` figure for the same features is lower (0.04146). Two things differ between
  those runs (the second half of 2021 moves from fit to early stopping, and early
  stopping no longer sees 2022), so the gap cannot be split between them. Use the
  `phase6` numbers for comparisons.

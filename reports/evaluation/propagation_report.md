# Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience

## Airport and route association

This is an advanced retrospective portfolio study for Data Science, applied science, forecasting, operations research, decision science, transportation, and aviation analytics roles. This report documents airport-level associations, **not** aircraft-tail propagation or causal transmission.

## Same-year airport association

The centrality-versus-disruption analysis retains 449 origin airport-years meeting its minimum completed-flight count. Graph measures and disruption outcomes in the same UTC year describe association; their ordering does not establish that centrality preceded an event.

![Centrality versus disruption](../figures/centrality_vs_disruption.png)

## Consecutive-month association

The flight sample and temporal-split manifest were joined by `source_row_number`. Outcomes were grouped by the manifest's prediction-timestamp UTC month at each origin airport. Each retained preceding and current month has at least 50 scheduled and 50 completed flights. The preceding month is non-overlapping with the current month.

| Outcome | Airport-month pairs | Airports | Pooled Pearson r | Within-airport Pearson r |
|---|---:|---:|---:|---:|
| severe | 6,263 | 159 | 0.378 | 0.256 |
| cancellation | 6,263 | 159 | 0.340 | 0.321 |

The retained current-month range runs from 2019-02-01 through 2023-08-01. Within-airport correlations centre each airport's rates on its own mean. They do not control for seasonal or network-wide shocks, changes in flight mix, or dependence between consecutive pairs.

## What this does not establish

- These are retrospective outcome associations, not pre-flight predictive features. Actual outcome availability before each prediction timestamp has not been demonstrated.
- The analysis has not linked individual aircraft: the audited source lacks a usable tail identifier. It cannot estimate aircraft-leg cascades or tail-based propagation.
- A next-window **route** disruption association and downstream-flight exposure have not been estimated here. Annual route counts alone cannot establish either result.
- Partial 2023 coverage and changes across operating regimes limit comparisons. These results must not be used to tune models against the reserved 2023 final-test period.

The title `propagation_report.md` is the planned output name; its contents deliberately limit claims to the airport association that was actually measured. This is not a dispatch, optimisation, or real-time control system.


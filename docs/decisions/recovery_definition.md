# Recovery Definition

## Purpose

This document defines how the project will identify disruption episodes and measure recovery.

Recovery analysis is retrospective. It investigates how long eligible aircraft rotations, airports, or network conditions take to return to a defined normal state after disruption.

It is not a live operational recovery-management system and does not prescribe airline actions.

## Analytical levels

The project may analyse recovery at three levels:

1. **Aircraft rotation level**  
   Recovery of an eligible sequence of flights linked through a validated aircraft tail identifier.

2. **Airport level**  
   Recovery of airport disruption conditions measured in documented time windows.

3. **Network level**  
   Recovery of route-network disruption measures over documented time windows.

Aircraft-rotation recovery is the primary recovery analysis because it provides the clearest flight-to-flight sequencing when tail identifiers and timestamps are sufficiently reliable.

## Candidate normal-operation threshold

The primary candidate definition of normal operation is:

```text
Arrival delay below 15 minutes
```

Formally:

\[
\text{NormalOperation}_i =
\begin{cases}
1, & \text{if } \text{ArrivalDelayMinutes}_i < 15 \\
0, & \text{if } \text{ArrivalDelayMinutes}_i \geq 15
\end{cases}
\]

This is a project-defined analytical threshold. It is not an official airline, airport, or regulatory definition.

## Candidate disruption trigger

For aircraft-rotation recovery analysis, a disruption episode may begin when an eligible completed flight meets a documented disruption condition.

The primary candidate trigger is:

```text
Arrival delay greater than or equal to 120 minutes
```

This aligns the primary recovery trigger with the project’s severe-arrival-delay target.

The final trigger definition will be confirmed after the Phase 1 data audit verifies the available arrival-delay, cancellation, diversion, tail-number, and timestamp fields.

## Primary recovery event

For an eligible aircraft rotation, recovery occurs when a later eligible operated flight returns to normal operation:

```text
Recovery occurs when a later eligible flight has arrival delay below 15 minutes.
```

The recovery time is measured from the start of the disruption episode to the actual arrival time of the first later eligible flight that meets the normal-operation condition.

Formally:

\[
\text{RecoveryTime} =
\text{RecoveryEventTimeUTC} -
\text{EpisodeStartTimeUTC}
\]

All time differences will use UTC-normalised timestamps.

Local timestamps from different airports must not be subtracted directly.

## Candidate episode construction

An aircraft-rotation disruption episode will be constructed only when all required records pass rotation-eligibility checks.

A candidate episode contains:

1. An eligible disruption-trigger flight.
2. Zero or more later eligible flights in the same validated tail-number sequence.
3. The first later eligible normal-operation flight, if observed.
4. A recovery event or a right-censoring outcome.

A sequence must not be treated as a reliable aircraft rotation merely because flights share a carrier, route, airport, or similar schedule.

## Eligibility requirements

A flight sequence is eligible for aircraft-level recovery analysis only when:

- A valid tail identifier is available.
- Flights are ordered using validated UTC timestamps.
- The prior-to-current link passes documented turnaround-plausibility checks.
- Required actual arrival timestamps are available.
- Required final arrival-delay outcomes are available.
- Cancellation and diversion handling follows the documented source-data rules.
- The sequence does not contain unresolved timestamp ambiguity that prevents reliable ordering.

The final rules will be implemented after the data audit and documented in:

```text
docs/data/rotation_reconstruction_audit.md
```

## Cancellation and diversion handling

Cancelled and diverted flights may interrupt an otherwise eligible recovery sequence.

The final handling of these records will depend on the audited source fields and must be documented before survival analysis begins.

Possible outcomes include:

- Treating a cancelled or diverted flight as an episode continuation.
- Treating it as a separate disruption outcome.
- Excluding an ambiguous sequence from aircraft-level recovery analysis.
- Applying a documented sensitivity analysis.

The project must not silently classify missing actual-arrival outcomes as recovered flights.

## Right censoring

A disruption episode is right-censored when recovery is not observed before the available eligible sequence ends.

Examples include:

- The aircraft tail has no later eligible flight in the available dataset.
- The available data period ends before recovery is observed.
- Later flights have missing or invalid outcomes.
- The sequence becomes ambiguous or fails eligibility checks before recovery is observed.

For censored episodes:

```text
RecoveryObserved = 0
```

For observed recovery events:

```text
RecoveryObserved = 1
```

The censoring time must be recorded in UTC and calculated from the disruption-episode start.

## Sensitivity analysis

The project will test the sensitivity of recovery findings to alternative definitions where data support them.

Candidate sensitivity variations include:

| Component | Primary candidate | Sensitivity candidates |
|---|---|---|
| Disruption trigger | Arrival delay at least 120 minutes | Arrival delay at least 60, 90, or 180 minutes |
| Normal-operation threshold | Arrival delay below 15 minutes | Alternative low-delay thresholds to be documented |
| Sequence treatment | Eligible validated tail sequences | Stricter turnaround or timestamp-quality criteria |
| Cancellation handling | To be finalised after audit | Alternative documented treatment rules |
| Diversion handling | To be finalised after audit | Alternative documented treatment rules |

## Survival-analysis outputs

Where cohort size and assumptions permit, the project will report:

- Number of disruption episodes.
- Number and proportion of observed recoveries.
- Number and proportion of right-censored episodes.
- Kaplan-Meier recovery curves.
- Median recovery time where estimable.
- Recovery distributions by operating regime.
- Recovery distributions by carrier, airport, route, and disruption severity where meaningful.
- Cox proportional-hazards or accelerated-failure-time model results only after assumption checks.
- Sensitivity-analysis results.

## Interpretation limits

Recovery results describe patterns in the selected historical dataset and eligible analytical cohort.

They do not prove that a specific event, carrier, airport, route, aircraft, weather condition, or operational decision caused a faster or slower recovery.

All findings must state the episode definition, normal-operation threshold, eligibility conditions, censoring rules, and data-coverage limitations.
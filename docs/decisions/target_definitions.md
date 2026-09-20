# Target Definitions

## Purpose

This document defines the analytical targets for **Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience**.

All target definitions are project-specific analytical definitions. They are not official airline, airport, FAA, NOAA, or regulatory classifications.

The exact source column names, data types, coverage, missingness, and outcome availability will be confirmed during the Phase 1 data-source audit before target-construction code is written.

## Target 1: Severe arrival delay

### Primary classification target

The primary severe-delay classification target is:

```text
SevereDelay120 = 1 if arrival delay is greater than or equal to 120 minutes
SevereDelay120 = 0 if arrival delay is less than 120 minutes
```

Formally:

\[
\text{SevereDelay120}_i =
\begin{cases}
1, & \text{if } \text{ArrivalDelayMinutes}_i \geq 120 \\
0, & \text{if } \text{ArrivalDelayMinutes}_i < 120
\end{cases}
\]

### Eligible cohort

A flight is eligible for the severe arrival-delay target only when:

- The flight was completed or has a valid final arrival-delay outcome.
- The arrival-delay field is available and valid.
- The flight is not cancelled.
- The outcome belongs to the predicted flight rather than a previous aircraft leg.

Diverted-flight handling will be finalised after the source-data audit because the available fields and final-outcome rules must be confirmed first.

### Sensitivity analysis

The project will also evaluate the following severe-delay thresholds:

```text
SevereDelay60  = arrival delay greater than or equal to 60 minutes
SevereDelay90  = arrival delay greater than or equal to 90 minutes
SevereDelay120 = arrival delay greater than or equal to 120 minutes
SevereDelay180 = arrival delay greater than or equal to 180 minutes
```

The 120-minute threshold remains the primary headline target unless later documented evidence requires a protocol revision.

## Target 2: Cancellation

### Classification target

```text
Cancelled = 1 if the source dataset marks the flight as cancelled
Cancelled = 0 otherwise
```

The exact source field and valid values will be confirmed during the data-source audit.

### Eligible cohort

All flight records with a valid cancellation indicator are eligible for the cancellation target.

Cancellation prevalence will be calculated directly from the audited dataset. The project will not assume an event rate before the data audit.

## Target 3: Arrival-delay regression

### Regression target

For eligible completed flights:

```text
ArrivalDelayMinutes = final arrival delay in minutes
```

### Eligible cohort

A flight is eligible for arrival-delay regression only when:

- It is not cancelled.
- A final arrival-delay outcome is available.
- The arrival-delay value is valid.
- The record passes documented data-quality checks.

The completed-flight cohort size, exclusions, missingness, and treatment of diverted flights will be reported before regression models are trained.

## Target 4: Aircraft-rotation disruption propagation

This is a retrospective analytical target, not a pre-flight prediction target by default.

For aircraft-flight sequences that pass tail-identifier, timestamp-ordering, turnaround-plausibility, and prior-leg-availability checks, the project will analyse:

- Previous-leg arrival delay.
- Current-flight arrival delay.
- Current-flight severe-delay status.
- Current-flight cancellation status.
- Delay amplification between previous and current legs.
- Cascade depth.
- Cascade duration.
- Cumulative rotation delay.
- Downstream severe-delay and cancellation risk.

No aircraft-level propagation claim will be made for ambiguous, incomplete, incorrectly ordered, or unreliable tail-number sequences.

## Target 5: Recovery

Recovery will be analysed through disruption episodes built from eligible aircraft-rotation sequences and, where appropriate, airport or network time windows.

### Candidate normal-operation definition

A candidate project-defined normal state is:

```text
Arrival delay below 15 minutes
```

### Candidate recovery definition

A disruption episode is considered recovered when a later eligible operated flight returns to an arrival delay below 15 minutes.

Recovery definitions, episode-start rules, right-censoring treatment, and sensitivity analyses will be documented before survival modelling.

## Target 6: Airport and network disruption

Airport and route disruption measures will be derived from flight-level outcomes using documented past-only time windows.

Candidate measures include:

- Airport severe-delay rate.
- Airport cancellation rate.
- Airport delay pressure.
- Route severe-delay rate.
- Route cancellation rate.
- Airport and route exposure.
- Downstream disruption indicators.
- Network recovery measures.

These are descriptive analytical measures. They are not treated as causal measures or as a single network-accuracy target.

## Target 7: Weather-enhanced analysis

Weather is an explanatory feature family, not an outcome target.

The project will test whether eligible origin and destination weather features add measurable value to:

- Severe-delay prediction.
- Cancellation prediction.
- Arrival-delay regression.
- Aircraft-rotation propagation analysis.
- Recovery analysis.
- Airport and network disruption analysis.

Only weather observations available before the defined prediction timestamp may be used in a predictive model.
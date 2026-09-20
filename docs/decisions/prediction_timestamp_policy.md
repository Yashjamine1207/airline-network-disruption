# Prediction Timestamp Policy

## Purpose

This document defines the point in time at which the project makes a prediction.

The prediction timestamp is the boundary between information that is allowed in a model and information that is prohibited because it would create future-data leakage.

For any predicted flight \(i\), only information available on or before its prediction timestamp may be used.

## Primary prediction setting

The primary project setting is a **pre-flight disruption-risk prediction**.

The model estimates the risk that a scheduled flight will experience:

- A severe arrival delay.
- A cancellation.
- An arrival delay in minutes for the completed-flight regression cohort.

The project is retrospective. It does not produce live airline predictions or real-time operational instructions.

## Primary prediction timestamp

For each flight, define the primary prediction timestamp as:

```text
PredictionTimestampUTC =
ScheduledDepartureTimestampUTC - 2 hours
```

The scheduled departure timestamp must first be reconstructed using:

- The flight date in the source dataset.
- The original scheduled local departure time.
- The origin airport timezone.
- Documented handling of invalid times, date rollovers, and daylight-saving transitions.

The final prediction timestamp will be stored in UTC for safe ordering, joining, and availability checks.

The original local date, local scheduled departure time, origin timezone, and reconstructed UTC timestamp will also be retained for auditability.

## Reason for the two-hour horizon

A two-hour pre-flight horizon provides a clear and consistent analytical boundary.

It allows the project to evaluate whether schedule, historical, carrier, route, airport, eligible prior-leg, network, and historical weather information can identify disruption risk before the scheduled departure time.

This is a project design choice. It is not intended to represent an airline dispatch, passenger notification, crew-planning, or airport-control process.

## Permitted information

A feature may be used only if it was available on or before:

```text
PredictionTimestampUTC
```

Examples of potentially permitted information include:

- Scheduled flight date and scheduled local departure time.
- Origin and destination airport identifiers.
- Scheduled route and carrier identifiers.
- Calendar information derived from the schedule, such as month, day of week, and hour of day.
- Historical carrier, airport, route, and schedule-based disruption rates calculated only from earlier eligible flights.
- Rolling historical aggregates with windows ending on or before the prediction timestamp.
- Origin or destination weather observations recorded on or before the prediction timestamp.
- A previous aircraft leg only when its actual outcome was known on or before the current flight prediction timestamp.
- Network features calculated from past-only flight windows ending on or before the prediction timestamp.

## Prohibited information

The following information must never enter a pre-flight model for the predicted flight:

- Actual departure time.
- Actual arrival time.
- Actual departure delay.
- Actual arrival delay.
- Final flight duration.
- Cancellation reason.
- Delay-cause fields.
- Final disposition or outcome fields.
- Diversion outcome fields.
- Any post-departure operational variable.
- Any weather observation after the prediction timestamp.
- A complete-day weather aggregate that includes later observations.
- Future flights at the origin, destination, carrier, route, or aircraft tail.
- Future airport congestion or disruption measures.
- Future network features or centrality values.
- Historical statistics, target encodings, imputers, scalers, feature selection, calibration maps, or model parameters fitted using validation or final-test data.

## Previous-aircraft-leg availability rule

A previous aircraft leg may contribute an actual operational feature only when all of the following conditions are met:

1. The prior flight has the same validated tail identifier.
2. The prior flight is earlier in validated UTC time order.
3. The prior flight passes rotation-eligibility and turnaround-plausibility checks.
4. The prior flight's actual arrival outcome occurred on or before the current flight's prediction timestamp.

Formally:

\[
\text{PriorLegActualArrivalUTC} \leq \text{CurrentPredictionTimestampUTC}
\]

If this condition is not met, the prior-leg actual outcome is unavailable and must not be used.

## Weather availability rule

A weather observation may be matched to a predicted flight only when:

```text
WeatherObservationTimestampUTC <= PredictionTimestampUTC
```

The project will use either:

- The nearest valid weather observation at or before the prediction timestamp, or
- A documented aggregation window that ends on or before the prediction timestamp.

The project will preserve the matched station identifier, UTC observation time, observation quality metadata, and time gap between the observation and the prediction timestamp.

## Network availability rule

Airport and route network features must be calculated using only flight information available before or at the prediction timestamp.

A network graph constructed from future flights, the full test period, or later same-day flights must not be used to create a feature for an earlier prediction.

## Separate retrospective analyses

The following analyses are retrospective and are not governed by the two-hour pre-flight prediction timestamp in the same way:

- Aircraft-rotation reconstruction.
- Delay-propagation analysis.
- Recovery and survival analysis.
- Weather-event analysis.
- Post-event delay-cause attribution.
- Airport and network descriptive analysis.

These analyses must still use valid UTC ordering and clearly documented observation windows.

## Implementation requirements

The codebase must:

- Create `scheduled_departure_utc`.
- Create `prediction_timestamp_utc`.
- Retain the original local source time fields.
- Record the origin airport timezone used in each conversion.
- Flag missing, invalid, ambiguous, and daylight-saving-sensitive timestamps.
- Ensure every predictive feature has a documented availability rule.
- Include unit tests proving that future flight, weather, aircraft-rotation, and network information cannot enter a model row.

## Protocol-change rule

The two-hour pre-flight horizon is the default Phase 0 policy.

If the source-data audit shows that required schedule fields or timestamp quality are insufficient, the policy may be revised only through a documented decision-log entry before model development begins. Any revision must state:

- The reason for the change.
- The old and new timestamp definitions.
- The affected targets and features.
- The expected effect on comparability.
- The updated leakage tests required.
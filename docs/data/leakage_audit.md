# Leakage Prevention and Point-in-Time Feature Policy

## Purpose

This document defines how the project prevents future-data leakage.

A model has leakage when it uses information that would not have been known at the point the prediction is supposed to be made. Leakage can make offline results look unrealistically strong and invalidates the project’s predictive claims.

The primary project prediction timestamp is defined in:

```text
docs/decisions/prediction_timestamp_policy.md
```

For each predicted flight:

```text
PredictionTimestampUTC =
ScheduledDepartureTimestampUTC - 2 hours
```

Only information available on or before `PredictionTimestampUTC` may be used as a predictive feature.

## Core rule

For every feature value \(X_i\) used to predict flight \(i\):

\[
\text{FeatureAvailabilityTimeUTC}_i \leq \text{PredictionTimestampUTC}_i
\]

If this condition cannot be demonstrated, the feature must not be used in a pre-flight prediction model.

## Information allowed in pre-flight models

Subject to data-audit confirmation and point-in-time construction rules, pre-flight models may use:

- Flight schedule information.
- Scheduled origin and destination airports.
- Scheduled carrier and route identifiers.
- Calendar features derived from the schedule.
- Historical airport, route, carrier, and schedule risk measures calculated from earlier eligible records only.
- Past-only rolling aggregates.
- Weather observations available on or before the prediction timestamp.
- Eligible prior-aircraft-leg outcomes only when the prior outcome was available before the current prediction timestamp.
- Airport and route network features calculated from past-only windows ending on or before the prediction timestamp.

## Information prohibited in pre-flight models

The following fields or information types are prohibited as predictors for the predicted flight:

- Actual departure time.
- Actual arrival time.
- Actual departure delay.
- Actual arrival delay.
- Actual elapsed time.
- Final flight duration.
- Cancellation reason.
- Delay-cause fields.
- Carrier-delay, weather-delay, NAS-delay, security-delay, or late-aircraft-delay outcome fields.
- Final cancellation status when predicting cancellation.
- Final severe-delay status when predicting severe delay.
- Final diversion status or final disposition fields.
- Any final outcome field from the predicted flight.
- Future airport congestion.
- Future airport disruption measures.
- Future route disruption measures.
- Future carrier disruption measures.
- Future flight records.
- Future aircraft legs.
- Future network graph statistics.
- Weather observations recorded after the prediction timestamp.
- Whole-day aggregates that include observations from later in the day.
- Any manually entered information that depends on knowledge of the final outcome.

## Target-specific exclusions

### Severe-arrival-delay prediction

When predicting `SevereDelay120`, do not use:

- Arrival delay.
- Departure delay.
- Actual departure time.
- Actual arrival time.
- Delay-cause fields.
- Cancellation reason.
- Any post-departure information from the predicted flight.

### Cancellation prediction

When predicting `Cancelled`, do not use:

- Cancellation reason.
- Final cancellation indicator.
- Actual departure or arrival outcomes from the predicted flight.
- Delay-cause fields.
- Any field populated only after cancellation or final disposition.

### Arrival-delay regression

When predicting final arrival delay in minutes, do not use:

- Actual arrival delay.
- Actual arrival time.
- Actual departure delay.
- Actual departure time.
- Actual elapsed time.
- Delay-cause fields.
- Any final outcome field from the predicted flight.

## Historical-feature policy

Historical carrier, airport, route, schedule, and calendar risk features must be calculated from records that occurred before the current flight prediction timestamp.

For each evaluation fold:

- Training-period rows may be used to calculate historical features for later rows.
- Validation-period rows must not influence training-period features.
- Final-test rows must not influence training or validation features.
- A flight must not contribute to its own historical feature value.
- Same-day future flights must not contribute to earlier same-day flight features.

Examples of allowed historical features:

- Severe-delay rate for a route using eligible earlier flights only.
- Cancellation rate for an origin airport using eligible earlier flights only.
- Rolling 30-day carrier disruption rate ending before the prediction timestamp.
- Historical median arrival delay for an airport-hour combination using earlier completed flights only.

## Encoding, preprocessing, and tuning policy

The following operations must be fitted inside the training portion of each chronological fold only:

- Category encoders.
- Target encoders.
- Imputers.
- Scalers.
- Feature selectors.
- Dimensionality-reduction methods.
- Resampling methods.
- Calibration methods.
- Hyperparameter selection.
- Threshold selection.

The validation period may be used for model selection, calibration selection, and threshold selection.

The final temporal test period must remain untouched until feature families, model family, hyperparameters, calibration method, and alert-ranking policy are locked.

## Aircraft-rotation policy

Actual information from a prior aircraft leg is permitted only when:

1. The previous flight has the same validated tail identifier.
2. The previous flight is earlier in validated UTC order.
3. The rotation link passes documented eligibility checks.
4. The previous flight’s actual arrival time is on or before the current flight’s prediction timestamp.

Formally:

\[
\text{PriorLegActualArrivalUTC} \leq \text{CurrentPredictionTimestampUTC}
\]

A shared carrier, airport, route, or schedule pattern alone is not evidence of a valid aircraft rotation.

## Weather-feature policy

A weather observation is eligible only when:

\[
\text{WeatherObservationTimestampUTC} \leq \text{PredictionTimestampUTC}
\]

Weather matching will use either:

- The nearest valid observation at or before the prediction timestamp, or
- A documented aggregation window ending on or before the prediction timestamp.

The project will retain the weather-station identifier, observation UTC time, quality information, airport-station mapping method, and weather-to-prediction time gap.

## Network-feature policy

Airport-route network features must be calculated from a past-only flight window.

For a flight predicted at time \(T\), the corresponding network graph can contain only eligible flights whose relevant information was available on or before \(T\).

The project must not:

- Build a graph using the full final-test period.
- Use a final-period centrality score to predict earlier flights.
- Use future same-day flights in airport or route disruption features.
- Use future outcomes to construct network exposure measures.

## Retrospective-only fields

Delay-cause fields may be used only for retrospective post-event attribution analysis.

When such analyses are performed, reports must state clearly that:

- The fields were not used in pre-flight predictive models.
- The analysis occurs after the flight outcome is known.
- Associations are not proof of causation.

## Required leakage tests

The project will implement automated tests that verify:

- No feature timestamp is later than the prediction timestamp.
- No future weather observation is matched to a prediction row.
- No future flight enters an historical aggregate.
- No flight contributes to its own historical target rate.
- No future network information enters a network feature.
- No unavailable prior-aircraft-leg actual outcome enters a rotation feature.
- Preprocessing objects are fitted only on training-fold data.
- Calibration and threshold selection exclude the final temporal test period.
- The final test set is not used during feature selection or hyperparameter tuning.

## Audit output

Before headline modelling results are reported, the project will produce a leakage audit containing:

- Feature name.
- Source field or source table.
- Feature formula.
- Feature availability rule.
- Lookback window.
- Prediction timestamp relationship.
- Missing-data handling.
- Evaluation-fold fitting rule.
- Leakage test reference.
- Approval or exclusion decision.
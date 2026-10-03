# Phase 3 Feature Policy

## Purpose

This document defines which information may be used as a predictor for a flight at prediction time.

## Prediction timestamp

For pre-flight prediction tasks, the prediction timestamp is the scheduled departure time of the predicted flight.

The current dataset stores scheduled departure time in local airport clock time through:

- `FL_DATE`
- `CRS_DEP_TIME`
- `ORIGIN`

Timestamp normalisation will be implemented and audited before time-dependent features are used.

Until that implementation is complete, no feature may assume that a local timestamp is UTC.

## Allowed feature information

A feature may be used only when its source information would have been available by the prediction timestamp.

Permitted feature families include:

- Scheduled flight date.
- Scheduled departure time.
- Origin airport.
- Destination airport.
- Route.
- Carrier identifier.
- Flight number, where treated as a schedule identifier rather than an outcome.
- Calendar features derived from the scheduled departure date.
- Historical carrier, airport, and route statistics calculated only from earlier records.
- Past-only rolling statistics calculated only from earlier records.
- Flight-count features calculated from information available before the prediction timestamp.

## Prohibited predictor fields

The following fields are targets, outcomes, or post-outcome information and must not be predictors in pre-flight models:

- `ARR_DELAY`
- `CANCELLED`
- `DIVERTED`
- `CANCELLATION_CODE`
- `DEP_TIME`
- `DEP_DELAY`
- `TAXI_OUT`
- `WHEELS_OFF`
- `WHEELS_ON`
- `TAXI_IN`
- `ARR_TIME`
- `ELAPSED_TIME`
- `AIR_TIME`
- `DELAY_DUE_CARRIER`
- `DELAY_DUE_WEATHER`
- `DELAY_DUE_NAS`
- `DELAY_DUE_SECURITY`
- `DELAY_DUE_LATE_AIRCRAFT`

The following target-derived fields are also prohibited as predictors:

- `cancelled_target`
- `arrival_delay_minutes`
- `severe_delay_60`
- `severe_delay_90`
- `severe_delay_120`
- `severe_delay_180`
- `completed_flight`
- `target_exclusion_reason`

## Historical feature rule

Historical statistics must be calculated using earlier observations only.

For a prediction row at time `T`:

```text
historical feature data timestamp < T
```

The current row must not contribute to its own historical statistic.

Validation and final-test rows must not be used to fit historical encodings, aggregations, imputers, scalers, or other transformations used by an earlier period.

## Future-information rule

The following information is prohibited:

- Future flights.
- Future cancellation outcomes.
- Future arrival delays.
- Future departure delays.
- Future airport disruption rates.
- Future route disruption rates.
- Future carrier disruption rates.
- Future network aggregates.
- Complete-day aggregates containing observations after the prediction timestamp.
- Any feature calculated from the target row's outcome.

## Project-specific exclusions

Aircraft rotation features are excluded because the source has no usable tail-number field.

Weather features are excluded because NOAA weather data is not part of the project scope.

Network features are deferred until a separate past-only construction and test policy is implemented.

## Required tests

The feature pipeline must include tests proving that:

1. Current-row outcomes are not used as predictors.
2. Historical features use only earlier timestamps.
3. Validation and final-test observations do not fit transformations used for earlier periods.
4. Target columns are excluded from the predictor matrix.
5. Future flight rows cannot enter a prediction row's historical aggregation.
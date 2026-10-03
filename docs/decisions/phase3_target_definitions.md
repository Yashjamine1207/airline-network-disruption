# Phase 3 Target Definitions

## Scope

Targets were constructed from:

```text
data/raw/flights_kaggle/flights_sample_3m.csv
```

The raw source contains 3,000,000 scheduled flight records covering:

```text
2019-01-01 to 2023-08-31
```

The raw source file was not modified.

## Prediction timestamp

The prediction timestamp for pre-flight classification and regression tasks is the scheduled departure time of the flight.

Pre-flight model features must use only information available by that timestamp.

The target outcome itself is not available at prediction time and must never be used as an input feature.

## Cancellation target

The cancellation target is defined from the source field `CANCELLED`:

```text
cancelled_target = 1 when CANCELLED == 1
cancelled_target = 0 when CANCELLED == 0
```

Observed results:

| Measure | Value |
|---|---:|
| Total rows | 3,000,000 |
| Observed cancellation labels | 3,000,000 |
| Cancelled flights | 79,140 |
| Cancellation prevalence | 2.6380% |

## Completed-flight cohort

A flight is eligible for completed-flight targets when all conditions hold:

```text
CANCELLED == 0
DIVERTED == 0
ARR_DELAY is present and finite
```

Observed cohort:

| Measure | Value |
|---|---:|
| Completed flights | 2,913,802 |
| Missing or unavailable regression targets | 86,198 |

The completed-flight cohort is used for arrival-delay regression and severe-arrival-delay classification.

## Severe-arrival-delay targets

Severe arrival delay is a project-defined analytical target based on `ARR_DELAY`.

The primary target is:

```text
severe_delay_120 = 1 when ARR_DELAY >= 120 minutes
```

Sensitivity thresholds are also constructed:

```text
severe_delay_60  = 1 when ARR_DELAY >= 60 minutes
severe_delay_90  = 1 when ARR_DELAY >= 90 minutes
severe_delay_180 = 1 when ARR_DELAY >= 180 minutes
```

Only eligible completed flights receive a severe-delay label. Cancelled, diverted, and invalid-arrival-delay records have missing severe-delay labels.

Observed results:

| Target | Observed labels | Positive events | Prevalence |
|---|---:|---:|---:|
| `severe_delay_60` | 2,913,802 | 178,412 | 6.1230% |
| `severe_delay_90` | 2,913,802 | 108,115 | 3.7104% |
| `severe_delay_120` | 2,913,802 | 69,801 | 2.3955% |
| `severe_delay_180` | 2,913,802 | 32,621 | 1.1195% |

The 120-minute threshold is a project-defined classification threshold. It is not an official classification supplied by the dataset publisher.

## Arrival-delay regression target

The regression target is:

```text
arrival_delay_minutes = ARR_DELAY
```

It is available only for the completed-flight cohort.

The target is measured in minutes and may be negative when a completed flight arrives earlier than scheduled.

## Excluded source fields

The following source fields are outcomes or post-outcome information and must not be used as predictors in pre-flight models:

- `ARR_DELAY`
- `CANCELLED`
- `DIVERTED`
- `CANCELLATION_CODE`
- `DELAY_DUE_CARRIER`
- `DELAY_DUE_WEATHER`
- `DELAY_DUE_NAS`
- `DELAY_DUE_SECURITY`
- `DELAY_DUE_LATE_AIRCRAFT`
- Actual departure, wheels-off, wheels-on, arrival, and elapsed-time outcomes.

The delay-cause fields may be used only in retrospective descriptive attribution analysis, if such analysis is later added.

## Excluded project analyses

The audited source does not contain a usable tail-number or aircraft identifier. Therefore, this project will not claim:

- Aircraft rotation reconstruction.
- Tail-based propagation.
- Aircraft cascade depth.
- Tail-based recovery.

NOAA weather data is also excluded from the project. No weather features or weather-event analysis will be constructed.

## Target output

The validated target table is:

```text
data/processed/targets/flight_targets.csv
```

It contains 3,000,000 rows and 24 columns.

Target construction version:

```text
phase3_step2_v1
```

Validation script:

```text
scripts/validate_targets.py
```
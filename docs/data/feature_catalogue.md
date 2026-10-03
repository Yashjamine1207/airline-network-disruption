# Feature Catalogue

## Status

This is the initial Phase 3 feature catalogue.

Features will be implemented only after their source fields, availability time, lookback rule, missing-data rule, and leakage behaviour are defined.

## Prediction timestamp

For a predicted flight, the intended prediction timestamp is its scheduled departure time:

```text
FL_DATE + CRS_DEP_TIME at the ORIGIN airport
```

The timestamp must be converted from origin-local time to an auditable UTC representation before temporal ordering and historical features are implemented.

## Feature groups

### 1. Schedule features

| Feature | Source field(s) | Formula | Availability | Missing-data rule |
|---|---|---|---|---|
| Scheduled departure hour | `CRS_DEP_TIME` | Hour extracted from scheduled departure time | Before scheduled departure | Invalid values become missing and are flagged |
| Scheduled departure minute | `CRS_DEP_TIME` | Minute extracted from scheduled departure time | Before scheduled departure | Invalid values become missing and are flagged |
| Scheduled arrival hour | `CRS_ARR_TIME` | Hour extracted from scheduled arrival time | Before scheduled departure | Invalid values become missing and are flagged |
| Scheduled arrival minute | `CRS_ARR_TIME` | Minute extracted from scheduled arrival time | Before scheduled departure | Invalid values become missing and are flagged |
| Scheduled elapsed time | `CRS_ELAPSED_TIME` | Source scheduled duration | Before scheduled departure | Preserve missingness and add a missingness flag |
| Distance | `DISTANCE` | Source scheduled distance | Before scheduled departure | Preserve missingness and add a missingness flag |

### 2. Calendar features

| Feature | Source field(s) | Formula | Availability | Missing-data rule |
|---|---|---|---|---|
| Year | `FL_DATE` | Calendar year | Before scheduled departure | Invalid dates are excluded from modelling |
| Month | `FL_DATE` | Calendar month | Before scheduled departure | Invalid dates are excluded from modelling |
| Day of week | `FL_DATE` | Calendar weekday | Before scheduled departure | Invalid dates are excluded from modelling |
| Day of month | `FL_DATE` | Calendar day | Before scheduled departure | Invalid dates are excluded from modelling |
| Week of year | `FL_DATE` | Calendar ISO week | Before scheduled departure | Invalid dates are excluded from modelling |
| Is weekend | `FL_DATE` | Saturday or Sunday indicator | Before scheduled departure | Invalid dates are excluded from modelling |

### 3. Entity features

| Feature | Source field(s) | Formula | Availability | Missing-data rule |
|---|---|---|---|---|
| Carrier identifier | `AIRLINE_CODE`, `DOT_CODE` | Audited carrier identifier | Before scheduled departure | Apply documented identifier policy |
| Origin airport | `ORIGIN` | Source airport code | Before scheduled departure | Missing values are flagged |
| Destination airport | `DEST` | Source airport code | Before scheduled departure | Missing values are flagged |
| Route | `ORIGIN`, `DEST` | `ORIGIN + '-' + DEST` | Before scheduled departure | Missing route components are flagged |
| Flight number | `FL_NUMBER` | Source schedule identifier | Before scheduled departure | Missing values are flagged |

### 4. Historical features

Historical features must use only records earlier than the prediction timestamp.

| Feature | Source field(s) | Formula | Lookback | Missing-data rule |
|---|---|---|---|---|
| Prior carrier flight count | Carrier identifier | Count of earlier flights | Project-defined trailing window | Zero or missing flag when no history exists |
| Prior origin flight count | `ORIGIN` | Count of earlier flights | Project-defined trailing window | Zero or missing flag when no history exists |
| Prior destination flight count | `DEST` | Count of earlier flights | Project-defined trailing window | Zero or missing flag when no history exists |
| Prior route flight count | Route | Count of earlier flights | Project-defined trailing window | Zero or missing flag when no history exists |
| Prior carrier cancellation rate | Carrier and `CANCELLED` | Earlier cancellations / earlier flights | Project-defined trailing window | Missing when no eligible history exists |
| Prior origin cancellation rate | Origin and `CANCELLED` | Earlier cancellations / earlier flights | Project-defined trailing window | Missing when no eligible history exists |
| Prior destination cancellation rate | Destination and `CANCELLED` | Earlier cancellations / earlier flights | Project-defined trailing window | Missing when no eligible history exists |
| Prior route cancellation rate | Route and `CANCELLED` | Earlier cancellations / earlier flights | Project-defined trailing window | Missing when no eligible history exists |

Historical rates must be calculated within the training portion of each temporal split when they are used for model training and evaluation.

### 5. Rolling features

Rolling features must be based on past rows only.

Potential features include:

- Prior flights from the same origin in the previous hour.
- Prior flights from the same origin in the previous three hours.
- Prior cancellations from the same origin in the previous day.
- Prior severe delays from the same route in the previous seven days.
- Prior carrier cancellation rate in the previous seven days.
- Prior destination disruption rate in the previous seven days.

The exact windows must be frozen in configuration before model evaluation.

### 6. Target-derived exclusions

The following fields must never enter a pre-flight predictor matrix:

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
- All `DELAY_DUE_*` fields
- All constructed target columns
- Any future aggregate containing these outcomes

### 7. Excluded feature families

#### Aircraft rotations

Excluded because the audited source contains no usable tail-number or aircraft identifier.

#### Weather

Excluded because NOAA weather data was removed from the project scope.

#### Network features

Not yet implemented. Any future network feature must be calculated from past-only flight records and documented separately before use.

## Implementation status

| Feature family | Status |
|---|---|
| Target construction | Complete |
| Schedule features | Not implemented |
| Calendar features | Not implemented |
| Entity features | Not implemented |
| Historical features | Not implemented |
| Rolling features | Not implemented |
| Rotation features | Permanently excluded |
| Weather features | Permanently excluded |
| Network features | Deferred |
| Leakage tests | Not implemented |

## Code versioning

Every generated feature table must include:

- Feature code version.
- Source file identifier.
- Source coverage.
- Prediction timestamp definition.
- Feature configuration version.
- Missing-data policy.
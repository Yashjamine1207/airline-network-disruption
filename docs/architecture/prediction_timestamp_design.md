# Prediction Timestamp Design

## Primary pre-flight prediction horizon

For each scheduled flight:

```text
PredictionTimestampUTC = ScheduledDepartureTimestampUTC - 2 hours
```

The model estimates disruption risk using only information available on or before this UTC timestamp.

```mermaid
flowchart LR

    A[Source flight date<br/>and scheduled local departure time]
    B[Origin airport<br/>IANA timezone mapping]
    A --> C[Reconstruct scheduled departure<br/>in origin local time]
    B --> C

    C --> D[Convert scheduled departure<br/>to UTC]

    D --> E[Subtract two hours]

    E --> F[PredictionTimestampUTC]

    F --> G{Was information available<br/>at or before this time?}

    G -->|Yes| H[Eligible predictive feature]
    G -->|No| I[Prohibited: future information]
```

## Information timeline

```mermaid
timeline
    title Flight prediction information boundary

    section Before PredictionTimestampUTC
      Historical route, carrier, and airport outcomes : Allowed when built from earlier eligible flights only
      Past weather observations : Allowed when observed at or before the prediction timestamp
      Previous aircraft-leg outcome : Allowed only if prior actual arrival occurred before the prediction timestamp
      Past-only airport-route network features : Allowed when graph window ends at or before the prediction timestamp

    section PredictionTimestampUTC
      Pre-flight disruption risk is estimated : Severe-delay risk, cancellation risk, or arrival-delay estimate

    section After PredictionTimestampUTC
      Actual departure of predicted flight : Prohibited
      Actual arrival of predicted flight : Prohibited
      Final arrival delay : Prohibited
      Cancellation reason and delay causes : Prohibited
      Future weather observations : Prohibited
      Later flights and later aircraft legs : Prohibited
      Future airport and network conditions : Prohibited
```

## Previous-aircraft-leg decision rule

```mermaid
flowchart TD

    A[Candidate prior flight<br/>with same tail identifier] --> B{Prior leg passes<br/>rotation eligibility checks?}

    B -->|No| C[Do not use as prior-leg feature]
    B -->|Yes| D{Prior actual arrival UTC<br/>is on or before current<br/>PredictionTimestampUTC?}

    D -->|No| E[Prior outcome was not yet known<br/>Do not use actual prior-leg outcome]
    D -->|Yes| F[Prior-leg actual outcome<br/>is eligible as a feature]
```

## Weather-observation decision rule

```mermaid
flowchart TD

    A[Candidate NOAA weather observation] --> B{Valid station match<br/>and acceptable quality?}

    B -->|No| C[Do not use observation]
    B -->|Yes| D{Observation timestamp UTC<br/>is on or before<br/>PredictionTimestampUTC?}

    D -->|No| E[Future weather<br/>Prohibited]
    D -->|Yes| F[Eligible weather observation]

    F --> G[Record station ID, observation time,<br/>quality metadata, and time gap]
```

## Non-negotiable rules

- Scheduled departure is reconstructed from the source flight date, scheduled local departure time, and origin-airport timezone.
- UTC is used for feature availability checks, weather matching, aircraft sequencing, network windows, and time comparisons.
- Original local source time fields are retained for auditability.
- A model must never use actual outcomes from the flight it predicts.
- A model must never use future weather, future flights, future prior-leg outcomes, future airport measures, or future network information.
- Historical aggregates, encoders, scalers, imputers, calibration methods, feature selection, and tuning must be fitted inside chronological training folds only.
- The final temporal test period must not influence feature, model, calibration, threshold, or policy selection.
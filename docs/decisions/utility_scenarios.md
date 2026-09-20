# Operational-Value Scenario Policy

## Purpose

This document defines how the project will assess whether disruption-risk models can prioritise flights for hypothetical analytical review.

This is a retrospective ranking and evaluation exercise.

It is not an airline operating policy, an autonomous decision system, a dispatch process, a crew-management process, a passenger rebooking system, or a real-time alerting platform.

## Core principle

Classification models produce a ranked list of flights by predicted disruption risk.

Instead of presenting a single probability threshold as universally correct, the project will evaluate how many disruption events are captured when a hypothetical analyst reviews only the highest-risk flights.

The model is therefore assessed as a prioritisation tool for retrospective analysis.

## Illustrative review capacities

The project will evaluate ranked flights at these review capacities:

| Capacity label | Flights reviewed |
|---|---:|
| Very limited review | Top 0.5% of ranked flights |
| Limited review | Top 1% of ranked flights |
| Moderate review | Top 5% of ranked flights |
| Broad review | Top 10% of ranked flights |

For each evaluation period, \(K\) is calculated as:

\[
K =
\lceil
\text{NumberOfEligibleFlights}
\times
\text{ReviewCapacity}
\rceil
\]

For example, if an evaluation cohort contains 100,000 eligible flights:

| Review capacity | Number of highest-risk flights reviewed |
|---|---:|
| Top 0.5% | 500 |
| Top 1% | 1,000 |
| Top 5% | 5,000 |
| Top 10% | 10,000 |

These capacity levels are illustrative analytical scenarios only.

They are not based on known airline staffing, operational capacity, cost, safety requirements, or business policy.

## Metrics reported at each capacity

For severe-delay and cancellation models, the project will report:

### Precision at K

```text
Precision@K =
Actual disruption events among the top K ranked flights
/
K
```

This answers:

```text
Among the flights selected for hypothetical review, what proportion were actual disruption events?
```

### Recall at K

```text
Recall@K =
Actual disruption events among the top K ranked flights
/
All actual disruption events in the eligible evaluation cohort
```

This answers:

```text
What proportion of all actual disruption events were captured within the top K ranked flights?
```

### Events captured at K

```text
EventsCaptured@K =
Number of actual disruption events among the top K ranked flights
```

### False alerts at K

```text
FalseAlerts@K =
Number of non-disruption flights among the top K ranked flights
```

### Alert rate

```text
AlertRate =
K
/
Number of eligible evaluation flights
```

The alert rate equals the selected review capacity.

## Separate scenarios by target

Operational-value scenarios will be reported separately for:

- Severe arrival delay at the primary 120-minute threshold.
- Severe-delay sensitivity thresholds of 60, 90, and 180 minutes.
- Cancellation prediction.
- Any separately documented propagation-risk prediction task.

A ranking for one target must not be interpreted as a ranking for another target unless explicitly evaluated.

## Probability thresholds

If the project reports a probability threshold in addition to ranked capacities:

- The threshold must be selected using validation data only.
- The same frozen threshold may be evaluated once on the final temporal test set.
- The threshold selection criterion must be documented.
- Resulting precision, recall, false positives, and false negatives must be reported.
- The threshold must not be described as an airline operational policy.

## Illustrative utility analysis

The project may compare models using a simple illustrative utility score only after primary metrics have been reported.

A generic scenario form is:

\[
\text{IllustrativeUtility} =
(\text{EventsCaptured} \times B)
-
(\text{FalseAlerts} \times C)
\]

Where:

- \(B\) is an assumed analytical value assigned to capturing one event.
- \(C\) is an assumed analytical cost assigned to one false alert.

If used, all values for \(B\) and \(C\) must be clearly labelled as hypothetical assumptions.

The project must not describe these assumptions as:

- Airline financial estimates.
- Verified operational costs.
- Passenger compensation values.
- Airport costs.
- Staffing costs.
- Airline policy.
- Recommended business decisions.

## Reporting requirements

Every prioritisation result must report:

- Target definition.
- Eligible cohort size.
- Event prevalence.
- Evaluation time period.
- Operating regime.
- Review capacity.
- Number of reviewed flights.
- Number of events captured.
- Precision@K.
- Recall@K.
- False alerts.
- Calibration status of the model.
- Whether the result comes from validation or the final temporal test set.
- Important coverage and data-quality limitations.

Where data support it, results may also be shown by:

- Carrier.
- Origin airport.
- Destination airport.
- Route.
- Weather-coverage availability.
- Tail-number and rotation eligibility.
- Operating regime.

## Interpretation limits

This analysis measures retrospective ranking quality within the selected historical data.

It does not demonstrate:

- Live system performance.
- Airline operational feasibility.
- Financial benefit.
- Safety benefit.
- Staffing requirements.
- Passenger impact.
- Causal reduction in delays or cancellations.
- Recommended intervention or dispatch actions.

All results remain retrospective analytical evidence.
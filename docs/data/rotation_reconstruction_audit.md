# Aircraft Rotation Reconstruction Audit

## Purpose

This document records whether the selected flight dataset supports reliable aircraft-level rotation reconstruction, disruption propagation analysis, cascade analysis, and tail-based recovery analysis.

The assessment is based on the audited local raw source file:

```text
data/raw/flights_kaggle/flights_sample_3m.csv
```

The source file was audited over the available flight-date coverage:

```text
2019-01-01 to 2023-08-31
```

## Required evidence for aircraft rotations

A reliable aircraft rotation requires, at minimum:

1. A stable aircraft or tail identifier for each eligible flight.
2. UTC-normalised scheduled and actual timestamps.
3. Valid time ordering within the same tail identifier.
4. Destination-to-next-origin continuity.
5. Plausible turnaround times.
6. Documented handling of cancellations, diversions, missing times, and ambiguous links.
7. Prior-leg actual outcomes available before the current flight prediction timestamp when a prior-leg outcome is used as a predictive feature.

Flights sharing a carrier, airport, route, flight number, or similar schedule are not sufficient evidence that they belong to the same aircraft rotation.

## Source-field audit result

The raw flight file was checked for recognised tail identifier fields:

| Candidate field | Present in source file |
|---|---:|
| `TAIL_NUMBER` | No |
| `TAIL_NUM` | No |
| `TAILNUMBER` | No |

No recognised aircraft or tail identifier field exists in the audited source file.

The generated evidence is stored in:

```text
reports/tables/tail_identifier_availability_audit.csv
reports/evaluation/entity_coverage_report.md
```

## Decision

Aircraft-level rotation reconstruction is not supported by the currently selected raw flight dataset.

Therefore, the following analyses are disabled for this project dataset:

- Aircraft-level prior-leg linkage.
- Tail-based delay propagation.
- Tail-based cascade depth and cascade duration.
- Cumulative delay within an aircraft rotation.
- Tail-based recovery episodes.
- LSTM, GRU, or Transformer sequences defined from aircraft rotations.
- Any claim that disruption propagated from one flight to another through the same physical aircraft.

## Permitted analyses

The lack of a tail identifier does not prevent the following analyses:

- Flight-level severe-arrival-delay classification.
- Cancellation classification.
- Arrival-delay regression on the defined completed-flight cohort.
- Carrier, airport, route, schedule, calendar, and historical past-only features.
- Airport-route network analysis based on observed origin-destination flights.
- Airport- or route-level disruption pressure and recovery patterns.
- Weather-enhanced flight-level and airport-level analysis after NOAA matching.
- Sequence models only if a separate, documented, leakage-safe non-tail temporal sequence design is later justified.

## Interpretation restrictions

This project must not use any proxy for tail identity.

In particular, it must not infer an aircraft rotation from:

- Same carrier.
- Same origin or destination.
- Same route.
- Same flight number.
- Consecutive flight times.
- Similar scheduled elapsed times.
- Any combination of the above without a verified aircraft identifier.

Any later discussion of propagation must be limited to airport-route network exposure, temporal disruption clustering, or other clearly labelled non-aircraft-level descriptive patterns.

## Consequence for later phases

The project scope remains an advanced retrospective aviation Data Science study, but the available data determines the claims that can be made.

Later phases will:

- Retain flight-level prediction, regression, regime analysis, carrier analysis, airport-route network analysis, weather matching, calibration, explainability, and error analysis.
- Exclude aircraft-rotation reconstruction, aircraft-level propagation, tail-based recovery survival analysis, and aircraft-rotation sequence models.
- Clearly state this source limitation in the data card, README, final technical report, and limitations section.

This is an evidence-led scope adjustment, not a data-quality failure. The project will be stronger because it does not claim aircraft-level findings without the identifiers required to support them.
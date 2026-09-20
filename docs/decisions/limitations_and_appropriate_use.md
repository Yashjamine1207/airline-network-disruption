# Appropriate Use and Limitations

## Project identity

**Airline Network Disruption Intelligence: Delay Prediction, Propagation, Recovery and Network Resilience** is an advanced retrospective Data Science portfolio study.

It is designed to demonstrate work relevant to Data Science, applied science, forecasting, operations research, decision science, transportation, and aviation analytics roles.

The project analyses historical flight-level operational data and historical weather observations to study severe delays, cancellations, aircraft-rotation disruption propagation, recovery, and airport-network resilience.

## Appropriate use

The project is intended for:

- Retrospective analysis of historical flight disruptions
- Reproducible machine-learning and time-series modelling experiments
- Analysis of aircraft-rotation sequences where identifiers and timestamps pass documented quality checks
- Descriptive airport-route network analysis
- Evaluation of weather, rotation, historical, and network features
- Portfolio, educational, and research demonstration purposes

All reported results are analytical findings based on the selected historical datasets and documented assumptions.

## Not a production airline system

This project is not:

- Airline operating software
- A flight-dispatch system
- A crew-assignment or aircraft-optimisation system
- A real-time disruption monitoring platform
- A booking, rebooking, or passenger-support system
- A live decision engine
- A safety-critical system
- A source of operational instructions for airlines, airports, passengers, or regulators

The project does not connect to live flight feeds, airline systems, booking systems, aircraft telemetry, passenger data, or real-time weather feeds.

## Data and interpretation limits

The official project pipeline uses only two external data sources:

1. The selected Kaggle Flight Delay and Cancellation Dataset.
2. NOAA Global Historical Climatology Network hourly weather observations.

The flight dataset currently available in the repository is a downloaded raw file named `flights_sample_3m.csv`. Its actual date coverage, row count, schema, missingness, licence or terms, and source details will be confirmed during the data-source audit.

Weather data will be acquired later and only for documented airports, stations, time ranges, and weather variables required by the analysis.

Model outputs represent statistical estimates or associations. They do not prove that a carrier, airport, route, weather condition, aircraft rotation, or network characteristic caused a delay, cancellation, or recovery outcome.

## Prediction and leakage limits

For every predictive task, the project will define a prediction timestamp.

A model may use only information that would have been available by that timestamp. Pre-flight models will not use actual departure or arrival outcomes from the predicted flight, final delay values, cancellation reasons, delay-cause fields, future weather, future aircraft legs, future airport conditions, or future network information.

## Interpretation limits

Aircraft-level propagation findings will be restricted to flight sequences with reliable tail identifiers, valid timestamp ordering, and documented eligibility checks.

Network measures, weather associations, SHAP values, and event-study results will be interpreted as descriptive or associational evidence. They will not be presented as causal proof.

Any analyst-capacity, threshold, cost, or utility scenario will be clearly labelled as an illustrative analytical assumption rather than an airline policy or financial estimate.
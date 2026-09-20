# Timezone and Timestamp Methodology

## Purpose

Flight time handling is a primary scientific requirement in this project.

The source flight data may contain dates and clock times recorded in local airport time. Because flights operate across multiple time zones and daylight-saving transitions, local timestamps from different airports must never be compared or subtracted directly.

This project will preserve the original local source fields and create auditable UTC-normalised timestamps for ordering, joining, sequencing, and time comparisons.

## Core timestamp rule

The project will retain both:

1. The original date and local time fields from the source dataset.
2. A reconstructed timezone-aware UTC timestamp where the required input fields are valid.

UTC timestamps will be used for:

- Flight ordering.
- Aircraft-rotation reconstruction.
- Prior-leg availability checks.
- Prediction timestamp construction.
- Weather matching.
- Network-window construction.
- Recovery-episode sequencing.
- Delay-propagation analysis.
- Any comparison between events at different airports.

## Airport-timezone mapping

The project will maintain a versioned airport-to-timezone mapping at:

```text
data/external/airport_timezone_mapping.csv
```

The mapping will contain, at minimum:

- Airport identifier used by the flight dataset.
- Airport name, where available.
- Latitude and longitude, where available.
- IANA timezone name.
- Mapping source or method.
- Mapping version.
- Mapping notes or exceptions.

Examples of valid IANA timezone names include:

```text
America/New_York
America/Chicago
America/Denver
America/Los_Angeles
America/Phoenix
Pacific/Honolulu
America/Anchorage
```

The implementation will use Python's standard-library `zoneinfo` module for timezone-aware conversion.

## Source local-time fields

The data-source audit will identify the actual source fields used to reconstruct timestamps.

Candidate fields may include:

- Flight date.
- Scheduled departure time.
- Scheduled arrival time.
- Actual departure time.
- Actual arrival time.
- Origin airport identifier.
- Destination airport identifier.
- Scheduled elapsed time.
- Actual elapsed time.
- Cancellation indicator.
- Diversion indicator.

No source field name will be assumed correct until the CSV schema and HTML dictionary have been audited.

## Timestamp types

The project may create the following timestamp fields when source data quality permits:

| Timestamp field | Description | Primary timezone |
|---|---|---|
| `flight_date_local` | Original flight date from the source dataset | Local source date |
| `scheduled_departure_local` | Reconstructed scheduled origin departure datetime | Origin airport timezone |
| `scheduled_departure_utc` | Scheduled origin departure converted to UTC | UTC |
| `scheduled_arrival_local` | Reconstructed scheduled destination arrival datetime | Destination airport timezone |
| `scheduled_arrival_utc` | Scheduled destination arrival converted to UTC | UTC |
| `actual_departure_local` | Reconstructed actual origin departure datetime | Origin airport timezone |
| `actual_departure_utc` | Actual origin departure converted to UTC | UTC |
| `actual_arrival_local` | Reconstructed actual destination arrival datetime | Destination airport timezone |
| `actual_arrival_utc` | Actual destination arrival converted to UTC | UTC |
| `prediction_timestamp_utc` | Scheduled departure UTC minus two hours | UTC |

The project will preserve the original raw time columns even after these derived timestamps are created.

## Local clock-time parsing

Many flight datasets store local times as integer or string clock values such as:

```text
5
45
930
1545
2400
```

During Phase 1, parsing logic will:

- Convert valid clock values to zero-padded `HH:MM` format.
- Interpret `5` as `00:05`.
- Interpret `45` as `00:45`.
- Interpret `930` as `09:30`.
- Interpret `1545` as `15:45`.
- Treat `2400` as midnight on the following local calendar day.
- Flag invalid values, including impossible hours or minutes.
- Preserve the original raw value for auditability.

## Date-rollover handling

A flight may cross midnight locally or in UTC.

The project must not assume that arrival occurs on the same local date as departure.

For scheduled and actual arrivals, the implementation will determine whether a date rollover is required using documented source fields and consistency checks, including:

- Scheduled elapsed time, when valid.
- Actual elapsed time, when valid.
- UTC ordering after timezone conversion.
- Flight cancellation and diversion status.
- Plausible flight-duration checks.

Any unresolved or ambiguous rollover case will be flagged rather than silently forced into a sequence.

## Daylight-saving-time handling

The project will use IANA timezone rules through `zoneinfo`.

The implementation must identify and document:

- Ambiguous local timestamps during the autumn daylight-saving transition.
- Non-existent local timestamps during the spring daylight-saving transition.
- The conversion policy applied to each affected record.
- Any records excluded because no reliable timestamp can be produced.

Timezone conversion tests must cover at least:

- A normal same-day flight.
- A flight that crosses local midnight.
- A flight with different origin and destination timezones.
- A spring daylight-saving transition.
- An autumn daylight-saving transition.
- A missing or invalid local clock-time value.

## Cancelled and diverted flights

Cancelled flights may lack actual departure and arrival timestamps.

For cancelled flights:

- Scheduled timestamps may still be reconstructed when source fields permit.
- Actual timestamp fields will remain missing when no valid actual outcome exists.
- Cancellation prediction must not require actual timestamps from the predicted flight.

Diverted flights may have incomplete or non-standard arrival outcomes.

The final treatment of diverted flights will be documented after the source-data audit confirms the available fields and outcome definitions.

## Quality flags

Every reconstructed timestamp must have auditable quality information.

Candidate flags include:

- `has_valid_origin_timezone`
- `has_valid_destination_timezone`
- `has_valid_scheduled_departure_time`
- `has_valid_scheduled_arrival_time`
- `has_valid_actual_departure_time`
- `has_valid_actual_arrival_time`
- `scheduled_departure_rollover_applied`
- `scheduled_arrival_rollover_applied`
- `actual_departure_rollover_applied`
- `actual_arrival_rollover_applied`
- `dst_ambiguous`
- `dst_nonexistent`
- `timestamp_conversion_status`
- `timestamp_conversion_reason`

## Prohibited calculations

The project must never:

- Subtract a local departure time at one airport from a local arrival time at another airport without timezone normalisation.
- Sort aircraft rotations by unvalidated local timestamps across airports.
- Match weather using a local timestamp without a documented timezone conversion.
- Use an actual timestamp from the predicted flight in a pre-flight predictive model.
- Guess an
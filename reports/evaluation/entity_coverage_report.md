# Entity Coverage and Rotation Feasibility Audit

## Purpose

This Phase 2 audit assesses the source support for longitudinal carrier analysis, airport-route network analysis, and later aircraft-rotation reconstruction. It is descriptive and does not infer aircraft rotations.

## Carrier identifier audit

- Observed airline-code/DOT-code pairs: 18.
- One-to-one observed pairs: 18.
- Pairs requiring mapping review: 0.
- Longitudinal carrier analysis must use the documented identifier policy, not carrier-code assumptions.

## Tail identifier audit

- No recognised tail identifier field (`TAIL_NUMBER`, `TAIL_NUM`, or `TAILNUMBER`) exists in this raw source file.
- Rotation analysis status: `not_supported_by_this_source_file_no_tail_identifier_field`.
- A tail field, if present, is necessary but not sufficient for propagation claims; UTC ordering, airport continuity, turnaround plausibility, and prior-leg availability still require later eligibility checks.

## Airport coverage by regime

| Regime | Role | Unique airports | Flight appearances |
|---|---|---:|---:|
| 2019 pre-COVID reference | destination | 360 | 757,673 |
| 2019 pre-COVID reference | origin | 360 | 757,673 |
| 2020 COVID operational shock | destination | 367 | 479,350 |
| 2020 COVID operational shock | origin | 365 | 479,350 |
| 2021-2023 recovery and transition | destination | 374 | 1,762,977 |
| 2021-2023 recovery and transition | origin | 374 | 1,762,977 |

## Highest-volume route by regime

| Regime | Route | Route flights | Share of regime flights |
|---|---|---:|---:|
| 2019 pre-COVID reference | SFO-LAX | 1,561 | 0.21% |
| 2020 COVID operational shock | SFO-LAX | 894 | 0.19% |
| 2021-2023 recovery and transition | LAX-SFO | 2,929 | 0.17% |

## Generated evidence

- Carrier mapping: `reports/tables/carrier_identifier_mapping.csv`
- Carrier coverage: `reports/tables/carrier_coverage_by_regime.csv`
- Airport coverage: `reports/tables/airport_coverage_by_regime.csv`
- Top routes: `reports/tables/top_routes_by_regime.csv`
- Tail availability: `reports/tables/tail_identifier_availability_audit.csv`

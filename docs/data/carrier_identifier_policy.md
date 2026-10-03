# Carrier Identifier Policy

## Audited source fields

The raw flight source contains `AIRLINE_CODE` and `DOT_CODE`. This audit records the observed mapping between both fields over the available period rather than assuming carrier-code stability from a dataset description.

## Policy

- Use `DOT_CODE` as the primary longitudinal carrier identifier when a stable one-to-one relationship is confirmed by the mapping audit.
- Retain `AIRLINE_CODE` for readable reporting and for detecting code-to-identifier mapping changes.
- Do not merge or rename carriers based only on assumptions about airline brands, mergers, or codes.
- Observed carrier-code/DOT pairs: 18.
- One-to-one observed pairs: 18.
- Pairs requiring review: 0.

The detailed evidence is stored in `reports/tables/carrier_identifier_mapping.csv`.

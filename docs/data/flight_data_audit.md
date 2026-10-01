# Flight Data Audit (Raw Kaggle CSV)

**Generated at:** 2026-09-22T02:09:43.858361Z

## Basic info

- Total rows: 3000000
- Total columns: 32

## Date coverage (FL_DATE)

- Parsing status: parsed_successfully
- Min date: 2019-01-01
- Max date: 2023-08-31
- Unique raw date values: 1704
- Valid parsed date rows: 3000000
- Invalid or missing date rows: 0
- Years present: [2019, 2020, 2021, 2022, 2023]
- Detected source format(s): {'YYYY-MM-DD': 3000000}

## Key uniqueness check

- Key columns: ['FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST']
- Total rows: 3000000
- Rows with non-null key: 3000000
- Duplicate key rows: 0
- Key is unique (among non-null): True

## Missingness summary (top 30)

Top 30 columns by % missing:

| column                  |   count_missing |   pct_missing |
|:------------------------|----------------:|--------------:|
| CANCELLATION_CODE       |         2920860 |  97.362       |
| DELAY_DUE_LATE_AIRCRAFT |         2466137 |  82.2046      |
| DELAY_DUE_CARRIER       |         2466137 |  82.2046      |
| DELAY_DUE_SECURITY      |         2466137 |  82.2046      |
| DELAY_DUE_NAS           |         2466137 |  82.2046      |
| DELAY_DUE_WEATHER       |         2466137 |  82.2046      |
| ARR_DELAY               |           86198 |   2.87327     |
| ELAPSED_TIME            |           86198 |   2.87327     |
| AIR_TIME                |           86198 |   2.87327     |
| WHEELS_ON               |           79944 |   2.6648      |
| TAXI_IN                 |           79944 |   2.6648      |
| ARR_TIME                |           79942 |   2.66473     |
| WHEELS_OFF              |           78806 |   2.62687     |
| TAXI_OUT                |           78806 |   2.62687     |
| DEP_DELAY               |           77644 |   2.58813     |
| DEP_TIME                |           77615 |   2.58717     |
| CRS_ELAPSED_TIME        |              14 |   0.000466667 |
| DEST_CITY               |               0 |   0           |
| CRS_DEP_TIME            |               0 |   0           |
| DEST                    |               0 |   0           |
| DOT_CODE                |               0 |   0           |
| FL_NUMBER               |               0 |   0           |
| ORIGIN                  |               0 |   0           |
| ORIGIN_CITY             |               0 |   0           |
| AIRLINE                 |               0 |   0           |
| AIRLINE_DOT             |               0 |   0           |
| FL_DATE                 |               0 |   0           |
| AIRLINE_CODE            |               0 |   0           |
| DIVERTED                |               0 |   0           |
| CANCELLED               |               0 |   0           |

## Categorical field summaries

### AIRLINE_CODE

- Unique values: 18
- Missing values: 0

Top values:

| value   |   count |
|:--------|--------:|
| WN      |  576470 |
| DL      |  395239 |
| AA      |  383106 |
| OO      |  343737 |
| UA      |  254504 |
| YX      |  143107 |
| MQ      |  121256 |
| B6      |  112844 |
| 9E      |  112463 |
| OH      |  107050 |
| AS      |  100467 |
| NK      |   95711 |
| YV      |   65012 |
| F9      |   64466 |
| G4      |   52738 |
| HA      |   32114 |
| QX      |   20634 |
| EV      |   19082 |

### DOT_CODE

- Unique values: 18
- Missing values: 0

Top values:

|   value |   count |
|--------:|--------:|
|   19393 |  576470 |
|   19790 |  395239 |
|   19805 |  383106 |
|   20304 |  343737 |
|   19977 |  254504 |
|   20452 |  143107 |
|   20398 |  121256 |
|   20409 |  112844 |
|   20363 |  112463 |
|   20397 |  107050 |
|   19930 |  100467 |
|   20416 |   95711 |
|   20378 |   65012 |
|   20436 |   64466 |
|   20368 |   52738 |
|   19690 |   32114 |
|   19687 |   20634 |
|   20366 |   19082 |

### ORIGIN

- Unique values: 380
- Missing values: 0

Top values:

| value   |   count |
|:--------|--------:|
| ATL     |  153556 |
| DFW     |  130334 |
| ORD     |  122296 |
| DEN     |  119919 |
| CLT     |   94304 |
| LAX     |   85872 |
| PHX     |   74815 |
| LAS     |   73470 |
| SEA     |   70906 |
| MCO     |   63883 |
| LGA     |   62574 |
| IAH     |   62542 |
| DTW     |   62279 |
| MSP     |   60061 |
| SFO     |   59531 |
| BOS     |   55394 |
| DCA     |   53283 |
| EWR     |   52989 |
| SLC     |   52027 |
| JFK     |   50466 |

### DEST

- Unique values: 380
- Missing values: 0

Top values:

| value   |   count |
|:--------|--------:|
| ATL     |  153569 |
| DFW     |  129770 |
| ORD     |  123334 |
| DEN     |  119592 |
| CLT     |   95413 |
| LAX     |   85621 |
| PHX     |   75605 |
| LAS     |   73462 |
| SEA     |   70832 |
| MCO     |   63818 |
| IAH     |   62201 |
| DTW     |   62187 |
| LGA     |   61968 |
| MSP     |   59729 |
| SFO     |   59008 |
| BOS     |   55611 |
| DCA     |   53501 |
| EWR     |   52824 |
| SLC     |   52063 |
| JFK     |   50268 |

### CANCELLATION_CODE

- Unique values: 4
- Missing values: 2920860

Top values:

| value   |   count |
|:--------|--------:|
| B       |   28772 |
| D       |   24417 |
| A       |   19476 |
| C       |    6475 |

### DIVERTED

- Unique values: 2
- Missing values: 0

Top values:

|   value |   count |
|--------:|--------:|
|       0 | 2992944 |
|       1 |    7056 |

### CANCELLED

- Unique values: 2
- Missing values: 0

Top values:

|   value |   count |
|--------:|--------:|
|       0 | 2920860 |
|       1 |   79140 |

## Numeric delay field summaries

### DEP_DELAY

- Non-missing count: 2922356, min: -90.0, max: 2966.0, mean: 10.12, median: -2.00

### ARR_DELAY

- Non-missing count: 2913802, min: -96.0, max: 2934.0, mean: 4.26, median: -7.00

### DELAY_DUE_CARRIER

- Non-missing count: 533863, min: 0.0, max: 2934.0, mean: 24.76, median: 4.00

### DELAY_DUE_WEATHER

- Non-missing count: 533863, min: 0.0, max: 1653.0, mean: 3.99, median: 0.00

### DELAY_DUE_NAS

- Non-missing count: 533863, min: 0.0, max: 1741.0, mean: 13.16, median: 0.00

### DELAY_DUE_SECURITY

- Non-missing count: 533863, min: 0.0, max: 1185.0, mean: 0.15, median: 0.00

### DELAY_DUE_LATE_AIRCRAFT

- Non-missing count: 533863, min: 0.0, max: 2557.0, mean: 25.47, median: 0.00

## Notes / Next steps

- Use this audit to confirm date ranges before defining regime splits (2019–2021, 2022, 2023).
- Check carrier codes (AIRLINE_CODE vs DOT_CODE) for longitudinal consistency.
- Inspect high-missingness columns before feature engineering.
- Use cancellation/diversion summaries to define cancellation and diversion cohorts.

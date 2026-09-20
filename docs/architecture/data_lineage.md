# Data Lineage

```mermaid
flowchart LR

    A[data/raw/flights_kaggle/<br/>flights_sample_3m.csv]
    B[data/raw/flights_kaggle/<br/>dictionary.html]
    C[data/raw/noaa_ghcnh/<br/>selected NOAA GHCNh extracts]

    A --> D[Raw-source validation]
    B --> D
    C --> E[Weather-source validation]

    D --> F[data/external/<br/>source register and field mapping]
    E --> F

    D --> G[data/interim/flights_standardised/<br/>standardised flight Parquet]
    E --> H[data/interim/weather_standardised/<br/>standardised weather Parquet]

    F --> I[data/external/<br/>airport timezone mapping]
    F --> J[data/external/<br/>airport-station mapping]

    G --> K[Timezone reconstruction]
    I --> K

    K --> L[data/interim/flights_standardised/<br/>UTC-normalised flight table]

    H --> M[Past-only weather matching]
    J --> M
    L --> M

    L --> N[Flight data-quality audit]
    L --> O[Target construction]
    L --> P[Historical past-only features]
    L --> Q[Aircraft-rotation reconstruction]
    L --> R[Past-only network-window construction]

    M --> S[data/processed/flight_panel/<br/>weather-enriched analytical panel]
    O --> S
    P --> S
    Q --> S
    R --> S

    Q --> T[data/processed/rotations/<br/>eligible rotations and propagation tables]
    Q --> U[data/processed/survival_episodes/<br/>recovery and censoring episodes]
    R --> V[data/processed/network_windows/<br/>airport-route graph metrics]

    S --> W[data/features/tabular/<br/>tabular model matrices]
    S --> X[data/features/sequences/<br/>past-only sequence matrices]
    V --> W
    T --> W
    T --> X

    W --> Y[Baselines, statistical models,<br/>LightGBM and XGBoost]
    X --> Z[TensorFlow LSTM, GRU,<br/>compact Transformer]

    Y --> AA[Metrics, calibration,<br/>error analysis and explainability]
    Z --> AA

    T --> AB[Propagation analysis]
    U --> AC[Survival and recovery analysis]
    V --> AD[Network resilience analysis]

    AA --> AE[reports/tables and reports/figures]
    AB --> AE
    AC --> AE
    AD --> AE

    AE --> AF[Final technical report,<br/>README, optional read-only presentation]
```

## Lineage controls

- `data/raw/` contains immutable downloaded source material and is excluded from Git.
- `data/interim/`, `data/processed/`, and `data/features/` contain generated outputs and are excluded from Git.
- `data/external/` stores small, versioned documentation and mappings required to reproduce the pipeline.
- Every transformation must have a documented script, configuration, input source, output location, and quality check.
- Flight and weather lineage must retain source file identity, source schema version, processing time, and file hashes where feasible.
- Every feature used in a predictive model must be traceable to its source fields, transformation logic, availability time, and leakage-control rule.
- Only small permitted or synthetic examples may be stored in `data/sample/`.
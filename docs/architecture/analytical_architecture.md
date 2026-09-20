# Analytical Architecture

```mermaid
flowchart TD

    A[Raw Kaggle flight data<br/>flights_sample_3m.csv] --> B[Source audit and ingestion]
    A2[Raw Kaggle column dictionary<br/>dictionary.html] --> B

    W[NOAA GHCNh weather extracts<br/>retrieved later after airport audit] --> C[Weather standardisation and station-quality checks]

    B --> D[Flight standardisation<br/>Parquet working tables]
    D --> E[Airport timezone mapping]
    E --> F[Local timestamp reconstruction]
    F --> G[UTC normalisation and timestamp-quality flags]

    G --> H[Data quality and regime analysis]
    C --> I[Airport-to-station mapping]
    I --> J[Past-only weather matching]
    G --> K[Target construction]
    G --> L[Past-only historical features]
    G --> M[Eligible aircraft rotation reconstruction]
    G --> N[Past-only airport-route network windows]

    K --> O[Severe delay classification]
    K --> P[Cancellation classification]
    K --> Q[Arrival-delay regression]
    M --> R[Propagation and cascade analysis]
    M --> S[Recovery episodes and survival analysis]

    L --> T[Leakage-safe feature sets]
    J --> T
    M --> T
    N --> T

    T --> U[Historical and statistical baselines]
    T --> V[Logistic, Linear, Ridge, Elastic Net]
    T --> W2[LightGBM and XGBoost]
    T --> X[TensorFlow LSTM, GRU, compact Transformer challengers]

    U --> Y[Chronological validation and expanding-window backtests]
    V --> Y
    W2 --> Y
    X --> Y

    Y --> Z[Calibration, ranking scenarios, explainability, error analysis]
    R --> AA[Propagation and network resilience evidence]
    S --> AB[Recovery and survival evidence]

    Z --> AC[Final evidence-led model selection]
    AA --> AD[Final technical report]
    AB --> AD
    AC --> AD

    AD --> AE[Optional read-only portfolio presentation]
```

## Architecture rules

- The official pipeline contains exactly two external sources: the selected Kaggle flight dataset and NOAA GHCNh historical weather observations.
- Raw source files remain immutable under `data/raw/` and are excluded from Git.
- Original local time fields are retained, while UTC-normalised timestamps are used for ordering, joining, weather matching, and sequence construction.
- Every predictive feature must be available by the documented prediction timestamp.
- Model evaluation uses chronological validation and an untouched final temporal test period.
- Aircraft-level propagation and recovery claims are restricted to eligible validated tail-number sequences.
- LSTM, GRU, and Transformer models are challengers evaluated only after stable tabular pipelines exist.
- The project is retrospective analytical research, not live airline operational software.
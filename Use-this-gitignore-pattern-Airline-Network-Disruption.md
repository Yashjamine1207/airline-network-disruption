# `.gitignore` pattern — Airline Network Disruption Intelligence

Copy this into the root `.gitignore` file.

```gitignore
# Environment variables, local credentials and secret material
.env
.env.*
!.env.example
*.pem
*.key
*.crt
secrets/
.kaggle/

# Python environments and bytecode
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/
.python-version

# Python tooling and test output
.pytest_cache/
.mypy_cache/
.pyre/
.pytype/
.ruff_cache/
.coverage
.coverage.*
htmlcov/

# Jupyter
.ipynb_checkpoints/

# External/raw and generated data: never commit full flight or NOAA files
data/raw/*
data/interim/*
data/processed/*
data/features/*

# Preserve only source documentation and tiny safe/synthetic recruiter examples
!data/external/
!data/external/dataset_readme.md
!data/external/source_register.csv
!data/external/flight_field_mapping.md
!data/external/ghcnh_field_mapping.md
!data/external/airport_station_mapping.csv
!data/external/airport_timezone_mapping.csv
!data/sample/
!data/sample/**

# Analytical engines, caches, temporary database files
*.duckdb
*.duckdb.wal
*.db
*.sqlite
*.sqlite3
.cache/
tmp/
temp/

# Models, training output and experiment tracking
models/artifacts/*
models/checkpoints/*
mlruns/
mlartifacts/

# TensorFlow/Keras output
*.h5
*.hdf5
*.keras
*.ckpt
*.ckpt.index
*.ckpt.data-*
checkpoint
logs/tensorboard/

# Logs
logs/
*.log
*.out

# Temporary/generated reports; curate final small figures/tables deliberately
reports/generated/
reports/drafts/
reports/tmp/
figures/generated/

# Optional Streamlit local configuration
.streamlit/secrets.toml
.streamlit/config.toml

# IDE and operating-system files
.vscode/settings.json
.vscode/launch.json
.idea/
.DS_Store
Thumbs.db

# Packaging and build output
build/
dist/
*.egg-info/
.eggs/

# Optional local Power BI/presentation autosave output
presentation/*.pbix
presentation/*.pbit
presentation/~$*
```

## Project-specific rules

- Never commit the full Kaggle flight dataset, full NOAA GHCNh downloads, derived Parquet panel, flight sequences, network windows, or feature matrices.
- Keep only a tiny safe sample in `data/sample/`. A synthetic example is preferred because it removes ambiguity about third-party redistribution.
- Store source URLs, access dates, field definitions, download instructions, airport/station mappings, and data-quality metadata in `data/external/` and `docs/data/` rather than committing source data.
- Never commit Kaggle credentials, any `.kaggle/` content, an environment file, model checkpoints, MLflow artefacts, DuckDB databases, or local report drafts.
- Curated small final figures/tables can be committed when they make the README/reports understandable. Do not commit repeated intermediate outputs or large binaries.
- Run `git status --ignored` and inspect every staged file before the first public push.
- Ignore rules do not remove a file already tracked by Git. If a large, restricted, or sensitive file was committed accidentally, remove it from version control before publishing. If a credential was ever committed, rotate it immediately.

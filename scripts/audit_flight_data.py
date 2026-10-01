"""
Audit script for the raw Kaggle flight delay/cancellation CSV.

Purpose
-------
- Confirm schema (columns, dtypes).
- Confirm row count and date coverage.
- Check key uniqueness (flight-level key).
- Quantify missingness per column.
- Detect duplicates.
- Summarise key categorical fields (carriers, airports, cancellation codes, diversion).
- Write machine-readable and human-readable audit reports.

This is the first step of Phase 2 (Data quality, EDA, regime analysis).
It establishes data fitness before any EDA plots or modelling.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# =========================
# CONFIGURATION
# =========================

# Path to the raw Kaggle flight CSV (adjust if your filename differs)
RAW_FLIGHT_CSV = Path("data/raw/flights_kaggle/flights_sample_3m.csv")

# Output paths
AUDIT_JSON_PATH = Path("data/external/flight_data_audit.json")
AUDIT_SUMMARY_CSV_PATH = Path("data/external/flight_data_audit_summary.csv")
AUDIT_MD_PATH = Path("docs/data/flight_data_audit.md")

# Ensure output directories exist
AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

# Define the expected primary key for a flight record.
# We'll test uniqueness for this combination.
# Typical candidate: [FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, DEST]
# Adjust if your dictionary suggests a different natural key.
FLIGHT_KEY_COLS = [
    "FL_DATE",
    "AIRLINE_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
]

# =========================
# HELPER FUNCTIONS
# =========================


def load_raw_flights(path: Path) -> pd.DataFrame:
    """
    Load the raw flight CSV with minimal parsing.
    - Keep columns as-is.
    - Do not infer dates yet; we'll inspect them.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df


def compute_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missingness per column:
    - count_missing: number of missing/empty entries.
    - pct_missing: percentage of rows missing.
    """
    # Treat empty strings and pure whitespace as missing as well as NaN/None
    def is_missing(series: pd.Series) -> pd.Series:
        if series.dtype == "object":
            return series.isna() | (series.str.strip() == "")
        return series.isna()

    missing_counts = df.apply(is_missing).sum()
    total = len(df)
    summary = pd.DataFrame(
        {
            "column": missing_counts.index,
            "count_missing": missing_counts.values,
            "pct_missing": (missing_counts.values / total) * 100.0,
        }
    )
    summary = summary.sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return summary


def check_key_uniqueness(df: pd.DataFrame, key_cols: list) -> dict:
    """
    Check if the given key columns uniquely identify rows.
    Returns a dict with:
    - key_columns
    - total_rows
    - non_null_key_rows: rows where all key columns are non-missing
    - duplicate_key_rows: number of rows that are duplicates w.r.t. the key
    - is_unique_key: True if no duplicates among non-null key rows
    """
    # Make a copy to avoid modifying original
    tmp = df[list(key_cols)].copy()

    # Identify missingness in key columns (treat empty strings as missing)
    def is_missing_series(s: pd.Series) -> pd.Series:
        if s.dtype == "object":
            return s.isna() | (s.str.strip() == "")
        return s.isna()

    key_missing_mask = pd.concat([is_missing_series(tmp[col]) for col in key_cols], axis=1).any(axis=1)

    non_null_df = tmp[~key_missing_mask].copy()
    total_rows = len(df)
    non_null_rows = len(non_null_df)

    # Count duplicates
    duplicated_mask = non_null_df.duplicated(keep=False)
    duplicate_rows = int(duplicated_mask.sum())
    is_unique = duplicate_rows == 0

    return {
        "key_columns": key_cols,
        "total_rows": int(total_rows),
        "non_null_key_rows": int(non_null_rows),
        "duplicate_key_rows": int(duplicate_rows),
        "is_unique_key": is_unique,
    }


def summarise_categorical_field(df: pd.DataFrame, col: str, top_n: int = 20) -> dict:
    """
    Produce a summary for a categorical field:
    - n_unique
    - top_n values with counts
    - missing count
    """
    series = df[col]
    # Treat empty strings as missing
    if series.dtype == "object":
        missing_mask = series.isna() | (series.str.strip() == "")
    else:
        missing_mask = series.isna()

    n_missing = int(missing_mask.sum())
    non_missing = series[~missing_mask]
    n_unique = non_missing.nunique()
    top_counts = non_missing.value_counts().head(top_n)

    return {
        "column": col,
        "n_unique": int(n_unique),
        "n_missing": n_missing,
        "top_values": {str(k): int(v) for k, v in top_counts.items()},
    }


def compute_date_coverage(df: pd.DataFrame, date_col: str = "FL_DATE") -> dict:
    """
    Parse the flight-date column using supported source formats and report
    actual coverage.
    Supported formats:
    - YYYY-MM-DD  e.g. 2019-01-01
    - YYYYMMDD    e.g. 20190101
    - MM/DD/YYYY  e.g. 01/01/2019
    The source format is not assumed from the data dictionary alone. The
    function records which format(s) were actually found in the raw CSV.
    """
    if date_col not in df.columns:
        return {
            "date_column": date_col,
            "min_date": None,
            "max_date": None,
            "n_unique_raw_date_values": 0,
            "n_valid_dates": 0,
            "n_invalid_or_missing_dates": int(len(df)),
            "years_present": [],
            "detected_date_formats": {},
            "status": "date_column_not_found",
        }
    raw_dates = df[date_col].astype("string").str.strip()
    raw_dates = raw_dates.mask(raw_dates.isin(["", "nan", "None", "<NA>"]))
    parsed_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # Parse only still-unparsed non-missing rows. This prevents a later
    # format from overwriting a date already parsed using an earlier format.
    supported_formats = {
        "YYYY-MM-DD": "%Y-%m-%d",
        "YYYYMMDD": "%Y%m%d",
        "MM/DD/YYYY": "%m/%d/%Y",
    }
    detected_date_formats = {}
    for format_name, format_string in supported_formats.items():
        remaining_mask = raw_dates.notna() & parsed_dates.isna()
        if not remaining_mask.any():
            break
        candidate_dates = pd.to_datetime(
            raw_dates.loc[remaining_mask],
            format=format_string,
            errors="coerce",
        )
        successfully_parsed = candidate_dates.notna()
        parsed_index = candidate_dates.index[successfully_parsed]
        parsed_dates.loc[parsed_index] = candidate_dates.loc[parsed_index]
        detected_date_formats[format_name] = int(successfully_parsed.sum())

    # Do not include unused formats in the final report.
    detected_date_formats = {
        format_name: count
        for format_name, count in detected_date_formats.items()
        if count > 0
    }

    valid_dates = parsed_dates.dropna()
    n_unique_raw_date_values = int(raw_dates.dropna().nunique())
    n_valid_dates = int(valid_dates.shape[0])
    n_invalid_or_missing_dates = int(parsed_dates.isna().sum())

    if valid_dates.empty:
        return {
            "date_column": date_col,
            "min_date": None,
            "max_date": None,
            "n_unique_raw_date_values": n_unique_raw_date_values,
            "n_valid_dates": n_valid_dates,
            "n_invalid_or_missing_dates": n_invalid_or_missing_dates,
            "years_present": [],
            "detected_date_formats": detected_date_formats,
            "status": "no_valid_dates_parsed",
        }

    return {
        "date_column": date_col,
        "min_date": valid_dates.min().date().isoformat(),
        "max_date": valid_dates.max().date().isoformat(),
        "n_unique_raw_date_values": n_unique_raw_date_values,
        "n_valid_dates": n_valid_dates,
        "n_invalid_or_missing_dates": n_invalid_or_missing_dates,
        "years_present": sorted(valid_dates.dt.year.unique().tolist()),
        "detected_date_formats": detected_date_formats,
        "status": "parsed_successfully",
    }


def build_audit_report(df: pd.DataFrame) -> dict:
    """
    Build a comprehensive audit report as a dictionary.
    """
    total_rows = len(df)
    columns = list(df.columns)
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    # Date coverage
    date_coverage = compute_date_coverage(df, "FL_DATE")

    # Key uniqueness
    key_audit = check_key_uniqueness(df, FLIGHT_KEY_COLS)

    # Missingness
    missing_summary = compute_missingness_summary(df)

    # Categorical summaries for key fields
    categorical_fields = [
        "AIRLINE_CODE",
        "DOT_CODE",
        "ORIGIN",
        "DEST",
        "CANCELLATION_CODE",
        "DIVERTED",
        "CANCELLED",
    ]
    # Filter to fields that exist
    categorical_fields = [c for c in categorical_fields if c in df.columns]

    categorical_summaries = {}
    for col in categorical_fields:
        categorical_summaries[col] = summarise_categorical_field(df, col)

    # Simple extreme-value checks for numeric delay fields
    numeric_delay_cols = [
        "DEP_DELAY",
        "ARR_DELAY",
        "DELAY_DUE_CARRIER",
        "DELAY_DUE_WEATHER",
        "DELAY_DUE_NAS",
        "DELAY_DUE_SECURITY",
        "DELAY_DUE_LATE_AIRCRAFT",
    ]
    numeric_delay_cols = [c for c in numeric_delay_cols if c in df.columns]

    numeric_summaries = {}
    for col in numeric_delay_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        count = series.count()
        if count == 0:
            numeric_summaries[col] = {
                "count_non_missing": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
            }
        else:
            numeric_summaries[col] = {
                "count_non_missing": int(count),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
            }

    audit = {
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "date_coverage": date_coverage,
        "key_uniqueness": key_audit,
        "missingness_summary": missing_summary.to_dict(orient="records"),
        "categorical_summaries": categorical_summaries,
        "numeric_delay_summaries": numeric_summaries,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return audit


def write_audit_outputs(audit: dict, json_path: Path, summary_csv_path: Path, md_path: Path):
    """
    Write audit outputs:
    - JSON (full machine-readable report)
    - CSV (missingness summary)
    - Markdown (human-readable summary)
    """
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # CSV: missingness summary
    missing_df = pd.DataFrame(audit["missingness_summary"])
    missing_df.to_csv(summary_csv_path, index=False)

    # Markdown report
    md_lines = []
    md_lines.append("# Flight Data Audit (Raw Kaggle CSV)")
    md_lines.append("")
    md_lines.append(f"**Generated at:** {audit['generated_at']}")
    md_lines.append("")

    md_lines.append("## Basic info")
    md_lines.append("")
    md_lines.append(f"- Total rows: {audit['total_rows']}")
    md_lines.append(f"- Total columns: {len(audit['columns'])}")
    md_lines.append("")

    md_lines.append("## Date coverage (FL_DATE)")
    md_lines.append("")
    dc = audit["date_coverage"]
    md_lines.append(f"- Parsing status: {dc['status']}")
    md_lines.append(f"- Min date: {dc['min_date']}")
    md_lines.append(f"- Max date: {dc['max_date']}")
    md_lines.append(f"- Unique raw date values: {dc['n_unique_raw_date_values']}")
    md_lines.append(f"- Valid parsed date rows: {dc['n_valid_dates']}")
    md_lines.append(f"- Invalid or missing date rows: {dc['n_invalid_or_missing_dates']}")
    md_lines.append(f"- Years present: {dc['years_present']}")
    md_lines.append(f"- Detected source format(s): {dc['detected_date_formats']}")
    md_lines.append("")

    md_lines.append("## Key uniqueness check")
    md_lines.append("")
    ku = audit["key_uniqueness"]
    md_lines.append(f"- Key columns: {ku['key_columns']}")
    md_lines.append(f"- Total rows: {ku['total_rows']}")
    md_lines.append(f"- Rows with non-null key: {ku['non_null_key_rows']}")
    md_lines.append(f"- Duplicate key rows: {ku['duplicate_key_rows']}")
    md_lines.append(f"- Key is unique (among non-null): {ku['is_unique_key']}")
    md_lines.append("")

    md_lines.append("## Missingness summary (top 30)")
    md_lines.append("")
    md_lines.append("Top 30 columns by % missing:")
    md_lines.append("")
    miss_df = pd.DataFrame(audit["missingness_summary"]).head(30)
    md_lines.append(miss_df.to_markdown(index=False))
    md_lines.append("")

    md_lines.append("## Categorical field summaries")
    md_lines.append("")
    for col, summary in audit["categorical_summaries"].items():
        md_lines.append(f"### {col}")
        md_lines.append("")
        md_lines.append(f"- Unique values: {summary['n_unique']}")
        md_lines.append(f"- Missing values: {summary['n_missing']}")
        md_lines.append("")
        md_lines.append("Top values:")
        md_lines.append("")
        top_df = pd.DataFrame(
            {"value": list(summary["top_values"].keys()), "count": list(summary["top_values"].values())}
        )
        md_lines.append(top_df.to_markdown(index=False))
        md_lines.append("")

    md_lines.append("## Numeric delay field summaries")
    md_lines.append("")
    for col, summary in audit["numeric_delay_summaries"].items():
        md_lines.append(f"### {col}")
        md_lines.append("")
        mean_str = f"{summary['mean']:.2f}" if summary["mean"] is not None else "N/A"
        median_str = f"{summary['median']:.2f}" if summary["median"] is not None else "N/A"
        md_lines.append(
            f"- Non-missing count: {summary['count_non_missing']}, "
            f"min: {summary['min']}, max: {summary['max']}, "
            f"mean: {mean_str}, median: {median_str}"
        )
        md_lines.append("")

    md_lines.append("## Notes / Next steps")
    md_lines.append("")
    md_lines.append("- Use this audit to confirm date ranges before defining regime splits (2019–2021, 2022, 2023).")
    md_lines.append("- Check carrier codes (AIRLINE_CODE vs DOT_CODE) for longitudinal consistency.")
    md_lines.append("- Inspect high-missingness columns before feature engineering.")
    md_lines.append("- Use cancellation/diversion summaries to define cancellation and diversion cohorts.")
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


# =========================
# MAIN
# =========================


def main():
    print("Loading raw flight CSV...")
    df = load_raw_flights(RAW_FLIGHT_CSV)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    print("Building audit report...")
    audit = build_audit_report(df)

    print("Writing audit outputs...")
    write_audit_outputs(audit, AUDIT_JSON_PATH, AUDIT_SUMMARY_CSV_PATH, AUDIT_MD_PATH)

    print("Audit complete.")
    print(f"- JSON: {AUDIT_JSON_PATH}")
    print(f"- Missingness CSV: {AUDIT_SUMMARY_CSV_PATH}")
    print(f"- Markdown report: {AUDIT_MD_PATH}")


if __name__ == "__main__":
    main()
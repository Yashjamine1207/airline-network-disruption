"""Phase 2: carrier, tail, airport, and route coverage audit.

Reads the immutable raw flight CSV in chunks and documents whether the data
supports longitudinal carrier analysis, aircraft rotation reconstruction, and
airport-route network analysis. It makes no causal or predictive claims.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("data/raw/flights_kaggle/flights_sample_3m.csv")
CHUNK_SIZE = 250_000

CARRIER_MAPPING_PATH = Path("reports/tables/carrier_identifier_mapping.csv")
CARRIER_REGIME_PATH = Path("reports/tables/carrier_coverage_by_regime.csv")
AIRPORT_REGIME_PATH = Path("reports/tables/airport_coverage_by_regime.csv")
TOP_ROUTE_PATH = Path("reports/tables/top_routes_by_regime.csv")
TAIL_AUDIT_PATH = Path("reports/tables/tail_identifier_availability_audit.csv")
POLICY_PATH = Path("docs/data/carrier_identifier_policy.md")
COVERAGE_REPORT_PATH = Path("reports/evaluation/entity_coverage_report.md")

REQUIRED_COLUMNS = {
    "FL_DATE", "AIRLINE_CODE", "DOT_CODE", "ORIGIN", "DEST", "CANCELLED", "DIVERTED"
}
TAIL_CANDIDATES = ("TAIL_NUMBER", "TAIL_NUM", "TAILNUMBER")
REGIMES = (
    "2019 pre-COVID reference",
    "2020 COVID operational shock",
    "2021-2023 recovery and transition",
)


def make_directories() -> None:
    for path in (
        CARRIER_MAPPING_PATH,
        CARRIER_REGIME_PATH,
        AIRPORT_REGIME_PATH,
        TOP_ROUTE_PATH,
        TAIL_AUDIT_PATH,
        POLICY_PATH,
        COVERAGE_REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def assign_regime(dates: pd.Series) -> pd.Series:
    conditions = [
        dates.between("2019-01-01", "2019-12-31"),
        dates.between("2020-01-01", "2020-12-31"),
        dates.between("2021-01-01", "2023-08-31"),
    ]
    return pd.Series(np.select(conditions, REGIMES, default="outside_audited_coverage"), index=dates.index)


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator * 100.0 / denominator, np.nan)


def combine(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True).groupby(keys, as_index=False).sum(numeric_only=True)


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def main() -> None:
    make_directories()

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    header = pd.read_csv(RAW_CSV, nrows=0).columns.tolist()
    missing = sorted(REQUIRED_COLUMNS.difference(header))
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")

    tail_column = next((column for column in TAIL_CANDIDATES if column in header), None)
    usecols = sorted(REQUIRED_COLUMNS.union({tail_column} if tail_column else set()))

    carrier_mapping_parts: list[pd.DataFrame] = []
    carrier_regime_parts: list[pd.DataFrame] = []
    route_parts: list[pd.DataFrame] = []
    airport_parts: list[pd.DataFrame] = []
    regime_parts: list[pd.DataFrame] = []

    tail_total_rows = 0
    tail_non_missing_rows = 0
    tail_unique_values: set[str] = set()

    print("Starting Phase 2 carrier, tail, airport, and route coverage audit...")
    for chunk_number, chunk in enumerate(
        pd.read_csv(RAW_CSV, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        dates = pd.to_datetime(chunk["FL_DATE"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise ValueError("Unexpected invalid FL_DATE encountered after the completed date audit.")

        frame = pd.DataFrame({
            "regime": assign_regime(dates),
            "airline_code": clean_text(chunk["AIRLINE_CODE"]),
            "dot_code": clean_text(chunk["DOT_CODE"]),
            "origin": clean_text(chunk["ORIGIN"]),
            "destination": clean_text(chunk["DEST"]),
            "cancelled": pd.to_numeric(chunk["CANCELLED"], errors="coerce"),
            "diverted": pd.to_numeric(chunk["DIVERTED"], errors="coerce"),
        })

        if not frame["cancelled"].isin([0, 1]).all() or not frame["diverted"].isin([0, 1]).all():
            raise ValueError("CANCELLED or DIVERTED contains a value other than 0 or 1.")

        frame = frame[frame["regime"].isin(REGIMES)].copy()
        frame["route"] = frame["origin"] + "-" + frame["destination"]
        frame["scheduled_flights"] = 1
        frame["cancelled_flights"] = frame["cancelled"].astype("int64")
        frame["diverted_flights"] = frame["diverted"].astype("int64")

        carrier_mapping_parts.append(
            frame.groupby(["airline_code", "dot_code"], as_index=False)
            .agg(scheduled_flights=("scheduled_flights", "sum"))
        )
        carrier_regime_parts.append(
            frame.groupby(["regime", "airline_code", "dot_code"], as_index=False)
            .agg(
                scheduled_flights=("scheduled_flights", "sum"),
                cancelled_flights=("cancelled_flights", "sum"),
                diverted_flights=("diverted_flights", "sum"),
            )
        )
        route_parts.append(
            frame.groupby(["regime", "route"], as_index=False)
            .agg(scheduled_flights=("scheduled_flights", "sum"))
        )
        regime_parts.append(
            frame.groupby("regime", as_index=False)
            .agg(scheduled_flights=("scheduled_flights", "sum"))
        )

        airport_long = pd.concat(
            [
                frame[["regime", "origin", "scheduled_flights"]].rename(columns={"origin": "airport"}).assign(role="origin"),
                frame[["regime", "destination", "scheduled_flights"]].rename(columns={"destination": "airport"}).assign(role="destination"),
            ],
            ignore_index=True,
        )
        airport_parts.append(
            airport_long.groupby(["regime", "role", "airport"], as_index=False)
            .agg(flight_appearances=("scheduled_flights", "sum"))
        )

        if tail_column:
            tail_values = clean_text(chunk[tail_column])
            tail_total_rows += len(tail_values)
            tail_non_missing_rows += int(tail_values.notna().sum())
            tail_unique_values.update(tail_values.dropna().unique().tolist())

        print(f"Processed chunk {chunk_number:,}: {len(chunk):,} rows")

    mapping = combine(carrier_mapping_parts, ["airline_code", "dot_code"])
    carrier_by_regime = combine(carrier_regime_parts, ["regime", "airline_code", "dot_code"])
    carrier_by_regime["cancellation_rate_pct"] = safe_percent(
        carrier_by_regime["cancelled_flights"], carrier_by_regime["scheduled_flights"]
    )
    carrier_by_regime["diversion_rate_pct"] = safe_percent(
        carrier_by_regime["diverted_flights"], carrier_by_regime["scheduled_flights"]
    )

    codes_per_dot = mapping.groupby("dot_code")["airline_code"].nunique().rename("airline_codes_per_dot")
    dots_per_code = mapping.groupby("airline_code")["dot_code"].nunique().rename("dot_codes_per_airline_code")
    mapping = mapping.merge(codes_per_dot, on="dot_code").merge(dots_per_code, on="airline_code")
    mapping["mapping_status"] = np.select(
        [
            mapping["airline_codes_per_dot"].eq(1) & mapping["dot_codes_per_airline_code"].eq(1),
            mapping["airline_codes_per_dot"].gt(1),
            mapping["dot_codes_per_airline_code"].gt(1),
        ],
        ["one_to_one_observed", "multiple_airline_codes_for_dot", "multiple_dot_codes_for_airline_code"],
        default="requires_review",
    )

    route_counts = combine(route_parts, ["regime", "route"])
    regime_totals = combine(regime_parts, ["regime"])
    # Explicit suffixes avoid ambiguous scheduled_flights_x/y column names after
    # merging route-level counts with each regime's total scheduled-flight count.
    top_routes = route_counts.merge(
        regime_totals,
        on="regime",
        how="left",
        suffixes=("_route", "_regime"),
    )
    top_routes = top_routes.rename(
        columns={
            "scheduled_flights_route": "route_flights",
            "scheduled_flights_regime": "regime_flights",
        }
    )
    top_routes["route_share_pct"] = safe_percent(
        top_routes["route_flights"],
        top_routes["regime_flights"],
    )
    top_routes = top_routes.sort_values(
        ["regime", "route_flights"],
        ascending=[True, False],
    )
    top_routes["rank_within_regime"] = (
        top_routes.groupby("regime").cumcount() + 1
    )
    top_routes = top_routes[
        top_routes["rank_within_regime"] <= 20
    ].copy()

    airport_coverage = combine(airport_parts, ["regime", "role", "airport"])
    airport_summary = (
        airport_coverage.groupby(["regime", "role"], as_index=False)
        .agg(unique_airports=("airport", "nunique"), flight_appearances=("flight_appearances", "sum"))
    )

    if tail_column:
        tail_audit = pd.DataFrame([{
            "tail_identifier_column": tail_column,
            "source_field_available": True,
            "total_rows": tail_total_rows,
            "non_missing_tail_rows": tail_non_missing_rows,
            "missing_tail_rows": tail_total_rows - tail_non_missing_rows,
            "tail_completeness_pct": tail_non_missing_rows * 100.0 / tail_total_rows,
            "unique_non_missing_tail_identifiers": len(tail_unique_values),
            "rotation_analysis_status": "requires_later_timestamp_and_linkage_eligibility_audit",
        }])
    else:
        tail_audit = pd.DataFrame([{
            "tail_identifier_column": pd.NA,
            "source_field_available": False,
            "total_rows": 3_000_000,
            "non_missing_tail_rows": 0,
            "missing_tail_rows": 3_000_000,
            "tail_completeness_pct": 0.0,
            "unique_non_missing_tail_identifiers": 0,
            "rotation_analysis_status": "not_supported_by_this_source_file_no_tail_identifier_field",
        }])

    mapping.to_csv(CARRIER_MAPPING_PATH, index=False)
    carrier_by_regime.to_csv(CARRIER_REGIME_PATH, index=False)
    airport_coverage.to_csv(AIRPORT_REGIME_PATH, index=False)
    top_routes.to_csv(TOP_ROUTE_PATH, index=False)
    tail_audit.to_csv(TAIL_AUDIT_PATH, index=False)

    one_to_one = int((mapping["mapping_status"] == "one_to_one_observed").sum())
    mapping_issues = int((mapping["mapping_status"] != "one_to_one_observed").sum())
    top_route_summary = top_routes[top_routes["rank_within_regime"].eq(1)].copy()

    policy_lines = [
        "# Carrier Identifier Policy",
        "",
        "## Audited source fields",
        "",
        "The raw flight source contains `AIRLINE_CODE` and `DOT_CODE`. This audit records the observed mapping between both fields over the available period rather than assuming carrier-code stability from a dataset description.",
        "",
        "## Policy",
        "",
        "- Use `DOT_CODE` as the primary longitudinal carrier identifier when a stable one-to-one relationship is confirmed by the mapping audit.",
        "- Retain `AIRLINE_CODE` for readable reporting and for detecting code-to-identifier mapping changes.",
        "- Do not merge or rename carriers based only on assumptions about airline brands, mergers, or codes.",
        f"- Observed carrier-code/DOT pairs: {len(mapping):,}.",
        f"- One-to-one observed pairs: {one_to_one:,}.",
        f"- Pairs requiring review: {mapping_issues:,}.",
        "",
        "The detailed evidence is stored in `reports/tables/carrier_identifier_mapping.csv`.",
        "",
    ]
    POLICY_PATH.write_text("\n".join(policy_lines), encoding="utf-8")

    report_lines = [
        "# Entity Coverage and Rotation Feasibility Audit",
        "",
        "## Purpose",
        "",
        "This Phase 2 audit assesses the source support for longitudinal carrier analysis, airport-route network analysis, and later aircraft-rotation reconstruction. It is descriptive and does not infer aircraft rotations.",
        "",
        "## Carrier identifier audit",
        "",
        f"- Observed airline-code/DOT-code pairs: {format_int(len(mapping))}.",
        f"- One-to-one observed pairs: {format_int(one_to_one)}.",
        f"- Pairs requiring mapping review: {format_int(mapping_issues)}.",
        "- Longitudinal carrier analysis must use the documented identifier policy, not carrier-code assumptions.",
        "",
        "## Tail identifier audit",
        "",
        f"- Source tail identifier field: `{tail_column}`." if tail_column else "- No recognised tail identifier field (`TAIL_NUMBER`, `TAIL_NUM`, or `TAILNUMBER`) exists in this raw source file.",
        f"- Rotation analysis status: `{tail_audit.iloc[0]['rotation_analysis_status']}`.",
        "- A tail field, if present, is necessary but not sufficient for propagation claims; UTC ordering, airport continuity, turnaround plausibility, and prior-leg availability still require later eligibility checks.",
        "",
        "## Airport coverage by regime",
        "",
        "| Regime | Role | Unique airports | Flight appearances |",
        "|---|---|---:|---:|",
    ]
    for _, row in airport_summary.sort_values(["regime", "role"]).iterrows():
        report_lines.append(
            f"| {row['regime']} | {row['role']} | {format_int(row['unique_airports'])} | {format_int(row['flight_appearances'])} |"
        )

    report_lines.extend([
        "",
        "## Highest-volume route by regime",
        "",
        "| Regime | Route | Route flights | Share of regime flights |",
        "|---|---|---:|---:|",
    ])
    for _, row in top_route_summary.sort_values("regime").iterrows():
        report_lines.append(
            f"| {row['regime']} | {row['route']} | {format_int(row['route_flights'])} | {row['route_share_pct']:.2f}% |"
        )

    report_lines.extend([
        "",
        "## Generated evidence",
        "",
        f"- Carrier mapping: `{CARRIER_MAPPING_PATH.as_posix()}`",
        f"- Carrier coverage: `{CARRIER_REGIME_PATH.as_posix()}`",
        f"- Airport coverage: `{AIRPORT_REGIME_PATH.as_posix()}`",
        f"- Top routes: `{TOP_ROUTE_PATH.as_posix()}`",
        f"- Tail availability: `{TAIL_AUDIT_PATH.as_posix()}`",
        "",
    ])
    COVERAGE_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nEntity coverage audit complete.")
    print(f"- Report: {COVERAGE_REPORT_PATH}")
    print(f"- Carrier policy: {POLICY_PATH}")
    print(f"- Tail audit: {TAIL_AUDIT_PATH}")


if __name__ == "__main__":
    main()
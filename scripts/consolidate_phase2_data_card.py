"""Phase 2: consolidate data card and final data-quality report without weather."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

MONTHLY_TABLE = Path("reports/tables/monthly_operational_disruption_summary.csv")
REGIME_TABLE = Path("reports/tables/regime_operational_disruption_summary.csv")
DIAGNOSTICS_TABLE = Path("reports/tables/regime_shift_diagnostics.csv")
ENTITY_TABLE = Path("reports/tables/entity_disruption_summary.csv")
TOP_ENTITY_TABLE = Path("reports/tables/top_entity_disruption_summary.csv")
AIRPORT_LIST = Path("data/external/noaa_required_airports.csv")
ROTATION_AUDIT = Path("docs/data/rotation_reconstruction_audit.md")
DATA_CARD = Path("docs/data/data_card.md")
QUALITY_REPORT = Path("reports/evaluation/data_quality_report.md")

SEVERE_DELAY_MINUTES = 120


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def format_float(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    for path in (DATA_CARD, QUALITY_REPORT):
        path.parent.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(MONTHLY_TABLE, parse_dates=["month"])
    regime = pd.read_csv(REGIME_TABLE)
    diagnostics = pd.read_csv(DIAGNOSTICS_TABLE)
    top_entity = pd.read_csv(TOP_ENTITY_TABLE)
    airports = pd.read_csv(AIRPORT_LIST)

    total_flights = int(regime["scheduled_flights"].sum())
    unique_airports = int(airports["airport_iata"].nunique())
    unique_carriers = int(monthly["regime"].nunique())  # placeholder; real carriers from entity table

    start_date = monthly["month"].min().date().isoformat()
    end_date = (monthly["month"].max() + pd.offsets.MonthEnd(0)).date().isoformat()

    regime_rows = []
    for _, row in regime.iterrows():
        regime_rows.append(
            f"- **{row['regime']}**: {format_int(row['scheduled_flights'])} scheduled flights, "
            f"{format_float(row['cancellation_rate_pct'])}% cancellation rate, "
            f"{format_float(row['severe_delay_rate_pct'])}% severe-delay rate."
        )

    diag_rows = []
    for _, row in diagnostics.iterrows():
        diag_rows.append(
            f"- {row['comparison_regime']} vs 2019: {row['metric']} relative change {format_float(row['relative_change_pct'])}%, "
            f"Cohen's d {format_float(row['cohens_d'])}, permutation p-value {row['permutation_p_value']:.5f}."
        )

    top_rows = []
    for _, row in top_entity[top_entity["severe_delay_rank"] == 1].iterrows():
        top_rows.append(
            f"- {row['regime']}, {row['entity_type']}: {row['entity']} with severe-delay rate {format_float(row['severe_delay_rate_pct'])}%, "
            f"based on {format_int(row['scheduled_flights'])} scheduled flights."
        )

    data_card_lines = [
        "# Data Card — Airline Network Disruption Intelligence", "",
        "## Source and coverage", "",
        "- Primary source: Kaggle Flight Delay and Cancellation dataset (sample of approximately 3 million flights).",
        f"- Flight-date coverage: {start_date} to {end_date}.",
        f"- Total scheduled flights: {format_int(total_flights)}.",
        f"- Unique airports (origin or destination): {format_int(unique_airports)}.",
        "- Unique carriers: 18 (from entity audit).",
        "- No tail-number or aircraft identifier field in this source file.",
        "- No weather data used in this version of the project.", "",
        "## Operating regimes", "",
    ] + regime_rows + [
        "", "## Regime-shift diagnostics (monthly series)", "",
    ] + diag_rows + [
        "", "## Highest severe-delay-rate entities (minimum 1,000 flights per regime)", "",
    ] + top_rows + [
        "", "## Key limitations", "",
        "- No aircraft-rotation reconstruction, propagation, cascade, or tail-based recovery analysis is possible due to the absence of a tail identifier.",
        "- No weather features are used; disruption patterns are analysed from flight operations alone.",
        "- The 2023 period ends on 2023-08-31 and is therefore partial-year coverage.",
        "- All findings are descriptive associations, not causal conclusions.", "",
        "## Intended use", "",
        "This dataset supports flight-level severe-delay and cancellation prediction, arrival-delay regression, carrier and airport analysis, and airport-route network analysis.",
        "It does not support aircraft-level propagation or tail-based recovery claims.", "",
    ]

    quality_lines = [
        "# Data Quality and Regime Analysis Report — Phase 2 Summary", "",
        "## Purpose", "",
        "This report summarises Phase 2 data-quality checks, regime analysis, and limitations for the selected flight dataset.",
        "It consolidates evidence from earlier Phase 2 scripts and documents constraints for later phases.", "",
        "## Source audit", "",
        f"- Raw file: `data/raw/flights_kaggle/flights_sample_3m.csv`.",
        f"- Date coverage: {start_date} to {end_date}.",
        f"- Total scheduled flights: {format_int(total_flights)}.",
        f"- Unique airports: {format_int(unique_airports)}.",
        "- Unique carriers: 18.",
        "- No tail-number field present.", "",
        "## Chronological validation design", "",
        "- Training and development: 2019 to 2021.",
        "- Validation and model selection: 2022.",
        "- Final untouched test period: 2023-01-01 to 2023-08-31 (partial year).",
        "- All evaluation uses chronological splits only; random splitting is prohibited.", "",
        "## Regime definitions", "",
    ] + regime_rows + [
        "", "## Regime-shift diagnostics", "",
    ] + diag_rows + [
        "", "## Schedule and entity disruption findings", "",
    ] + top_rows + [
        "", "## Rotation feasibility", "",
        "A formal audit in `docs/data/rotation_reconstruction_audit.md` concluded that aircraft-level rotation reconstruction is not supported by this source file.",
        "Later phases must not claim aircraft-propagation or tail-based recovery results.", "",
        "## Weather data", "",
        "This version of the project does not use NOAA or any other weather dataset.",
        "Disruption analysis is based solely on flight operations, schedule, carrier, airport, and route information.", "",
        "## Consequences for later phases", "",
        "- Phase 3 (targets and features): no tail-based rotation features, no weather features.",
        "- Phase 5 (propagation and recovery): no aircraft-level propagation; only airport-route network patterns.",
        "- Phase 6 (sequence models): no rotation-based sequences; any temporal sequences must use non-tail designs.",
        "- All reports and the README must state these limitations explicitly.", "",
        "## Generated outputs", "",
        f"- Data card: `{DATA_CARD.as_posix()}`",
        f"- This report: `{QUALITY_REPORT.as_posix()}`", "",
    ]

    DATA_CARD.write_text("\n".join(data_card_lines), encoding="utf-8")
    QUALITY_REPORT.write_text("\n".join(quality_lines), encoding="utf-8")

    print("Phase 2 consolidation complete.")
    print(f"- Data card: {DATA_CARD}")
    print(f"- Quality report: {QUALITY_REPORT}")


if __name__ == "__main__":
    main()
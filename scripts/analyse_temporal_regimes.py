"""
Phase 2: Temporal EDA and operating-regime analysis.

This script creates reproducible monthly and regime-level summaries from the
raw Kaggle flight CSV without modifying the immutable source data.

It reports:
- Scheduled flight volume.
- Cancellation rate.
- Diversion rate.
- Completed non-diverted flight cohort size.
- Mean arrival delay.
- Severe arrival-delay prevalence, using the project-defined threshold:
  ARR_DELAY >= 120 minutes.
- Arrival-delay distribution figures using a labelled analysis sample.

Operating regimes:
- 2019: pre-COVID reference.
- 2020: COVID operational shock.
- 2021 to 2023-08-31: recovery and transition.

Important cohort definitions:
- Scheduled flights: every source row.
- Cancelled flights: CANCELLED == 1.
- Diverted flights: DIVERTED == 1.
- Completed arrival-delay cohort:
  CANCELLED == 0, DIVERTED == 0, and ARR_DELAY is present.
- Severe delay:
  ARR_DELAY >= 120 minutes within the completed arrival-delay cohort.

This is descriptive retrospective EDA. It does not build predictive features
and does not make causal claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_FLIGHT_CSV: Final = Path("data/raw/flights_kaggle/flights_sample_3m.csv")

MONTHLY_OUTPUT_PATH: Final = Path(
    "reports/tables/monthly_operational_disruption_summary.csv"
)
REGIME_OUTPUT_PATH: Final = Path(
    "reports/tables/regime_operational_disruption_summary.csv"
)
REPORT_OUTPUT_PATH: Final = Path(
    "reports/evaluation/regime_shift_report.md"
)

VOLUME_FIGURE_PATH: Final = Path(
    "reports/figures/monthly_flight_volume_by_regime.png"
)
DISRUPTION_FIGURE_PATH: Final = Path(
    "reports/figures/monthly_cancellation_diversion_severe_delay_rates.png"
)
DELAY_DISTRIBUTION_FIGURE_PATH: Final = Path(
    "reports/figures/arrival_delay_distribution_by_regime.png"
)

CHUNK_SIZE: Final = 250_000

# The full raw data is processed for all tables and rates.
# A capped sample is used only for the distribution plot to keep memory use
# controlled on a local computer.
MAX_DELAY_SAMPLE_PER_CHUNK: Final = 5_000

SEVERE_DELAY_THRESHOLD_MINUTES: Final = 120

REQUIRED_COLUMNS: Final = [
    "FL_DATE",
    "CANCELLED",
    "DIVERTED",
    "ARR_DELAY",
]

REGIME_ORDER: Final = [
    "2019 pre-COVID reference",
    "2020 COVID operational shock",
    "2021-2023 recovery and transition",
]


# =============================================================================
# HELPERS
# =============================================================================


def ensure_output_directories() -> None:
    """Create output folders if they do not already exist."""
    for path in [
        MONTHLY_OUTPUT_PATH,
        REGIME_OUTPUT_PATH,
        REPORT_OUTPUT_PATH,
        VOLUME_FIGURE_PATH,
        DISRUPTION_FIGURE_PATH,
        DELAY_DISTRIBUTION_FIGURE_PATH,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)


def assign_regime(flight_dates: pd.Series) -> pd.Series:
    """
    Assign each validated flight date to one project-defined operating regime.

    The audited source coverage ends on 2023-08-31. Any date outside the
    confirmed coverage is marked as 'outside_audited_coverage' so it cannot
    silently enter the reported regime summaries.
    """
    conditions = [
        (flight_dates >= pd.Timestamp("2019-01-01"))
        & (flight_dates <= pd.Timestamp("2019-12-31")),
        (flight_dates >= pd.Timestamp("2020-01-01"))
        & (flight_dates <= pd.Timestamp("2020-12-31")),
        (flight_dates >= pd.Timestamp("2021-01-01"))
        & (flight_dates <= pd.Timestamp("2023-08-31")),
    ]

    choices = REGIME_ORDER

    regime = np.select(
        conditions,
        choices,
        default="outside_audited_coverage",
    )

    return pd.Series(regime, index=flight_dates.index, dtype="string")


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a percentage rate while avoiding division-by-zero errors."""
    return np.where(
        denominator > 0,
        (numerator / denominator) * 100.0,
        np.nan,
    )


def format_count(value: int | float) -> str:
    """Format integer-like values for a readable Markdown report."""
    return f"{int(value):,}"


def format_percentage(value: float) -> str:
    """Format percentage values for a readable Markdown report."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def format_minutes(value: float) -> str:
    """Format delay values for a readable Markdown report."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f} minutes"


# =============================================================================
# CHUNK PROCESSING
# =============================================================================


def process_flight_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform one raw-data chunk into:
    1. Exact monthly additive metrics.
    2. A bounded completed-flight delay sample for distribution plotting.

    All returned counts, sums, and sum-of-squares values are additive across
    chunks, so final monthly/regime estimates use the full 3-million-row CSV.
    """
    flight_dates = pd.to_datetime(
        chunk["FL_DATE"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    invalid_date_count = int(flight_dates.isna().sum())
    if invalid_date_count > 0:
        raise ValueError(
            f"Found {invalid_date_count:,} invalid FL_DATE values in a chunk. "
            "The audit confirmed YYYY-MM-DD with no invalid dates, so stop "
            "instead of silently excluding unexpected rows."
        )

    cancelled = pd.to_numeric(chunk["CANCELLED"], errors="coerce")
    diverted = pd.to_numeric(chunk["DIVERTED"], errors="coerce")
    arrival_delay = pd.to_numeric(chunk["ARR_DELAY"], errors="coerce")

    invalid_cancelled = ~cancelled.isin([0, 1])
    invalid_diverted = ~diverted.isin([0, 1])

    if invalid_cancelled.any():
        invalid_values = chunk.loc[invalid_cancelled, "CANCELLED"].dropna().unique()
        raise ValueError(
            "CANCELLED contains values other than 0 or 1: "
            f"{invalid_values[:10].tolist()}"
        )

    if invalid_diverted.any():
        invalid_values = chunk.loc[invalid_diverted, "DIVERTED"].dropna().unique()
        raise ValueError(
            "DIVERTED contains values other than 0 or 1: "
            f"{invalid_values[:10].tolist()}"
        )

    month = flight_dates.dt.to_period("M").dt.to_timestamp()

    completed_arrival_cohort = (
        cancelled.eq(0)
        & diverted.eq(0)
        & arrival_delay.notna()
    )

    severe_delay = (
        completed_arrival_cohort
        & arrival_delay.ge(SEVERE_DELAY_THRESHOLD_MINUTES)
    )

    # Retain arrival delay only where it belongs to the defined completed cohort.
    completed_arrival_delay = arrival_delay.where(completed_arrival_cohort)

    metrics = pd.DataFrame(
        {
            "month": month,
            "scheduled_flights": 1,
            "cancelled_flights": cancelled.eq(1).astype("int64"),
            "diverted_flights": diverted.eq(1).astype("int64"),
            "completed_arrival_delay_flights": completed_arrival_cohort.astype("int64"),
            "severe_delay_flights": severe_delay.astype("int64"),
            "arrival_delay_sum_minutes": completed_arrival_delay.fillna(0.0),
            "arrival_delay_sum_squares": completed_arrival_delay.pow(2).fillna(0.0),
        }
    )

    monthly_metrics = (
        metrics.groupby("month", as_index=False)
        .sum(numeric_only=True)
        .sort_values("month")
        .reset_index(drop=True)
    )

    delay_sample = pd.DataFrame(
        {
            "arrival_delay_minutes": completed_arrival_delay.loc[
                completed_arrival_cohort
            ],
            "regime": assign_regime(flight_dates.loc[completed_arrival_cohort]),
        }
    )

    if len(delay_sample) > MAX_DELAY_SAMPLE_PER_CHUNK:
        delay_sample = delay_sample.sample(
            n=MAX_DELAY_SAMPLE_PER_CHUNK,
            random_state=42,
        )

    return monthly_metrics, delay_sample


def build_monthly_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the immutable raw CSV in chunks and produce full-data monthly metrics.

    The returned delay sample is used only for plotting the distribution.
    All summary tables remain exact because they are based on additive
    aggregations from every source row.
    """
    if not RAW_FLIGHT_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {RAW_FLIGHT_CSV}\n"
            "Check RAW_FLIGHT_CSV at the top of this script."
        )

    monthly_parts: list[pd.DataFrame] = []
    delay_sample_parts: list[pd.DataFrame] = []

    print(f"Reading raw data in chunks of {CHUNK_SIZE:,} rows...")

    reader = pd.read_csv(
        RAW_FLIGHT_CSV,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        monthly_chunk, delay_sample_chunk = process_flight_chunk(chunk)

        monthly_parts.append(monthly_chunk)
        delay_sample_parts.append(delay_sample_chunk)

        print(
            f"Processed chunk {chunk_number:,}: "
            f"{len(chunk):,} source rows."
        )

    monthly_summary = (
        pd.concat(monthly_parts, ignore_index=True)
        .groupby("month", as_index=False)
        .sum(numeric_only=True)
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly_summary["regime"] = assign_regime(monthly_summary["month"])

    monthly_summary["cancellation_rate_pct"] = safe_rate(
        monthly_summary["cancelled_flights"],
        monthly_summary["scheduled_flights"],
    )

    monthly_summary["diversion_rate_pct"] = safe_rate(
        monthly_summary["diverted_flights"],
        monthly_summary["scheduled_flights"],
    )

    monthly_summary["severe_delay_rate_pct"] = safe_rate(
        monthly_summary["severe_delay_flights"],
        monthly_summary["completed_arrival_delay_flights"],
    )

    monthly_summary["mean_arrival_delay_minutes"] = np.where(
        monthly_summary["completed_arrival_delay_flights"] > 0,
        monthly_summary["arrival_delay_sum_minutes"]
        / monthly_summary["completed_arrival_delay_flights"],
        np.nan,
    )

    delay_sample = pd.concat(delay_sample_parts, ignore_index=True)

    return monthly_summary, delay_sample


def build_regime_summary(monthly_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exact monthly values into exact project-defined regime summaries.
    """
    regime_summary = (
        monthly_summary.groupby("regime", as_index=False)
        .agg(
            period_start=("month", "min"),
            period_end=("month", "max"),
            scheduled_flights=("scheduled_flights", "sum"),
            cancelled_flights=("cancelled_flights", "sum"),
            diverted_flights=("diverted_flights", "sum"),
            completed_arrival_delay_flights=(
                "completed_arrival_delay_flights",
                "sum",
            ),
            severe_delay_flights=("severe_delay_flights", "sum"),
            arrival_delay_sum_minutes=("arrival_delay_sum_minutes", "sum"),
            arrival_delay_sum_squares=("arrival_delay_sum_squares", "sum"),
        )
    )

    regime_summary = regime_summary[
        regime_summary["regime"].isin(REGIME_ORDER)
    ].copy()

    regime_summary["regime"] = pd.Categorical(
        regime_summary["regime"],
        categories=REGIME_ORDER,
        ordered=True,
    )
    regime_summary = regime_summary.sort_values("regime").reset_index(drop=True)

    regime_summary["cancellation_rate_pct"] = safe_rate(
        regime_summary["cancelled_flights"],
        regime_summary["scheduled_flights"],
    )

    regime_summary["diversion_rate_pct"] = safe_rate(
        regime_summary["diverted_flights"],
        regime_summary["scheduled_flights"],
    )

    regime_summary["severe_delay_rate_pct"] = safe_rate(
        regime_summary["severe_delay_flights"],
        regime_summary["completed_arrival_delay_flights"],
    )

    regime_summary["mean_arrival_delay_minutes"] = np.where(
        regime_summary["completed_arrival_delay_flights"] > 0,
        regime_summary["arrival_delay_sum_minutes"]
        / regime_summary["completed_arrival_delay_flights"],
        np.nan,
    )

    return regime_summary


# =============================================================================
# FIGURES
# =============================================================================


def create_figures(
    monthly_summary: pd.DataFrame,
    delay_sample: pd.DataFrame,
) -> None:
    """Create curated Phase 2 temporal EDA figures."""
    sns.set_theme(style="whitegrid", context="talk")

    plot_data = monthly_summary.copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    for regime, regime_data in plot_data.groupby("regime", observed=False):
        if regime not in REGIME_ORDER:
            continue

        ax.plot(
            regime_data["month"],
            regime_data["scheduled_flights"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=regime,
        )

    ax.set_title("Monthly Scheduled Flight Volume by Operating Regime")
    ax.set_xlabel("Month")
    ax.set_ylabel("Scheduled flights")
    ax.legend(title="Regime", fontsize=10)
    ax.ticklabel_format(axis="y", style="plain")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(VOLUME_FIGURE_PATH, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        plot_data["month"],
        plot_data["cancellation_rate_pct"],
        label="Cancellation rate",
        linewidth=2,
    )
    ax.plot(
        plot_data["month"],
        plot_data["diversion_rate_pct"],
        label="Diversion rate",
        linewidth=2,
    )
    ax.plot(
        plot_data["month"],
        plot_data["severe_delay_rate_pct"],
        label=f"Severe arrival-delay rate (≥ {SEVERE_DELAY_THRESHOLD_MINUTES} min)",
        linewidth=2,
    )

    ax.set_title("Monthly Disruption Rates")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DISRUPTION_FIGURE_PATH, dpi=160)
    plt.close(fig)

    plot_sample = delay_sample[
        delay_sample["regime"].isin(REGIME_ORDER)
        & delay_sample["arrival_delay_minutes"].between(-120, 600)
    ].copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    sns.histplot(
        data=plot_sample,
        x="arrival_delay_minutes",
        hue="regime",
        hue_order=REGIME_ORDER,
        bins=120,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=1.5,
        ax=ax,
    )

    ax.axvline(
        SEVERE_DELAY_THRESHOLD_MINUTES,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Severe-delay threshold: {SEVERE_DELAY_THRESHOLD_MINUTES} min",
    )
    ax.set_title(
        "Arrival-Delay Distribution by Operating Regime\n"
        "Completed, Non-Diverted Flights Only"
    )
    ax.set_xlabel("Arrival delay (minutes)")
    ax.set_ylabel("Estimated density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(DELAY_DISTRIBUTION_FIGURE_PATH, dpi=160)
    plt.close(fig)


# =============================================================================
# REPORTING
# =============================================================================


def write_regime_report(
    monthly_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    delay_sample: pd.DataFrame,
) -> None:
    """Write an evidence-led Markdown report from the computed summaries."""
    source_start = monthly_summary["month"].min().date().isoformat()
    source_end = (
        monthly_summary["month"].max() + pd.offsets.MonthEnd(0)
    ).date().isoformat()

    report_lines = [
        "# Regime Shift Analysis",
        "",
        "## Scope",
        "",
        "This report is a descriptive Phase 2 analysis of the audited raw "
        "Kaggle flight CSV. It does not provide causal claims or predictive "
        "model results.",
        "",
        f"- Source coverage analysed: {source_start} to {source_end}.",
        "- The 2023 period ends on 2023-08-31 and is therefore partial-year coverage.",
        f"- Severe delay is a project-defined threshold: arrival delay ≥ "
        f"{SEVERE_DELAY_THRESHOLD_MINUTES} minutes.",
        "- Arrival-delay statistics use completed, non-cancelled, "
        "non-diverted flights with a recorded arrival delay.",
        "- Cancellation and diversion rates use all scheduled source rows.",
        "",
        "## Regime definitions",
        "",
        "| Regime | Period |",
        "|---|---|",
        "| 2019 pre-COVID reference | 2019-01-01 to 2019-12-31 |",
        "| 2020 COVID operational shock | 2020-01-01 to 2020-12-31 |",
        "| 2021-2023 recovery and transition | 2021-01-01 to 2023-08-31 |",
        "",
        "## Exact regime-level summary",
        "",
        "| Regime | Period | Scheduled flights | Cancelled flights | Cancellation rate | Diverted flights | Diversion rate | Completed arrival-delay cohort | Mean arrival delay | Severe delays | Severe-delay rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in regime_summary.iterrows():
        period = (
            f"{row['period_start'].date().isoformat()} to "
            f"{(row['period_end'] + pd.offsets.MonthEnd(0)).date().isoformat()}"
        )

        report_lines.append(
            "| "
            f"{row['regime']} | "
            f"{period} | "
            f"{format_count(row['scheduled_flights'])} | "
            f"{format_count(row['cancelled_flights'])} | "
            f"{format_percentage(row['cancellation_rate_pct'])} | "
            f"{format_count(row['diverted_flights'])} | "
            f"{format_percentage(row['diversion_rate_pct'])} | "
            f"{format_count(row['completed_arrival_delay_flights'])} | "
            f"{format_minutes(row['mean_arrival_delay_minutes'])} | "
            f"{format_count(row['severe_delay_flights'])} | "
            f"{format_percentage(row['severe_delay_rate_pct'])} |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Differences between regimes describe observed changes in this source dataset.",
            "- The 2020 regime is retained as a distribution-shift period; it is not discarded as an outlier.",
            "- The recovery/transition regime has more months than the 2019 and 2020 regimes, so compare rates and monthly patterns rather than raw counts alone.",
            "- The arrival-delay distribution figure is based on a bounded analysis sample for plotting only. The monthly and regime tables use all available source rows.",
            "- Cancellation reasons and delay-cause fields are not used in this report to explain or predict disruption outcomes.",
            "",
            "## Generated outputs",
            "",
            f"- Monthly table: `{MONTHLY_OUTPUT_PATH.as_posix()}`",
            f"- Regime table: `{REGIME_OUTPUT_PATH.as_posix()}`",
            f"- Flight-volume figure: `{VOLUME_FIGURE_PATH.as_posix()}`",
            f"- Disruption-rate figure: `{DISRUPTION_FIGURE_PATH.as_posix()}`",
            f"- Arrival-delay distribution figure: `{DELAY_DISTRIBUTION_FIGURE_PATH.as_posix()}`",
            f"- Distribution plotting sample rows: {format_count(len(delay_sample))}",
            "",
        ]
    )

    REPORT_OUTPUT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the complete Phase 2 temporal EDA and regime analysis."""
    ensure_output_directories()

    print("Starting Phase 2 temporal EDA and regime analysis...")
    monthly_summary, delay_sample = build_monthly_summary()

    regime_summary = build_regime_summary(monthly_summary)

    monthly_summary.to_csv(MONTHLY_OUTPUT_PATH, index=False)
    regime_summary.to_csv(REGIME_OUTPUT_PATH, index=False)

    print("Creating figures...")
    create_figures(monthly_summary, delay_sample)

    print("Writing Markdown report...")
    write_regime_report(
        monthly_summary=monthly_summary,
        regime_summary=regime_summary,
        delay_sample=delay_sample,
    )

    print("\nPhase 2 temporal EDA complete.")
    print(f"- Monthly summary: {MONTHLY_OUTPUT_PATH}")
    print(f"- Regime summary: {REGIME_OUTPUT_PATH}")
    print(f"- Report: {REPORT_OUTPUT_PATH}")
    print(f"- Figure: {VOLUME_FIGURE_PATH}")
    print(f"- Figure: {DISRUPTION_FIGURE_PATH}")
    print(f"- Figure: {DELAY_DISTRIBUTION_FIGURE_PATH}")


if __name__ == "__main__":
    main()
"""Phase 2: disruption variation by scheduled hour, carrier, origin airport, and route."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_CSV = Path("data/raw/flights_kaggle/flights_sample_3m.csv")
CHUNK_SIZE = 250_000
SEVERE_DELAY_MINUTES = 120
MIN_ENTITY_FLIGHTS = 1_000

HOUR_TABLE = Path("reports/tables/scheduled_departure_hour_disruption_summary.csv")
ENTITY_TABLE = Path("reports/tables/entity_disruption_summary.csv")
TOP_ENTITY_TABLE = Path("reports/tables/top_entity_disruption_summary.csv")
HOUR_FIGURE = Path("reports/figures/scheduled_departure_hour_disruption_rates.png")
REPORT = Path("reports/evaluation/schedule_entity_disruption_report.md")

USECOLS = ["FL_DATE", "CRS_DEP_TIME", "AIRLINE_CODE", "ORIGIN", "DEST", "CANCELLED", "DIVERTED", "ARR_DELAY"]
REGIMES = ["2019 pre-COVID reference", "2020 COVID operational shock", "2021-2023 recovery and transition"]


def regime(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([
        dates.between("2019-01-01", "2019-12-31"),
        dates.between("2020-01-01", "2020-12-31"),
        dates.between("2021-01-01", "2023-08-31"),
    ], REGIMES, default="outside_audited_coverage"), index=dates.index)


def departure_hour(values: pd.Series) -> pd.Series:
    """Parse HHMM scheduled departure times; 2400 is next-day midnight (hour 0)."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & numeric.ge(0) & numeric.le(2400) & ((numeric % 100) < 60)
    hour = (numeric // 100).where(valid)
    return hour.mask(hour.eq(24), 0).astype("Int64")


def combine(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True).groupby(keys, as_index=False).sum(numeric_only=True)


def rates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["cancellation_rate_pct"] = np.where(frame["scheduled_flights"] > 0, 100 * frame["cancelled_flights"] / frame["scheduled_flights"], np.nan)
    frame["diversion_rate_pct"] = np.where(frame["scheduled_flights"] > 0, 100 * frame["diverted_flights"] / frame["scheduled_flights"], np.nan)
    frame["severe_delay_rate_pct"] = np.where(frame["completed_arrival_flights"] > 0, 100 * frame["severe_delay_flights"] / frame["completed_arrival_flights"], np.nan)
    frame["mean_arrival_delay_minutes"] = np.where(frame["completed_arrival_flights"] > 0, frame["arrival_delay_sum"] / frame["completed_arrival_flights"], np.nan)
    return frame


def main() -> None:
    for path in (HOUR_TABLE, ENTITY_TABLE, TOP_ENTITY_TABLE, HOUR_FIGURE, REPORT):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    hour_parts, entity_parts = [], []
    print("Starting Phase 2 schedule and entity disruption analysis...")

    for number, chunk in enumerate(pd.read_csv(RAW_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        dates = pd.to_datetime(chunk["FL_DATE"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise ValueError("Unexpected invalid FL_DATE after completed date audit.")
        cancelled = pd.to_numeric(chunk["CANCELLED"], errors="coerce")
        diverted = pd.to_numeric(chunk["DIVERTED"], errors="coerce")
        arrival_delay = pd.to_numeric(chunk["ARR_DELAY"], errors="coerce")
        if not cancelled.isin([0, 1]).all() or not diverted.isin([0, 1]).all():
            raise ValueError("CANCELLED or DIVERTED contains a value other than 0 or 1.")

        frame = pd.DataFrame({
            "regime": regime(dates),
            "scheduled_hour_local": departure_hour(chunk["CRS_DEP_TIME"]),
            "carrier": chunk["AIRLINE_CODE"].astype("string").str.strip(),
            "origin": chunk["ORIGIN"].astype("string").str.strip(),
            "destination": chunk["DEST"].astype("string").str.strip(),
            "cancelled_flights": cancelled.eq(1).astype("int64"),
            "diverted_flights": diverted.eq(1).astype("int64"),
        })
        frame = frame[frame["regime"].isin(REGIMES)].copy()
        completed = cancelled.eq(0) & diverted.eq(0) & arrival_delay.notna()
        frame["completed_arrival_flights"] = completed.loc[frame.index].astype("int64")
        frame["severe_delay_flights"] = (completed & arrival_delay.ge(SEVERE_DELAY_MINUTES)).loc[frame.index].astype("int64")
        frame["arrival_delay_sum"] = arrival_delay.loc[frame.index].where(completed.loc[frame.index], 0.0)
        frame["scheduled_flights"] = 1
        frame["route"] = frame["origin"] + "-" + frame["destination"]

        hour_parts.append(frame.dropna(subset=["scheduled_hour_local"]).groupby(["regime", "scheduled_hour_local"], as_index=False).sum(numeric_only=True))
        for entity_type, column in (("carrier", "carrier"), ("origin_airport", "origin"), ("route", "route")):
            grouped = frame.groupby(["regime", column], as_index=False)[["scheduled_flights", "cancelled_flights", "diverted_flights", "completed_arrival_flights", "severe_delay_flights", "arrival_delay_sum"]].sum()
            grouped = grouped.rename(columns={column: "entity"})
            grouped.insert(1, "entity_type", entity_type)
            entity_parts.append(grouped)
        print(f"Processed chunk {number:,}: {len(chunk):,} rows")

    hour_summary = rates(combine(hour_parts, ["regime", "scheduled_hour_local"]))
    entity_summary = rates(combine(entity_parts, ["regime", "entity_type", "entity"]))
    top_entities = entity_summary[entity_summary["scheduled_flights"] >= MIN_ENTITY_FLIGHTS].copy()
    top_entities = top_entities.sort_values(["regime", "entity_type", "severe_delay_rate_pct"], ascending=[True, True, False])
    top_entities["severe_delay_rank"] = top_entities.groupby(["regime", "entity_type"]).cumcount() + 1
    top_entities = top_entities[top_entities["severe_delay_rank"] <= 20]

    hour_summary.to_csv(HOUR_TABLE, index=False)
    entity_summary.to_csv(ENTITY_TABLE, index=False)
    top_entities.to_csv(TOP_ENTITY_TABLE, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    for name, subset in hour_summary.groupby("regime"):
        axes[0].plot(subset["scheduled_hour_local"], subset["cancellation_rate_pct"], marker="o", label=name)
        axes[1].plot(subset["scheduled_hour_local"], subset["severe_delay_rate_pct"], marker="o", label=name)
    axes[0].set(title="Cancellation Rate by Scheduled Local Departure Hour", ylabel="Cancellation rate (%)")
    axes[1].set(title=f"Severe Arrival-Delay Rate by Scheduled Local Departure Hour (≥ {SEVERE_DELAY_MINUTES} min)", xlabel="Scheduled departure hour (local)", ylabel="Severe-delay rate (%)")
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(HOUR_FIGURE, dpi=160)
    plt.close(fig)

    lines = [
        "# Schedule and Entity Disruption Analysis", "",
        "## Scope", "",
        "This Phase 2 descriptive analysis compares cancellation, diversion, mean arrival delay, and project-defined severe arrival delay across scheduled local departure hours, carriers, origin airports, and routes.",
        "", "## Cohorts and restrictions", "",
        "- Cancellation and diversion rates use all scheduled flights.",
        "- Arrival-delay and severe-delay statistics use completed, non-cancelled, non-diverted flights with recorded arrival delay.",
        f"- Severe delay is project-defined as arrival delay ≥ {SEVERE_DELAY_MINUTES} minutes.",
        f"- Entity rankings exclude groups with fewer than {MIN_ENTITY_FLIGHTS:,} scheduled flights in a regime to reduce unstable rare-group rates.",
        "- Results are descriptive associations, not causal explanations.", "",
        "## Highest severe-delay-rate entity in each eligible group", "",
        "| Regime | Entity type | Entity | Scheduled flights | Severe-delay rate | Cancellation rate |", "|---|---|---|---:|---:|---:|",
    ]
    for _, row in top_entities[top_entities["severe_delay_rank"].eq(1)].iterrows():
        lines.append(f"| {row['regime']} | {row['entity_type']} | {row['entity']} | {int(row['scheduled_flights']):,} | {row['severe_delay_rate_pct']:.2f}% | {row['cancellation_rate_pct']:.2f}% |")
    lines.extend(["", "## Generated outputs", "", f"- Hour summary: `{HOUR_TABLE.as_posix()}`", f"- Full entity summary: `{ENTITY_TABLE.as_posix()}`", f"- Top entity table: `{TOP_ENTITY_TABLE.as_posix()}`", f"- Figure: `{HOUR_FIGURE.as_posix()}`", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\nSchedule and entity analysis complete.")
    print(f"- Report: {REPORT}")


if __name__ == "__main__":
    main()
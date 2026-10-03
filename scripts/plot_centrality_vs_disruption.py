"""Plot descriptive airport centrality versus severe-delay rate.

Both inputs use scheduled-departure UTC years. Disruption rates use
origin-airport completed flights only. This is retrospective
association, not a pre-flight feature or causal estimate.
"""

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "reports/tables/route_disruption_by_utc_year.csv"
METRICS = (
    ROOT / "reports/tables/yearly_airport_network_metrics.csv"
)
TABLE = (
    ROOT / "reports/tables/"
    "airport_route_disruption_summary.csv"
)
FIGURE = (
    ROOT / "reports/figures/"
    "centrality_vs_disruption.png"
)

MIN_COMPLETED_FLIGHTS = 1_000
YEARS = (2019, 2020, 2021, 2022, 2023)


def main() -> None:
    for path in (ROUTES, METRICS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    routes = pd.read_csv(ROUTES)
    metrics = pd.read_csv(METRICS)

    counts = [
        "scheduled_flights",
        "cancelled_flights",
        "completed_arrivals",
        "severe_delay_flights",
    ]

    # Group by origin only: each flight contributes once to the rate.
    airport = (
        routes.groupby(
            ["utc_year", "origin_airport"],
            as_index=False,
        )[counts]
        .sum()
        .rename(columns={"origin_airport": "airport"})
    )

    if airport.duplicated(["utc_year", "airport"]).any():
        raise ValueError("Duplicate origin airport-year")
    if metrics.duplicated(["utc_year", "airport"]).any():
        raise ValueError("Duplicate network metric airport-year")

    joined = airport.merge(
        metrics,
        on=["utc_year", "airport"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            "Some origin airports lack same-year network metrics"
        )
    joined = joined.drop(columns="_merge")

    # Weighted out-degree and origin scheduled-flight count must
    # describe exactly the same UTC-year flight cohort.
    if not joined["weighted_out_degree"].eq(
        joined["scheduled_flights"]
    ).all():
        raise ValueError(
            "Network and outcome airport flight counts differ"
        )

    joined["severe_delay_rate_completed"] = (
        joined["severe_delay_flights"]
        / joined["completed_arrivals"].replace(0, float("nan"))
    )
    joined["cancellation_rate"] = (
        joined["cancelled_flights"]
        / joined["scheduled_flights"]
    )
    joined["meets_plot_minimum"] = (
        joined["completed_arrivals"] >= MIN_COMPLETED_FLIGHTS
    )
    joined["analysis_scope"] = (
        "retrospective_origin_airport_association"
    )

    eligible = joined.loc[joined["meets_plot_minimum"]]
    if eligible.empty:
        raise ValueError("No airports meet the plot minimum")
    if not eligible["severe_delay_rate_completed"].between(
        0, 1
    ).all():
        raise ValueError("Invalid severe-delay rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        2019: "#2563eb",
        2020: "#dc2626",
        2021: "#16a34a",
        2022: "#a855f7",
        2023: "#ea580c",
    }

    for year in YEARS:
        group = eligible.loc[eligible["utc_year"].eq(year)]
        ax.scatter(
            group["weighted_out_degree"],
            100 * group["severe_delay_rate_completed"],
            s=24,
            alpha=0.55,
            color=colors[year],
            label=f"{year}{' (partial)' if year == 2023 else ''}",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Origin scheduled flights in UTC year (log scale)"
    )
    ax.set_ylabel(
        "Severe arrival delay among completed origin flights (%)"
    )
    ax.set_title(
        "Airport traffic and severe delay: descriptive association"
    )
    ax.grid(alpha=0.2)
    ax.legend(title="UTC departure year")
    fig.tight_layout()

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    table_temp = TABLE.with_name(TABLE.name + ".tmp")
    figure_temp = FIGURE.with_name(FIGURE.name + ".tmp")

    try:
        joined.to_csv(table_temp, index=False)
        fig.savefig(figure_temp, format="png", dpi=160)
        os.replace(table_temp, TABLE)
        os.replace(figure_temp, FIGURE)
    finally:
        table_temp.unlink(missing_ok=True)
        figure_temp.unlink(missing_ok=True)
        plt.close(fig)

    print("Airport network/disruption association: PASS")
    print(f"Origin airport-year rows: {len(joined):,}")
    print(
        "Rows meeting minimum completed flights: "
        f"{len(eligible):,}"
    )
    print(
        eligible.groupby("utc_year")
        .size()
        .rename("plotted_airports")
        .to_string()
    )
    print(f"Table: {TABLE}")
    print(f"Figure: {FIGURE}")
    print("Association only; 2023 has partial-year coverage.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
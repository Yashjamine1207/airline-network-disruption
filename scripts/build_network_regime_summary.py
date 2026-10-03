"""Compare descriptive airport networks across operating regimes.

All graph and disruption values use scheduled-departure UTC years.
The 2023 flight and route totals are partial-year and are not given
full-year percentage changes against 2019. Results are associations,
not causal effects.
"""

from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPOSURE = ROOT / "reports/tables/network_exposure_summary.csv"
METRICS = (
    ROOT / "reports/tables/yearly_airport_network_metrics.csv"
)
OUTPUT = ROOT / "reports/tables/network_regime_summary.csv"

REGIMES = {
    2019: "pre_covid_reference",
    2020: "covid_shock",
    2021: "transition_2021",
    2022: "transition_2022",
    2023: "partial_2023",
}


def main() -> None:
    for path in (EXPOSURE, METRICS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    exposure = pd.read_csv(EXPOSURE)
    metrics = pd.read_csv(METRICS)

    if (
        set(exposure["utc_year"]) != set(REGIMES)
        or exposure.duplicated("utc_year").any()
    ):
        raise ValueError("Expected one exposure row per UTC year")

    distributions = (
        metrics.groupby("utc_year")
        .agg(
            airport_metric_rows=("airport", "size"),
            median_betweenness=(
                "betweenness_centrality", "median"
            ),
            p90_betweenness=(
                "betweenness_centrality",
                lambda values: values.quantile(0.90),
            ),
            median_pagerank=("pagerank", "median"),
            median_outgoing_route_hhi=(
                "outgoing_route_concentration_hhi",
                "median",
            ),
        )
        .reset_index()
    )

    result = exposure.merge(
        distributions,
        on="utc_year",
        how="left",
        validate="one_to_one",
    ).sort_values("utc_year").reset_index(drop=True)

    if not result["node_count"].eq(
        result["airport_metric_rows"]
    ).all():
        raise ValueError("Airport metric counts do not match graphs")

    if result.loc[
        result["utc_year"].eq(2023),
        "partial_year_coverage",
    ].ne(True).any():
        raise ValueError("2023 must be labelled partial-year")

    result.insert(
        1,
        "regime",
        result["utc_year"].map(REGIMES),
    )

    reference = result.loc[
        result["utc_year"].eq(2019)
    ].iloc[0]

    # Rate differences are percentage POINTS, not relative percentages.
    result["cancellation_rate_change_vs_2019_pp"] = (
        100 * (
            result["cancellation_rate"]
            - reference["cancellation_rate"]
        )
    )
    result["severe_delay_rate_change_vs_2019_pp"] = (
        100 * (
            result["severe_delay_rate_completed"]
            - reference["severe_delay_rate_completed"]
        )
    )
    result["density_change_vs_2019"] = (
        result["density"] - reference["density"]
    )

    # Raw yearly count comparisons are only appropriate for the
    # four full-year windows; 2023 remains visible without them.
    result["flight_volume_change_vs_2019_pct"] = (
        100 * (
            result["scheduled_flights"]
            / reference["scheduled_flights"]
            - 1
        )
    )
    result["route_count_change_vs_2019_pct"] = (
        100 * (
            result["edge_count"]
            / reference["edge_count"]
            - 1
        )
    )
    result.loc[
        result["partial_year_coverage"],
        [
            "flight_volume_change_vs_2019_pct",
            "route_count_change_vs_2019_pct",
        ],
    ] = float("nan")

    result["interpretation_scope"] = (
        "descriptive_utc_year_regime_association"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    try:
        result.to_csv(temporary, index=False)
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    print("Network regime summary: PASS")
    print(result[[
        "utc_year",
        "regime",
        "scheduled_flights",
        "edge_count",
        "density",
        "cancellation_rate",
        "severe_delay_rate_completed",
        "flight_volume_change_vs_2019_pct",
    ]].to_string(index=False))
    print(f"Output: {OUTPUT}")
    print(
        "2023 is partial-year; raw volume change versus 2019 "
        "is intentionally missing."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
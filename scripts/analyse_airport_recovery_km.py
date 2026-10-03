"""Exploratory Kaplan-Meier analysis of Phase 3 airport episodes.

The input contains sampled airport-day episodes from 2019-2022 only.
Most censoring follows observation gaps; independent censoring is not
established. Results are descriptive, not operational guarantees.
"""

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "data/processed/survival_episodes/"
    "airport_recovery_episodes.csv"
)
TABLE = ROOT / "reports/tables/recovery_summary.csv"
FIGURE = ROOT / "reports/figures/kaplan_meier_recovery.png"

REQUIRED = {
    "origin_airport",
    "disruption_start_date",
    "episode_end_date",
    "duration_days",
    "recovery_observed",
    "end_reason",
}


def validate_episodes(data: pd.DataFrame) -> pd.DataFrame:
    """Check the event clock, cohort, and censoring labels."""
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing episode fields: {sorted(missing)}")

    data = data.copy()
    data["start"] = pd.to_datetime(
        data["disruption_start_date"], errors="raise"
    )
    data["end"] = pd.to_datetime(
        data["episode_end_date"], errors="raise"
    )
    data["duration_days"] = pd.to_numeric(
        data["duration_days"], errors="raise"
    )
    data["recovery_observed"] = pd.to_numeric(
        data["recovery_observed"], errors="raise"
    )

    if len(data) != 2_547:
        raise ValueError(f"Expected 2,547 episodes; found {len(data)}")
    if not data["start"].dt.year.isin(
        [2019, 2020, 2021, 2022]
    ).all():
        raise ValueError("Unexpected episode start year")
    if not data["recovery_observed"].isin([0, 1]).all():
        raise ValueError("Recovery indicator must be binary")
    if not data["duration_days"].ge(1).all():
        raise ValueError("Episode duration must be positive")

    inclusive_days = (data["end"] - data["start"]).dt.days + 1
    if not data["duration_days"].eq(inclusive_days).all():
        raise ValueError(
            "Duration does not match inclusive episode dates"
        )
    if data.duplicated(
        ["origin_airport", "disruption_start_date"]
    ).any():
        raise ValueError("Duplicate airport episode start")

    expected_reasons = {
        1: {"two_consecutive_normal_days"},
        0: {
            "censored_observation_gap",
            "censored_last_qualifying_day",
        },
    }
    for event, reasons in expected_reasons.items():
        observed = set(
            data.loc[
                data["recovery_observed"].eq(event),
                "end_reason",
            ]
        )
        if not observed.issubset(reasons):
            raise ValueError(
                f"Unexpected end reason for event={event}: {observed}"
            )

    data["cohort"] = data["start"].dt.year.map({
        2019: "2019",
        2020: "2020",
        2021: "2021-2022",
        2022: "2021-2022",
    })
    return data


def fit_group(
    data: pd.DataFrame,
    label: str,
) -> tuple[dict, KaplanMeierFitter]:
    """Fit an exploratory right-censored recovery curve."""
    km = KaplanMeierFitter()
    km.fit(
        durations=data["duration_days"],
        event_observed=data["recovery_observed"],
        label=label,
    )

    median = float(km.median_survival_time_)
    summary = {
        "cohort": label,
        "episodes": len(data),
        "observed_recoveries": int(
            data["recovery_observed"].sum()
        ),
        "censored_episodes": int(
            data["recovery_observed"].eq(0).sum()
        ),
        "observation_gap_censored": int(
            data["end_reason"].eq(
                "censored_observation_gap"
            ).sum()
        ),
        "censoring_fraction": float(
            data["recovery_observed"].eq(0).mean()
        ),
        "km_median_recovery_days": (
            median if np.isfinite(median) else np.nan
        ),
        # Show how many episodes still have observed follow-up at
        # each horizon. Do not extend a KM estimate past all follow-up.
        "episodes_with_follow_up_to_day_7": int(
            data["duration_days"].ge(7).sum()
        ),
        "episodes_with_follow_up_to_day_14": int(
            data["duration_days"].ge(14).sum()
        ),
        "km_recovery_fraction_by_day_7": (
            float(1 - km.predict(7))
            if data["duration_days"].ge(7).any()
            else np.nan
        ),
        "km_recovery_fraction_by_day_14": (
            float(1 - km.predict(14))
            if data["duration_days"].ge(14).any()
            else np.nan
        ),
        "interpretation": (
            "exploratory; observation-gap censoring may be informative"
        ),
    }
    return summary, km


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing episodes: {INPUT}")

    data = validate_episodes(pd.read_csv(INPUT))
    if (
        int(data["recovery_observed"].sum()) != 1_677
        or int(data["recovery_observed"].eq(0).sum()) != 870
    ):
        raise ValueError("Episode event/censor totals changed")

    overall, overall_km = fit_group(data, "All 2019-2022")
    results = [overall]
    fitted = {}

    for label in ("2019", "2020", "2021-2022"):
        subset = data.loc[data["cohort"].eq(label)]
        row, fitted[label] = fit_group(subset, label)
        results.append(row)

    table = pd.DataFrame(results)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    overall_km.plot_survival_function(
        ax=axes[0], ci_show=True
    )
    axes[0].set_title("All sampled airport episodes")
    axes[0].set_ylabel("Estimated fraction not yet recovered")

    for km in fitted.values():
        km.plot_survival_function(
            ax=axes[1], ci_show=False
        )
    axes[1].set_title("By episode start regime")
    axes[1].set_ylabel(
        "Estimated fraction not yet recovered"
    )

    for ax in axes:
        ax.set_xlabel("Episode duration (inclusive days)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Exploratory recovery: observation-gap censoring may "
        "violate KM assumptions",
        fontsize=11,
    )
    fig.tight_layout()

    table_temp = TABLE.with_name(TABLE.name + ".tmp")
    figure_temp = FIGURE.with_name(FIGURE.name + ".tmp")
    try:
        table.to_csv(table_temp, index=False)
        fig.savefig(figure_temp, format="png", dpi=160)
        os.replace(table_temp, TABLE)
        os.replace(figure_temp, FIGURE)
    finally:
        table_temp.unlink(missing_ok=True)
        figure_temp.unlink(missing_ok=True)
        plt.close(fig)

    print("Exploratory airport recovery KM: PASS")
    print(table.to_string(index=False))
    print(f"Table: {TABLE}")
    print(f"Figure: {FIGURE}")
    print(
        "Warning: 853 censored episodes end at observation "
        "gaps; independent censoring is not established."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
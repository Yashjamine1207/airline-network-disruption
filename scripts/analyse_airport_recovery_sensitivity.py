"""Compare candidate airport-recovery episode definitions.

Uses 2019-2022 only through the existing aggregation function.
Outputs descriptive episode counts; it does not fit a survival model.
"""

from pathlib import Path
import os
import sys

import pandas as pd

import scripts.build_airport_recovery_episodes as recovery


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "reports/tables/"
    "airport_recovery_sensitivity.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

SCENARIOS = [
    ("primary_candidate", 20, 0.05, 0.02, 2),
    ("minimum_10_flights", 10, 0.05, 0.02, 2),
    ("minimum_50_flights", 50, 0.05, 0.02, 2),
    ("disruption_rate_10_percent", 20, 0.10, 0.02, 2),
    ("normal_rate_3_percent", 20, 0.05, 0.03, 2),
    ("one_normal_day", 20, 0.05, 0.02, 1),
    ("three_normal_days", 20, 0.05, 0.02, 3),
]


def main() -> None:
    original_rules = (
        recovery.MIN_COMPLETED,
        recovery.DISRUPTION_RATE,
        recovery.NORMAL_RATE,
        recovery.NORMAL_DAYS_REQUIRED,
    )

    try:
        # Aggregate once at the lowest tested volume threshold.
        recovery.MIN_COMPLETED = 10
        daily = recovery.aggregate_airport_days()

        results = []

        for (
            name,
            minimum,
            disruption_rate,
            normal_rate,
            normal_days,
        ) in SCENARIOS:
            recovery.MIN_COMPLETED = minimum
            recovery.DISRUPTION_RATE = disruption_rate
            recovery.NORMAL_RATE = normal_rate
            recovery.NORMAL_DAYS_REQUIRED = normal_days

            eligible_days = daily.loc[
                daily["completed_count"] >= minimum
            ].copy()

            episodes = []
            for airport, airport_days in eligible_days.groupby(
                "origin_airport", sort=True
            ):
                episodes.extend(
                    recovery.airport_episodes(
                        airport, airport_days
                    )
                )

            episode_table = pd.DataFrame(episodes)

            if episode_table.empty:
                observed = 0
                censored = 0
                median_observed_days = float("nan")
            else:
                observed = int(
                    episode_table["recovery_observed"].sum()
                )
                censored = int(
                    episode_table["recovery_observed"]
                    .eq(0)
                    .sum()
                )
                median_observed_days = (
                    episode_table.loc[
                        episode_table["recovery_observed"].eq(1),
                        "duration_days",
                    ].median()
                )

            results.append({
                "scenario": name,
                "minimum_completed_per_airport_day": minimum,
                "disruption_rate_threshold": disruption_rate,
                "normal_rate_threshold": normal_rate,
                "consecutive_normal_days_required": normal_days,
                "qualifying_airport_days": len(eligible_days),
                "airports_with_qualifying_days": (
                    eligible_days["origin_airport"].nunique()
                ),
                "episodes": len(episodes),
                "observed_recoveries": observed,
                "censored_episodes": censored,
                "median_days_among_observed_recoveries_only": (
                    median_observed_days
                ),
            })

            print(
                f"{name}: episodes={len(episodes):,}, "
                f"observed={observed:,}, censored={censored:,}"
            )

        result = pd.DataFrame(results)

        if not (
            result["episodes"]
            .eq(
                result["observed_recoveries"]
                + result["censored_episodes"]
            )
            .all()
        ):
            raise ValueError(
                "Episode counts do not reconcile."
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        TEMP.unlink(missing_ok=True)

        try:
            result.to_csv(TEMP, index=False)
            os.replace(TEMP, OUTPUT)
        except Exception:
            TEMP.unlink(missing_ok=True)
            raise

        print(f"\nSensitivity table: {OUTPUT}")
        print(
            "The observed-only median is descriptive; "
            "it is not a censoring-adjusted recovery median."
        )

    finally:
        (
            recovery.MIN_COMPLETED,
            recovery.DISRUPTION_RATE,
            recovery.NORMAL_RATE,
            recovery.NORMAL_DAYS_REQUIRED,
        ) = original_rules


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
"""Build retrospective, sample-based airport recovery episodes.

Development and validation years only (2019-2022).
No 2023 outcome is analysed.

Candidate definitions:
- Qualifying airport-day: >=20 completed sampled flights.
- Disruption starts: severe-delay rate >=5%.
- Recovery: two consecutive qualifying calendar days <=2%.
- An observation gap censors an open episode.

This is NOT aircraft recovery or a pre-flight predictor.
"""

from collections import defaultdict
from pathlib import Path
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/flights_kaggle/flights_sample_3m.csv"
OUTPUT = (
    ROOT / "data/processed/survival_episodes/"
    "airport_recovery_episodes.csv"
)
TEMP = OUTPUT.with_suffix(".csv.tmp")

CHUNK_SIZE = 250_000
MIN_COMPLETED = 20
DISRUPTION_RATE = 0.05
NORMAL_RATE = 0.02
NORMAL_DAYS_REQUIRED = 2


def aggregate_airport_days():
    """Count completed and severely delayed sampled flights per day."""
    totals = defaultdict(lambda: [0, 0])

    for number, chunk in enumerate(
        pd.read_csv(
            RAW,
            usecols=[
                "FL_DATE",
                "ORIGIN",
                "CANCELLED",
                "DIVERTED",
                "ARR_DELAY",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        # Do not inspect final-test outcomes.
        chunk = chunk.loc[
            chunk["FL_DATE"] < "2023-01-01"
        ].copy()

        completed = (
            chunk["CANCELLED"].eq(0)
            & chunk["DIVERTED"].eq(0)
            & chunk["ARR_DELAY"].notna()
        )
        chunk["completed"] = completed.astype("int8")
        chunk["severe"] = (
            completed & chunk["ARR_DELAY"].ge(120)
        ).astype("int8")

        grouped = chunk.groupby(
            ["ORIGIN", "FL_DATE"],
            sort=False,
        )[["completed", "severe"]].sum()

        for key, values in grouped.iterrows():
            totals[key][0] += int(values["completed"])
            totals[key][1] += int(values["severe"])

        print(f"Aggregated source chunk {number}")

    daily = pd.DataFrame(
        [
            (airport, date, values[0], values[1])
            for (airport, date), values in totals.items()
        ],
        columns=[
            "origin_airport",
            "flight_date",
            "completed_count",
            "severe_count",
        ],
    )
    daily["flight_date"] = pd.to_datetime(
        daily["flight_date"], errors="raise"
    )

    daily = daily.loc[
        daily["completed_count"] >= MIN_COMPLETED
    ].copy()
    daily["severe_rate"] = (
        daily["severe_count"]
        / daily["completed_count"]
    )

    return daily.sort_values(
        ["origin_airport", "flight_date"]
    )


def airport_episodes(airport, days):
    """Construct non-overlapping episodes for one airport."""
    episodes = []
    start = None
    last_day = None
    start_rate = None
    normal_streak = 0

    def finish(end_day, observed, reason):
        return {
            "origin_airport": airport,
            "disruption_start_date": start.date().isoformat(),
            "episode_end_date": end_day.date().isoformat(),
            "duration_days": (end_day - start).days + 1,
            "recovery_observed": int(observed),
            "end_reason": reason,
            "start_severe_rate": start_rate,
            "min_completed_flights_per_day": MIN_COMPLETED,
            "disruption_rate_threshold": DISRUPTION_RATE,
            "normal_rate_threshold": NORMAL_RATE,
            "normal_days_required": NORMAL_DAYS_REQUIRED,
            "episode_code_version": "airport_recovery_v1",
        }

    for row in days.itertuples(index=False):
        day = row.flight_date
        rate = row.severe_rate

        if (
            start is not None
            and last_day is not None
            and (day - last_day).days > 1
        ):
            # We cannot know whether recovery occurred in the gap.
            episodes.append(
                finish(
                    last_day,
                    False,
                    "censored_observation_gap",
                )
            )
            start = None
            normal_streak = 0

        if start is None:
            if rate >= DISRUPTION_RATE:
                start = day
                start_rate = rate
                normal_streak = 0
        else:
            normal_streak = (
                normal_streak + 1
                if rate <= NORMAL_RATE
                else 0
            )

            if normal_streak >= NORMAL_DAYS_REQUIRED:
                episodes.append(
                    finish(
                        day,
                        True,
                        "two_consecutive_normal_days",
                    )
                )
                start = None
                normal_streak = 0

        last_day = day

    if start is not None:
        episodes.append(
            finish(
                last_day,
                False,
                "censored_last_qualifying_day",
            )
        )

    return episodes


def main():
    if not RAW.is_file():
        raise FileNotFoundError(RAW)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    daily = aggregate_airport_days()

    episodes = []
    for airport, days in daily.groupby(
        "origin_airport", sort=True
    ):
        episodes.extend(
            airport_episodes(airport, days)
        )

    result = pd.DataFrame(episodes)

    if result.empty:
        raise ValueError(
            "No episodes found; do not write an empty result."
        )

    if result["duration_days"].le(0).any():
        raise ValueError("Invalid non-positive episode duration.")

    if result["episode_end_date"].gt(
        "2022-12-31"
    ).any():
        raise ValueError("Final-test date entered episodes.")

    TEMP.unlink(missing_ok=True)
    try:
        result.to_csv(TEMP, index=False)
        os.replace(TEMP, OUTPUT)
    except Exception:
        TEMP.unlink(missing_ok=True)
        raise

    print("\nCandidate airport recovery episodes created.")
    print(f"Qualifying airport-days: {len(daily):,}")
    print(f"Episodes: {len(result):,}")
    print(
        "Observed recovery: "
        f"{result['recovery_observed'].sum():,}"
    )
    print(
        "Censored: "
        f"{result['recovery_observed'].eq(0).sum():,}"
    )
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
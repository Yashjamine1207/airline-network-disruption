"""Tests for retrospective airport recovery episode construction."""

import pandas as pd

from scripts.build_airport_recovery_episodes import (
    airport_episodes,
)


def make_days(*date_and_rate):
    """Synthetic qualifying airport-days, already sorted by date."""
    return pd.DataFrame({
        "flight_date": [
            pd.Timestamp(date)
            for date, _ in date_and_rate
        ],
        "severe_rate": [
            rate for _, rate in date_and_rate
        ],
    })


def test_two_consecutive_normal_days_end_episode():
    days = make_days(
        ("2019-01-01", 0.10),
        ("2019-01-02", 0.01),
        ("2019-01-03", 0.02),
    )

    episodes = airport_episodes("JFK", days)

    assert len(episodes) == 1
    assert episodes[0]["recovery_observed"] == 1
    assert episodes[0]["duration_days"] == 3
    assert episodes[0]["episode_end_date"] == "2019-01-03"


def test_observation_gap_censors_instead_of_claiming_recovery():
    days = make_days(
        ("2019-01-01", 0.10),
        ("2019-01-02", 0.01),
        ("2019-01-04", 0.01),
    )

    episodes = airport_episodes("JFK", days)

    assert len(episodes) == 1
    assert episodes[0]["recovery_observed"] == 0
    assert episodes[0]["end_reason"] == (
        "censored_observation_gap"
    )
    assert episodes[0]["episode_end_date"] == "2019-01-02"


def test_non_normal_day_resets_recovery_streak():
    days = make_days(
        ("2019-01-01", 0.10),
        ("2019-01-02", 0.01),
        ("2019-01-03", 0.03),
        ("2019-01-04", 0.01),
        ("2019-01-05", 0.01),
    )

    episodes = airport_episodes("JFK", days)

    assert len(episodes) == 1
    assert episodes[0]["recovery_observed"] == 1
    assert episodes[0]["episode_end_date"] == "2019-01-05"


def test_open_episode_is_censored_at_last_qualifying_day():
    days = make_days(
        ("2019-01-01", 0.10),
        ("2019-01-02", 0.01),
    )

    episodes = airport_episodes("JFK", days)

    assert len(episodes) == 1
    assert episodes[0]["recovery_observed"] == 0
    assert episodes[0]["end_reason"] == (
        "censored_last_qualifying_day"
    )


def test_new_disruption_after_recovery_starts_new_episode():
    days = make_days(
        ("2019-01-01", 0.10),
        ("2019-01-02", 0.01),
        ("2019-01-03", 0.01),
        ("2019-01-04", 0.10),
        ("2019-01-05", 0.01),
        ("2019-01-06", 0.01),
    )

    episodes = airport_episodes("JFK", days)

    assert len(episodes) == 2
    assert all(
        episode["recovery_observed"] == 1
        for episode in episodes
    )
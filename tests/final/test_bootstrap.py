"""Day-level bootstrap for PR-AUC (Phase 7B, Step 12)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from airline_disruption.final.bootstrap import DayBootstrap


def make_data(n=3000, n_days=30, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    score = rng.random(n) + 0.4 * y
    if ties:
        score = np.round(score, 1)  # many equal scores
    days = rng.integers(0, n_days, n)
    return y, score, days


@pytest.mark.parametrize("ties", [False, True])
def test_point_estimate_equals_scikit_learn_average_precision(ties):
    y, score, days = make_data(ties=ties)
    boot = DayBootstrap(y, days, n_boot=5)
    assert boot.point(score) == pytest.approx(average_precision_score(y, score), abs=1e-12)


@pytest.mark.parametrize("ties", [False, True])
def test_every_resample_equals_scikit_learn_on_the_duplicated_rows(ties):
    """A resample of whole days is the same as repeating each day's rows as often as it was drawn."""
    y, score, days = make_data(n=2000, n_days=12, ties=ties)
    boot = DayBootstrap(y, days, n_boot=6, seed=3)
    pr, lift = boot.draws(score)
    for b in range(6):
        repeats = boot.day_weights[b][boot.codes]
        index = np.repeat(np.arange(len(y)), repeats)
        assert pr[b] == pytest.approx(average_precision_score(y[index], score[index]), abs=1e-12)
        assert lift[b] == pytest.approx(pr[b] / y[index].mean(), abs=1e-12)


def test_a_perfect_ranking_has_pr_auc_one_in_every_resample():
    y, _, days = make_data()
    perfect = y + np.linspace(0, 0.01, len(y))  # events always above non-events
    boot = DayBootstrap(y, days, n_boot=20)
    pr, _ = boot.draws(perfect)
    assert boot.point(perfect) == pytest.approx(1.0)
    assert np.allclose(pr, 1.0)


def test_summary_has_the_point_inside_a_sensible_interval():
    y, score, days = make_data(n=6000, n_days=60)
    table = DayBootstrap(y, days, n_boot=200).summarise({"model": score})
    row = table.iloc[0]
    assert row["pr_auc_low"] < row["pr_auc"] < row["pr_auc_high"]
    assert row["lift_low"] < row["lift"] < row["lift_high"]
    assert row["lift"] == pytest.approx(row["pr_auc"] / y.mean())


def test_paired_difference_is_positive_for_a_better_scorer_and_absent_for_the_reference():
    y, score, days = make_data(n=6000, n_days=60)
    noise = np.random.default_rng(9).random(len(y))
    table = DayBootstrap(y, days, n_boot=200).summarise({"good": score, "noise": noise}, reference="noise").set_index("scorer")
    assert np.isnan(table.loc["noise"].get("difference_vs_reference", np.nan))
    assert table.loc["good", "difference_low"] > 0
    assert table.loc["good", "share_better"] == 1.0


def test_the_same_seed_gives_the_same_resamples_and_another_seed_a_different_one():
    y, score, days = make_data()
    a = DayBootstrap(y, days, n_boot=30, seed=1).draws(score)[0]
    b = DayBootstrap(y, days, n_boot=30, seed=1).draws(score)[0]
    c = DayBootstrap(y, days, n_boot=30, seed=2).draws(score)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_bad_inputs_are_refused():
    y, score, days = make_data(n=100)
    with pytest.raises(ValueError):
        DayBootstrap(np.zeros(100, dtype=int), days)  # one class
    with pytest.raises(ValueError):
        DayBootstrap(y, days[:50])
    boot = DayBootstrap(y, days, n_boot=3)
    with pytest.raises(ValueError):
        boot.point(score[:50])
    bad = score.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        boot.point(bad)


def test_a_day_with_many_events_pulls_the_resampled_prevalence_with_it():
    """The resampled event rate must use the day weights, not the original rates."""
    y = np.array([1] * 50 + [0] * 50 + [0] * 100)
    days = np.array([0] * 100 + [1] * 100)  # day 0 has all the events
    boot = DayBootstrap(y, days, n_boot=50, seed=5)
    _, lift = boot.draws(np.linspace(0, 1, len(y)))
    only_day_one = [b for b in range(50) if boot.day_weights[b][0] == 0]
    assert only_day_one  # some resample drew only day 1: no events, lift undefined
    assert all(np.isnan(lift[b]) for b in only_day_one)


def test_reliability_figure_writes_a_file(tmp_path):
    pytest.importorskip("matplotlib")
    import pandas as pd
    from airline_disruption.explain import plots

    table = pd.DataFrame({"mean_predicted": [0.01, 0.02, 0.04], "observed_rate": [0.012, 0.021, 0.05]})
    assert plots.reliability_figure(tmp_path / "r.png", {"a": table}, "A title", "sub")
    assert (tmp_path / "r.png").stat().st_size > 1000
    assert not plots.reliability_figure(tmp_path / "none.png", {}, "x")

"""Tests for row subsampling, benchmark alignment and the plain-language verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airline_disruption.deep import protocol as pr


def bench(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": np.arange(n) + 100,
            "prediction_day_utc": pd.Timestamp("2022-03-01"),
            "label": (np.arange(n) % 3 == 0).astype("int8"),
            "lightgbm_base_no_year": np.linspace(0.01, 0.9, n),
        }
    )


def test_two_class_guard_names_the_rows_and_the_cause() -> None:
    pr.require_two_classes(np.array([0, 1, 0]), "fit")
    with pytest.raises(ValueError, match="fit rows contain only one class.*smoke"):
        pr.require_two_classes(np.zeros(5), "fit")


def test_subsample_is_sorted_reproducible_and_a_subset() -> None:
    rows = np.arange(0, 1000, 2)
    a, b = pr.subsample_rows(rows, 50, seed=1), pr.subsample_rows(rows, 50, seed=1)
    assert len(a) == 50 and np.array_equal(a, b) and np.all(np.diff(a) > 0) and set(a) <= set(rows)
    assert not np.array_equal(a, pr.subsample_rows(rows, 50, seed=2))


def test_subsample_returns_everything_when_the_limit_is_not_binding() -> None:
    rows = np.arange(10)
    assert np.array_equal(pr.subsample_rows(rows, None, 1), rows)
    assert np.array_equal(pr.subsample_rows(rows, 10, 1), rows)
    assert np.array_equal(pr.subsample_rows(rows, 99, 1), rows)
    with pytest.raises(ValueError, match="at least 1"):
        pr.subsample_rows(rows, 0, 1)


def test_scores_come_back_in_the_requested_row_order() -> None:
    table = bench()
    ids = np.array([105, 100, 119])
    labels = table["label"].to_numpy()[[5, 0, 19]]
    scores = pr.align_scores(ids, labels, table, "lightgbm_base_no_year")
    assert scores.tolist() == table["lightgbm_base_no_year"].to_numpy()[[5, 0, 19]].tolist()


def test_a_shuffled_predictions_table_gives_the_same_alignment() -> None:
    table = bench()
    ids = table["source_row_number"].to_numpy()
    reference = pr.align_scores(ids, table["label"].to_numpy(), table, "lightgbm_base_no_year")
    shuffled = table.sample(frac=1.0, random_state=3)
    assert np.array_equal(reference, pr.align_scores(ids, table["label"].to_numpy(), shuffled, "lightgbm_base_no_year"))


def test_a_missing_row_a_wrong_label_a_duplicate_and_a_bad_score_are_all_refused() -> None:
    table = bench()
    ids, labels = table["source_row_number"].to_numpy(), table["label"].to_numpy()
    with pytest.raises(ValueError, match="missing from the predictions table"):
        pr.align_scores(np.append(ids, 9999), np.append(labels, 0), table, "lightgbm_base_no_year")
    with pytest.raises(ValueError, match="Labels .* differ"):
        pr.align_scores(ids, 1 - labels, table, "lightgbm_base_no_year")
    with pytest.raises(ValueError, match="duplicate"):
        pr.align_scores(ids, labels, pd.concat([table, table.iloc[:1]]), "lightgbm_base_no_year")
    bad = table.copy()
    bad.loc[3, "lightgbm_base_no_year"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        pr.align_scores(ids, labels, bad, "lightgbm_base_no_year")
    with pytest.raises(ValueError, match="no column 'nope'"):
        pr.align_scores(ids, labels, table, "nope")


def test_verdict_reads_the_interval_not_the_point_estimate() -> None:
    assert "is better than LightGBM" in pr.verdict(0.003, 0.001, 0.005, "LSTM", "LightGBM")
    assert "LightGBM is better than LSTM" in pr.verdict(-0.003, -0.005, -0.001, "LSTM", "LightGBM")
    text = pr.verdict(0.004, -0.001, 0.009, "LSTM", "LightGBM")
    assert "No clear difference" in text and "+0.00400" in text
    assert "No clear difference" in pr.verdict(0.0, 0.0, 0.002, "a", "b")  # an interval touching zero is not clear


# ---------------------------------------------------------------------------
# Fit variants for the 2020 ablation
# ---------------------------------------------------------------------------
def year_seconds(years: list[int]) -> np.ndarray:
    """One second-precision timestamp per entry, at noon UTC on 1 July of that year."""
    return np.array([pd.Timestamp(f"{y}-07-01 12:00", tz="UTC").timestamp() for y in years], dtype="int64")


def test_calendar_year_is_read_in_utc_at_the_year_boundaries() -> None:
    stamps = np.array(
        [pd.Timestamp(t, tz="UTC").timestamp() for t in ("2019-12-31 23:59:59", "2020-01-01 00:00:00", "2020-12-31 23:59:59", "2021-01-01 00:00:00")],
        dtype="int64",
    )
    assert pr.calendar_year_utc(stamps).tolist() == [2019, 2020, 2020, 2021]


def make_years(n_per_year: int = 100) -> tuple[np.ndarray, np.ndarray]:
    years = [2019] * n_per_year + [2020] * n_per_year + [2021] * n_per_year
    return np.arange(len(years)), year_seconds(years)


def test_full_returns_every_row_and_no2020_drops_exactly_the_2020_rows() -> None:
    rows, seconds = make_years()
    assert np.array_equal(pr.fit_variant_rows(rows, seconds, "full", 1), rows)
    kept = pr.fit_variant_rows(rows, seconds, "no2020", 1)
    assert kept.tolist() == list(range(100)) + list(range(200, 300))
    assert (pr.calendar_year_utc(seconds[kept]) != 2020).all()


def test_matched_has_the_no2020_size_but_keeps_2020_rows_and_is_seeded() -> None:
    rows, seconds = make_years()
    a = pr.fit_variant_rows(rows, seconds, "matched", 1)
    b = pr.fit_variant_rows(rows, seconds, "matched", 1)
    c = pr.fit_variant_rows(rows, seconds, "matched", 2)
    assert len(a) == 200 and np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0) and set(a) <= set(rows)
    assert (pr.calendar_year_utc(seconds[a]) == 2020).any()  # it is NOT the no-2020 set


def test_variants_work_on_a_subset_of_positions() -> None:
    """``rows`` are positions into a longer array, as in the training script."""
    _, seconds = make_years()
    rows = np.arange(50, 250)  # 50 rows of 2019, 100 of 2020, 50 of 2021
    assert len(pr.fit_variant_rows(rows, seconds, "no2020", 1)) == 100
    assert len(pr.fit_variant_rows(rows, seconds, "matched", 1)) == 100


def test_the_ablation_refuses_a_set_without_2020_or_only_2020_or_an_unknown_name() -> None:
    rows = np.arange(4)
    with pytest.raises(ValueError, match="no rows from 2020"):
        pr.fit_variant_rows(rows, year_seconds([2019, 2019, 2021, 2021]), "no2020", 1)
    with pytest.raises(ValueError, match="Every fit row"):
        pr.fit_variant_rows(rows, year_seconds([2020] * 4), "no2020", 1)
    with pytest.raises(ValueError, match="Unknown fit variant"):
        pr.fit_variant_rows(rows, year_seconds([2019, 2020, 2020, 2021]), "without", 1)


def test_fit_composition_counts_rows_events_and_rates_by_year() -> None:
    _, seconds = make_years(10)
    labels = np.array([1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [0] * 10)
    table = pr.fit_composition(seconds, labels)
    assert table["year"].tolist() == [2019, 2020, 2021]
    assert table["rows"].tolist() == [10, 10, 10] and table["events"].tolist() == [2, 5, 0]
    assert table["event_rate"].tolist() == [0.2, 0.5, 0.0]
    assert table["share_of_fit_rows"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vs_full, vs_matched, expected",
    [(0.001, 0.001, True), (0.001, -0.0002, False), (-0.0002, 0.001, False), (0.0, 0.001, False), (0.001, 0.0, False),
     (float("nan"), 0.001, False), (0.001, float("nan"), False)],
)
def test_dropping_2020_needs_both_lower_ends_above_zero(vs_full: float, vs_matched: float, expected: bool) -> None:
    assert pr.decide_drop_2020(vs_full, vs_matched) is expected

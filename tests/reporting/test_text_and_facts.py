"""Formatting helpers, the dash rule and the small fact functions (Phase 7, Step 13)."""

import numpy as np
import pandas as pd
import pytest

from airline_disruption.reporting import facts
from airline_disruption.reporting import text as tx


# ---------------------------------------------------------------------------- text helpers
def test_numbers_are_formatted_the_same_way_every_time():
    assert tx.num(0.059137) == "0.0591"
    assert tx.pct(0.1047) == "10.5%"
    assert tx.pct(0.1047, 2) == "10.47%"
    assert tx.whole(1234567.4) == "1,234,567"
    assert tx.signed(0.0157) == "+0.0157"
    assert tx.signed(-0.0032) == "-0.0032"
    assert tx.ratio_text(12.695) == "12.7"


def test_missing_values_print_as_na_never_as_nan():
    for function in (tx.num, tx.pct, tx.whole, tx.signed, tx.ratio_text):
        assert function(np.nan) == "n/a"
        assert function(None) == "n/a"


def test_an_interval_uses_words_not_a_dash_and_survives_a_missing_end():
    assert tx.interval(0.054, 0.0652) == "(0.0540 to 0.0652)"
    assert tx.interval(0.037, 0.053, 1, as_percent=True) == "(3.7% to 5.3%)"
    assert tx.interval(np.nan, 0.1) == "(interval n/a)"


def test_a_table_needs_every_column_it_names():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(KeyError, match="missing columns"):
        tx.md_table(frame, [("A", "a", None), ("C", "c", None)])


def test_a_table_renders_header_rule_and_rows_and_escapes_pipes():
    frame = pd.DataFrame({"name": ["x|y"], "value": [0.5]})
    table = tx.md_table(frame, [("Name", "name", None), ("Value", "value", tx.pct)])
    lines = table.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x/y | 50.0% |"


def test_an_empty_table_says_so_instead_of_printing_a_bare_header():
    assert tx.md_table(pd.DataFrame({"a": []}), [("A", "a", None)]) == "_(no rows)_"


@pytest.mark.parametrize("dash", [chr(0x2014), chr(0x2013), chr(0x2012), chr(0x2015)])
def test_text_with_a_long_dash_is_refused(dash):
    with pytest.raises(ValueError, match="long dash"):
        tx.check_clean(f"one {dash} two")


@pytest.mark.parametrize("bad", ["nan%", "None", "{{x}}", "TODO"])
def test_text_with_an_unfilled_placeholder_is_refused(bad):
    with pytest.raises(ValueError):
        tx.check_clean(f"value {bad} here")


def test_clean_text_passes_through_unchanged():
    assert tx.check_clean("plain text, 12.5% and (1.0 to 2.0)") == "plain text, 12.5% and (1.0 to 2.0)"


def test_sections_are_separated_by_one_blank_line_and_end_with_one_newline():
    assert tx.join_sections(["a\n\n", "", "  ", "b"]) == "a\n\nb\n"


# ---------------------------------------------------------------------------- facts
def months():
    return pd.DataFrame({"group": ["m1", "m2", "m3", "tiny"], "rows": [1000, 1000, 2000, 10], "events": [10, 10, 40, 1],
                         "alert_rate_top1pct": [0.0, 0.01, 0.02, 0.5]})


def test_shares_of_alerts_add_up_to_one_and_follow_rows_times_rate():
    shaped = facts.with_alert_shares(months().iloc[:3])
    assert shaped["alerts"].tolist() == [0.0, 10.0, 40.0]
    assert shaped["share_of_alerts"].tolist() == [0.0, 0.2, 0.8]
    assert shaped["share_of_alerts"].sum() == pytest.approx(1.0)
    assert shaped["share_of_events"].sum() == pytest.approx(1.0)


def test_concentration_names_the_busiest_groups_and_ignores_tiny_ones():
    result = facts.concentration(months(), 1, min_rows=1000)
    assert result["groups"] == ["m3"]
    assert result["share_of_alerts"] == pytest.approx(0.8)
    assert result["share_of_rows"] == pytest.approx(0.5)  # 2000 of 4000 flights once the tiny group is dropped
    assert result["share_of_events"] == pytest.approx(40 / 60)


def test_concentration_of_an_empty_set_is_empty_not_an_error():
    result = facts.concentration(months().iloc[3:], 1, min_rows=1000)
    assert result["groups"] == [] and np.isnan(result["share_of_alerts"])


def test_groups_without_alerts_means_fewer_than_half_a_flight_flagged():
    quiet = facts.groups_without_alerts(months(), min_rows=1000)
    assert quiet["group"].tolist() == ["m1"]
    sparse = pd.DataFrame({"group": ["a"], "rows": [1000], "events": [5], "alert_rate_top1pct": [0.0004]})
    assert facts.groups_without_alerts(sparse)["group"].tolist() == ["a"]   # 0.4 of a flight rounds to none


def test_is_increasing_allows_ties_but_not_a_fall():
    assert facts.is_increasing([1, 2, 2, 3])
    assert not facts.is_increasing([1, 3, 2])
    assert not facts.is_increasing([1])


def test_one_returns_exactly_one_row_or_raises():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    assert facts.one(frame, a=2).b == "x"
    with pytest.raises(KeyError):
        facts.one(frame, a=1)
    with pytest.raises(KeyError):
        facts.one(frame, a=3)


def test_claim_lines_state_the_outcome_the_numbers_and_the_rule():
    lines = facts.claim_lines({"C1": {"statement": "the rule", "lift": 1.85451, "lift_low": 1.72907, "supported": True},
                               "C2": {"statement": "other rule", "baseline": "baseline_carrier_rate", "difference": -0.0032, "supported": False}})
    assert lines[0].startswith("**C1**: supported.")
    assert "PR-AUC lift 1.8545" in lines[0] and "lower end of the lift interval 1.7291" in lines[0] and "Rule: the rule" in lines[0]
    assert lines[1].startswith("**C2**: not supported.") and "-0.0032" in lines[1] and "Baseline: baseline_carrier_rate." in lines[1]


def test_whole_rounds_to_the_nearest_integer_not_down():
    assert tx.whole(1234.6) == "1,235"
    assert tx.whole(1234.4) == "1,234"

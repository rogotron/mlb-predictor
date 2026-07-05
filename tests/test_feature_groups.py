"""Tests for the shared feature-group importance mapping."""

from __future__ import annotations

from src.models.feature_groups import (
    OTHER_GROUP,
    assign_group,
    group_importances,
    group_name_and_source,
)


def test_specific_rules_win_over_broad_sp_rule():
    # Availability and arsenal features must not be swallowed by the broad "sp_".
    assert group_name_and_source("home_sp_days_rest")[0] == "Rest & Availability"
    assert group_name_and_source("away_sp_xwoba_arsenal")[0] == "Pitch Quality"
    assert group_name_and_source("home_sp_era_l3")[0] == "Starting Pitcher Quality"
    assert group_name_and_source("home_bullpen_pitches_l1d")[0] == "Bullpen Quality + Load"


def test_win_pct_splits_between_form_and_record():
    assert group_name_and_source("home_win_pct_l20")[0] == "Recent Form (L10/L20)"
    assert group_name_and_source("home_win_pct_home_std")[0] == "Home/Away Record"


def test_unmapped_feature_is_other():
    assert assign_group("totally_unknown_feature") is None
    assert group_name_and_source("totally_unknown_feature") == OTHER_GROUP


def test_group_importances_partition_sums_to_total():
    names = [
        "home_sp_era_l3", "away_sp_era_l3",     # SP quality
        "home_bullpen_pitches_l1d",             # bullpen
        "home_wins_l10",                        # form
        "mystery_feature",                      # other
    ]
    gains = [10.0, 10.0, 40.0, 20.0, 20.0]
    groups, total = group_importances(names, gains)

    assert total == 100.0
    # Sorted by gain descending: bullpen (40) first.
    assert groups[0]["name"] == "Bullpen Quality + Load"
    assert groups[0]["pct"] == 40.0
    # Shares sum to ~100 and an Other bucket exists for the unmapped feature.
    assert abs(sum(g["pct"] for g in groups) - 100.0) < 0.5
    assert any(g["name"] == OTHER_GROUP[0] and g["pct"] == 20.0 for g in groups)


def test_zero_gain_groups_are_dropped():
    # A model with no bullpen features should not surface a bullpen group.
    names = ["home_sp_era_l3", "home_wins_l10"]
    gains = [1.0, 1.0]
    groups, _ = group_importances(names, gains)
    assert all(g["name"] != "Bullpen Quality + Load" for g in groups)

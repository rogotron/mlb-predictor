"""Tests for the dashboard model-factor summary and per-game factors.

Guards the factor-honesty fix: the deployed model's real importances drive the
displayed factors, and no synthetic ("Home Field Factor") or excluded
(lineup/BvP, pitch-arsenal) groups leak in.
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.assemble import _build_factors, build_model_factor_summary


def _fake_pregame_model():
    # A stand-in for a fitted LightGBM model: only pregame_safe-style features.
    names = [
        "home_sp_era_l3", "away_sp_era_l3",
        "home_bullpen_pitches_l1d", "away_bullpen_xwoba_against_l14",
        "home_wins_l10", "away_run_diff_l20",
        "home_runs_per_game_std",
        "pf_runs",
    ]
    gains = [30.0, 30.0, 55.0, 25.0, 15.0, 15.0, 10.0, 2.0]
    return SimpleNamespace(feature_name_=names, feature_importances_=gains)


def test_summary_reports_only_real_groups():
    summary = build_model_factor_summary(_fake_pregame_model())
    names = {g["name"] for g in summary["factorGroups"]}

    assert summary["featureCount"] == 8
    assert summary["importanceMetric"] == "lightgbm_split_gain"
    assert "Starting Pitcher Quality" in names
    assert "Bullpen Quality + Load" in names
    # Excluded / synthetic groups must never appear for a pregame_safe model.
    assert "Home Field Factor" not in names
    assert "Lineup Matchup" not in names
    assert "Pitch Quality" not in names
    # Shares sum to ~100.
    assert abs(sum(g["pct"] for g in summary["factorGroups"]) - 100.0) < 1.0


def test_per_game_factors_carry_source_and_note():
    aw_p = {"name": "Ace A"}
    hm_p = {"name": "Hurler H"}
    rec = {"wPct": ".540"}
    factors = _build_factors(
        None, _fake_pregame_model(), aw_p, hm_p,
        ["W", "L"], ["L", "W"], 4.5, 3.9, rec, rec,
    )
    top = factors[0]
    assert top["name"] == "Bullpen Quality + Load"  # highest gain (40+20)
    assert "source" in top and "pct" in top and "note" in top
    # The pitcher-quality group carries this game's matchup in its note.
    sp = next(f for f in factors if f["name"] == "Starting Pitcher Quality")
    assert "Ace A" in sp["note"] and "Hurler H" in sp["note"]

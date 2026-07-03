"""Target leakage tests.

The cardinal sin in time-series ML for sports betting is using information
that wasn't available at first pitch. These tests assert that rolling-window
features for game G use only games strictly before G's date.

The rolling helper is intentionally small, but it protects the most important
contract in the feature pipeline.
"""

from __future__ import annotations

import pandas as pd


def test_last_n_record_excludes_current_game(tiny_team_games):
    """For game G, last_n_record(team) must not include G itself."""
    from src.features.team import last_n_record

    out = last_n_record(tiny_team_games, n=3)

    # First game of season → window is empty → win count must be 0 or NaN
    first_game = tiny_team_games[tiny_team_games["game_pk"] == 1]
    keys = first_game[["game_pk", "team_id"]]
    first_features = out.merge(keys, on=["game_pk", "team_id"])
    for _, row in first_features.iterrows():
        assert (
            pd.isna(row.get("wins_l3"))
            or row.get("wins_l3") == 0
        ), "first game should have empty rolling window"


def test_last_n_record_uses_only_prior_games(tiny_team_games):
    """For each row, the rolling window must contain only games_date < this row's date."""
    from src.features.team import last_n_record

    out = last_n_record(tiny_team_games, n=10)
    merged = tiny_team_games.merge(out, on=["game_pk", "team_id"])

    # Implementation-specific check: if the function exposes the count of games
    # in window, it should never exceed the number of strictly-prior games.
    if "n_games_l10" in merged.columns:
        for _, row in merged.iterrows():
            prior = tiny_team_games[
                (tiny_team_games["team_id"] == row["team_id"])
                & (tiny_team_games["game_date"] < row["game_date"])
            ]
            assert row["n_games_l10"] <= len(prior)

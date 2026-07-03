"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def tiny_team_games() -> pd.DataFrame:
    """A two-team, ten-game synthetic log for unit tests.

    Schema matches what src.features.team functions expect:
        game_pk, game_date, team_id, runs_for, runs_against, won
    """
    rows = []
    game_pk = 1
    for i in range(10):
        date_str = f"2024-04-{i + 1:02d}"
        # Team 1 wins on even days, team 2 on odd
        rf1, ra1 = (5, 3) if i % 2 == 0 else (2, 4)
        rows.append({
            "game_pk": game_pk,
            "game_date": pd.Timestamp(date_str),
            "team_id": 1,
            "runs_for": rf1,
            "runs_against": ra1,
            "won": rf1 > ra1,
        })
        rows.append({
            "game_pk": game_pk,
            "game_date": pd.Timestamp(date_str),
            "team_id": 2,
            "runs_for": ra1,
            "runs_against": rf1,
            "won": ra1 > rf1,
        })
        game_pk += 1
    return pd.DataFrame(rows)

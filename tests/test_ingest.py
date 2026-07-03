from __future__ import annotations

import pandas as pd

from src.data.ingest import process_games


def test_process_games_requires_final_scores() -> None:
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-05-01T23:05:00Z",
                "official_date": "2026-05-01",
                "status": "Final",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 4,
                "away_score": 2,
                "venue_id": 10,
                "venue_name": "Test Park",
            },
            {
                "game_pk": 2,
                "game_date": "2026-05-02T23:05:00Z",
                "official_date": "2026-05-02",
                "status": "Final",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": None,
                "away_score": None,
                "venue_id": 10,
                "venue_name": "Test Park",
            },
        ]
    )

    games = process_games(schedule, pd.DataFrame())

    assert games["game_pk"].tolist() == [1]
    assert games.loc[0, "target_home_win"] == 1
    assert games.loc[0, "target_total_runs"] == 6

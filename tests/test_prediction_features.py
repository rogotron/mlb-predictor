from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.build import (
    build_gamelog_pitcher_prediction_features,
    build_prediction_input,
)


def _game(
    game_pk: int,
    game_date: str,
    home_score: int,
    away_score: int,
    home_team_id: int = 1,
    away_team_id: int = 2,
) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": pd.Timestamp(game_date),
        "official_date": pd.Timestamp(game_date).date(),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_score": home_score,
        "away_score": away_score,
        "venue_id": 10,
        "venue_name": "Test Park",
        "target_home_win": int(home_score > away_score),
        "target_total_runs": home_score + away_score,
    }


def test_prediction_input_includes_most_recent_completed_game(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    pd.DataFrame(
        [
            _game(1, "2024-04-01", 5, 3),
            _game(2, "2024-04-02", 2, 4),
            _game(3, "2024-04-03", 6, 1),
        ]
    ).to_parquet(games_dir / "games_2024.parquet", index=False)

    slate = pd.DataFrame(
        [
            {
                "game_pk": 4,
                "game_date": "2024-04-04T23:05:00Z",
                "official_date": date(2024, 4, 4),
                "home_team_id": 1,
                "away_team_id": 2,
                "venue_id": 10,
            }
        ]
    )

    features = build_prediction_input(slate, tmp_path, target_date=date(2024, 4, 4))

    assert features.loc[0, "home_wins_l5"] == 2
    assert features.loc[0, "home_run_diff_l5"] == 5
    assert features.loc[0, "away_wins_l5"] == 1


def test_pitcher_prediction_features_include_latest_prior_start(tmp_path):
    logs_dir = tmp_path / "pitching_gamelogs"
    logs_dir.mkdir()
    pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": pd.Timestamp("2024-04-01"),
                "pitcher_id": 99,
                "side": "home",
                "ip": 5.0,
                "h": 5,
                "er": 5,
                "bb": 1,
                "k": 5,
                "hr": 1,
                "bf": 21,
            },
            {
                "game_pk": 2,
                "game_date": pd.Timestamp("2024-04-05"),
                "pitcher_id": 99,
                "side": "home",
                "ip": 6.0,
                "h": 3,
                "er": 1,
                "bb": 2,
                "k": 8,
                "hr": 0,
                "bf": 23,
            },
        ]
    ).to_parquet(logs_dir / "2024.parquet", index=False)

    slate = pd.DataFrame(
        [
            {
                "game_pk": 3,
                "game_date": "2024-04-10T23:05:00Z",
                "official_date": date(2024, 4, 10),
                "home_sp_id": 99,
                "away_sp_id": None,
            }
        ]
    )

    features = build_gamelog_pitcher_prediction_features(
        slate,
        tmp_path,
        target_date=date(2024, 4, 10),
    )

    assert features.loc[0, "home_sp_ip_per_start_l3"] == 5.5
    assert round(features.loc[0, "home_sp_era_std"], 3) == round(6 * 9 / 11, 3)

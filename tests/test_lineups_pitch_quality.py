from __future__ import annotations

import pandas as pd

from src.data.lineups import parse_lineup_payload
from src.data.pitch_quality import bullpen_pitch_quality_by_team, normalize_pitch_quality


def test_parse_lineup_payload_historical_confirmed_order():
    game_pk = 746002
    home_order = [592450, 665742, 660271, 547180, 641355, 518692, 669224, 663993, 621043]
    away_order = [665489, 608369, 621566, 656305, 642715, 663624, 543760, 641645, 666158]
    payload = {
        "gamePk": game_pk,
        "gameData": {"status": {"abstractGameState": "Final"}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "home": {
                        "battingOrder": home_order,
                        "pitchers": [605400],
                        "players": {
                            "ID605400": {"person": {"pitchHand": {"code": "R"}}},
                        },
                    },
                    "away": {
                        "battingOrder": away_order,
                        "pitchers": [666201],
                        "players": {
                            "ID666201": {"person": {"pitchHand": {"code": "L"}}},
                        },
                    },
                }
            }
        },
    }

    df = parse_lineup_payload(payload, historical=True)

    assert df.loc[0, "game_pk"] == game_pk
    assert df.loc[0, "home_lineup_ids"] == home_order
    assert df.loc[0, "away_lineup_ids"] == away_order
    assert df.loc[0, "lineup_status"] == "confirmed"
    assert df.loc[0, "home_sp_hand"] == "R"
    assert df.loc[0, "away_sp_hand"] == "L"


def test_pitch_quality_normalize_and_bullpen_weighted_join(tmp_path):
    teams = pd.DataFrame(
        [
            {"team_id": 147, "team_name": "Yankees", "abbreviation": "NYY"},
            {"team_id": 121, "team_name": "Mets", "abbreviation": "NYM"},
        ]
    )
    teams.to_parquet(tmp_path / "teams.parquet", index=False)
    raw = pd.DataFrame(
        [
            {"player_id": 669373, "team_name_alt": "DET", "pitch_type": "FF", "run_value_per_100": 0.7, "pitches": 244, "pitch_usage": 39.0, "est_woba": 0.331, "whiff_percent": 21.1},
            {"player_id": 669373, "team_name_alt": "DET", "pitch_type": "CH", "run_value_per_100": -2.4, "pitches": 188, "pitch_usage": 30.0, "est_woba": 0.241, "whiff_percent": 35.0},
            {"player_id": 99901, "team_name_alt": "NYY", "pitch_type": "FF", "run_value_per_100": -1.0, "pitches": 100, "pitch_usage": 70.0, "est_woba": 0.290, "whiff_percent": 30.0},
            {"player_id": 99902, "team_name_alt": "NYY", "pitch_type": "SL", "run_value_per_100": 1.0, "pitches": 50, "pitch_usage": 50.0, "est_woba": 0.320, "whiff_percent": 25.0},
        ]
    )

    quality = normalize_pitch_quality(raw)
    skubal = quality[quality["player_id"] == 669373].iloc[0]
    expected_rv = (0.7 * 39 + -2.4 * 30) / (39 + 30)
    assert round(skubal["rv_per_100"], 3) == round(expected_rv, 3)
    assert 0.20 <= skubal["xwoba_arsenal"] <= 0.36

    bullpen = bullpen_pitch_quality_by_team(quality[quality["player_id"] != 669373], raw_dir=tmp_path)
    row = bullpen[bullpen["team_id"] == 147].iloc[0]
    assert round(row["bp_rv_per_100_weighted"], 3) == round((-1.0 * 100 + 1.0 * 50) / 150, 3)
    assert round(row["bp_xwoba_arsenal_weighted"], 3) == round((0.290 * 100 + 0.320 * 50) / 150, 3)

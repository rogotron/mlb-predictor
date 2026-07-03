from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from src.data.update import (
    build_schedule_audit,
    filter_schedule_slate,
    today_in_schedule_timezone,
    write_schedule_audit,
)


def _game(
    game_pk: int,
    *,
    official_date: str = "2026-05-24",
    game_date: str = "2026-05-24T17:05:00Z",
    status: str = "Preview",
    status_detailed: str = "Scheduled",
    away_sp_id: int | None = 101,
    home_sp_id: int | None = 201,
    game_number: int = 1,
    doubleheader: str = "N",
) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "official_date": official_date,
        "away_team_id": 10,
        "home_team_id": 20,
        "away_team_name": "Away Club",
        "home_team_name": "Home Club",
        "away_sp_id": away_sp_id,
        "away_sp_name": "Away Starter" if away_sp_id else None,
        "home_sp_id": home_sp_id,
        "home_sp_name": "Home Starter" if home_sp_id else None,
        "scheduled_start_utc": game_date,
        "status": status,
        "status_detailed": status_detailed,
        "status_code": "S",
        "doubleheader": doubleheader,
        "game_number": game_number,
    }


def test_schedule_audit_normal_mlb_day_includes_final_and_tbd_probables(tmp_path) -> None:
    requested = date(2026, 5, 24)
    raw = pd.DataFrame(
        [
            _game(1, status="Final", status_detailed="Final"),
            _game(2, away_sp_id=None, home_sp_id=None),
        ]
    )

    filtered = filter_schedule_slate(raw, requested_date=requested)
    audit = write_schedule_audit(requested, raw, filtered, tmp_path / "schedule_audit.json")

    assert filtered["game_pk"].tolist() == [1, 2]
    assert audit["requested_date"] == "2026-05-24"
    assert audit["timezone_used"] == "America/New_York"
    assert audit["raw_games_count"] == 2
    assert audit["filtered_games_count"] == 2
    assert audit["excluded_games"] == []
    assert audit["included_games"][1]["missing_probable_pitchers"] == ["away", "home"]
    assert (tmp_path / "schedule_audit.json").exists()


def test_schedule_audit_doubleheader_keeps_both_games() -> None:
    requested = date(2026, 5, 24)
    raw = pd.DataFrame(
        [
            _game(
                10,
                game_date="2026-05-24T17:05:00Z",
                game_number=1,
                doubleheader="Y",
            ),
            _game(
                11,
                game_date="2026-05-24T23:05:00Z",
                game_number=2,
                doubleheader="Y",
            ),
        ]
    )

    filtered = filter_schedule_slate(raw, requested_date=requested)
    audit = build_schedule_audit(requested, raw, filtered)

    assert filtered["game_pk"].tolist() == [10, 11]
    assert [game["game_number"] for game in audit["included_games"]] == [1, 2]
    assert [game["doubleheader"] for game in audit["included_games"]] == ["Y", "Y"]


def test_schedule_audit_postponed_game_is_excluded_with_reason() -> None:
    requested = date(2026, 5, 24)
    raw = pd.DataFrame(
        [
            _game(20),
            _game(21, status="Preview", status_detailed="Postponed"),
        ]
    )

    filtered = filter_schedule_slate(raw, requested_date=requested)
    audit = build_schedule_audit(requested, raw, filtered)

    assert filtered["game_pk"].tolist() == [20]
    assert audit["filtered_games_count"] == 1
    assert audit["excluded_games"][0]["game_pk"] == 21
    assert audit["excluded_games"][0]["reasons"] == ["postponed"]


def test_schedule_audit_late_night_game_uses_mlb_official_date_not_utc_date() -> None:
    requested = date(2026, 5, 24)
    raw = pd.DataFrame(
        [
            _game(
                30,
                official_date="2026-05-24",
                game_date="2026-05-25T02:05:00Z",
            ),
            _game(
                31,
                official_date="2026-05-25",
                game_date="2026-05-25T23:05:00Z",
            ),
        ]
    )

    filtered = filter_schedule_slate(raw, requested_date=requested)
    audit = build_schedule_audit(requested, raw, filtered)

    assert filtered["game_pk"].tolist() == [30]
    assert audit["excluded_games"][0]["game_pk"] == 31
    assert audit["excluded_games"][0]["reasons"] == ["official_date_mismatch"]
    assert today_in_schedule_timezone(
        datetime(2026, 5, 25, 3, 30, tzinfo=UTC)
    ) == date(2026, 5, 24)

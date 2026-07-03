"""Daily incremental refresh.

Run from cron / scheduled task each morning. Fetches:
  - Yesterday's final scores (to extend training set)
  - Today's slate: probable starters, lineups, weather

Uses MLB Stats API (statsapi.mlb.com) — keyless.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
SCHEDULE_TIMEZONE_NAME = "America/New_York"
SCHEDULE_TIMEZONE = ZoneInfo(SCHEDULE_TIMEZONE_NAME)
_SESSION = requests.Session()
_SESSION.trust_env = False

SCHEDULE_AUDIT_PATH = PROCESSED_DIR / "schedule_audit.json"
SCHEDULE_COLUMNS = [
    "game_pk",
    "game_date",
    "official_date",
    "game_type",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "home_sp_id",
    "away_sp_id",
    "home_sp_name",
    "away_sp_name",
    "home_lineup_ids",
    "away_lineup_ids",
    "venue_id",
    "venue_name",
    "scheduled_start_utc",
    "status",
    "status_detailed",
    "status_code",
    "doubleheader",
    "game_number",
    "away_score",
    "home_score",
]


def today_in_schedule_timezone(now: datetime | None = None) -> date:
    """Return today's MLB schedule date in America/New_York."""
    if now is None:
        return datetime.now(SCHEDULE_TIMEZONE).date()
    if now.tzinfo is None:
        return now.replace(tzinfo=SCHEDULE_TIMEZONE).date()
    return now.astimezone(SCHEDULE_TIMEZONE).date()


def _person_payload(person: dict | None) -> tuple[int | None, str | None]:
    if not person:
        return None, None
    return person.get("id"), person.get("fullName") or person.get("name")


def _parse_schedule_response(data: dict) -> list[dict]:
    """Extract one row per game from a raw schedule API response."""
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]

            home_sp_id, home_sp_name = _person_payload(home.get("probablePitcher"))
            away_sp_id, away_sp_name = _person_payload(away.get("probablePitcher"))
            status = g.get("status", {}) or {}

            lineups = g.get("lineups", {})
            home_lineup = [p["id"] for p in lineups.get("homePlayers", [])] or None
            away_lineup = [p["id"] for p in lineups.get("awayPlayers", [])] or None

            games.append({
                "game_pk": g["gamePk"],
                "game_date": g["gameDate"],
                "official_date": g.get("officialDate") or d["date"],
                "game_type": g.get("gameType"),
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_team_name": home["team"].get("name"),
                "away_team_name": away["team"].get("name"),
                "home_sp_id": home_sp_id,
                "away_sp_id": away_sp_id,
                "home_sp_name": home_sp_name,
                "away_sp_name": away_sp_name,
                "home_lineup_ids": home_lineup,
                "away_lineup_ids": away_lineup,
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "scheduled_start_utc": g.get("gameDate"),
                "status": status.get("abstractGameState"),
                "status_detailed": status.get("detailedState"),
                "status_code": status.get("codedGameState") or status.get("statusCode"),
                "doubleheader": g.get("doubleHeader"),
                "game_number": g.get("gameNumber"),
                "away_score": away.get("score"),
                "home_score": home.get("score"),
            })
    return games


def _schedule_frame(games: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(games, columns=SCHEDULE_COLUMNS)


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _status_text(row: pd.Series) -> str:
    values = [row.get("status"), row.get("status_detailed"), row.get("status_code")]
    return " ".join(str(v).lower() for v in values if pd.notna(v))


def _exclusion_reasons(row: pd.Series, requested_date: date | None = None) -> list[str]:
    reasons: list[str] = []
    if requested_date is not None and _coerce_date(row.get("official_date")) != requested_date:
        reasons.append("official_date_mismatch")

    status_text = _status_text(row)
    if "postpon" in status_text:
        reasons.append("postponed")
    if "cancel" in status_text:
        reasons.append("canceled")
    if "suspend" in status_text:
        reasons.append("suspended")
    if "forfeit" in status_text:
        reasons.append("forfeit")
    return reasons


def filter_schedule_slate(slate: pd.DataFrame, requested_date: date | None = None) -> pd.DataFrame:
    """Remove games that should not appear on the requested MLB slate."""
    if slate.empty:
        return slate.copy()

    keep = [
        not _exclusion_reasons(row, requested_date=requested_date)
        for _, row in slate.iterrows()
    ]
    filtered = slate.loc[keep].copy()
    if filtered.empty:
        return filtered.reset_index(drop=True)

    filtered["official_date"] = pd.to_datetime(filtered["official_date"]).dt.date
    filtered["_scheduled_sort"] = pd.to_datetime(
        filtered["scheduled_start_utc"], errors="coerce", utc=True
    )
    filtered["_game_number_sort"] = pd.to_numeric(
        filtered.get("game_number"), errors="coerce"
    ).fillna(0)
    filtered = filtered.sort_values(
        ["_scheduled_sort", "_game_number_sort", "game_pk"],
        na_position="last",
    ).drop(columns=["_scheduled_sort", "_game_number_sort"])
    return filtered.reset_index(drop=True)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _audit_game(row: pd.Series, reasons: list[str] | None = None) -> dict[str, Any]:
    missing_probables = []
    if pd.isna(row.get("away_sp_id")):
        missing_probables.append("away")
    if pd.isna(row.get("home_sp_id")):
        missing_probables.append("home")

    entry = {
        "game_pk": _json_scalar(row.get("game_pk")),
        "official_date": _json_scalar(_coerce_date(row.get("official_date"))),
        "scheduled_start_utc": _json_scalar(row.get("scheduled_start_utc")),
        "status": _json_scalar(row.get("status")),
        "status_detailed": _json_scalar(row.get("status_detailed")),
        "status_code": _json_scalar(row.get("status_code")),
        "doubleheader": _json_scalar(row.get("doubleheader")),
        "game_number": _json_scalar(row.get("game_number")),
        "away_team_id": _json_scalar(row.get("away_team_id")),
        "away_team_name": _json_scalar(row.get("away_team_name")),
        "home_team_id": _json_scalar(row.get("home_team_id")),
        "home_team_name": _json_scalar(row.get("home_team_name")),
        "away_sp_id": _json_scalar(row.get("away_sp_id")),
        "away_sp_name": _json_scalar(row.get("away_sp_name")),
        "home_sp_id": _json_scalar(row.get("home_sp_id")),
        "home_sp_name": _json_scalar(row.get("home_sp_name")),
        "missing_probable_pitchers": missing_probables,
    }
    if reasons is not None:
        entry["reasons"] = reasons
    return entry


def build_schedule_audit(
    requested_date: date,
    raw_slate: pd.DataFrame,
    filtered_slate: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-serializable diagnostic for slate filtering."""
    included_game_pks = set(filtered_slate["game_pk"].tolist()) if "game_pk" in filtered_slate else set()
    excluded_games = []
    for _, row in raw_slate.iterrows():
        if row.get("game_pk") in included_game_pks:
            continue
        reasons = _exclusion_reasons(row, requested_date=requested_date)
        excluded_games.append(_audit_game(row, reasons or ["filtered_out"]))

    return {
        "requested_date": requested_date.isoformat(),
        "timezone_used": SCHEDULE_TIMEZONE_NAME,
        "raw_games_count": int(len(raw_slate)),
        "filtered_games_count": int(len(filtered_slate)),
        "excluded_games": excluded_games,
        "included_games": [_audit_game(row) for _, row in filtered_slate.iterrows()],
    }


def write_schedule_audit(
    requested_date: date,
    raw_slate: pd.DataFrame,
    filtered_slate: pd.DataFrame,
    audit_path: Path = SCHEDULE_AUDIT_PATH,
) -> dict[str, Any]:
    """Write schedule_audit.json and return the payload."""
    audit = build_schedule_audit(requested_date, raw_slate, filtered_slate)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return audit


def _fetch_probables_from_live_feed(game_pk: int) -> dict[str, int | str | None]:
    """Fetch probable pitchers from the live-feed endpoint as a fallback.

    The schedule endpoint sometimes omits probables depending on the hydrate
    string or how far out the slate is. The live feed is slower but tends to
    expose the same probable pitcher objects once MLB has announced them.
    """
    try:
        r = _SESSION.get(f"{BASE_URL}.1/game/{game_pk}/feed/live", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.debug("live-feed probable fallback failed game_pk=%s: %s", game_pk, exc)
        return {}

    game_data = data.get("gameData", {})
    probables = game_data.get("probablePitchers", {}) or {}
    home_id, home_name = _person_payload(probables.get("home"))
    away_id, away_name = _person_payload(probables.get("away"))

    if home_id is None or away_id is None:
        teams = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
        if home_id is None:
            home_id, home_name = _person_payload(teams.get("home", {}).get("probablePitcher"))
        if away_id is None:
            away_id, away_name = _person_payload(teams.get("away", {}).get("probablePitcher"))

    return {
        "home_sp_id": home_id,
        "home_sp_name": home_name,
        "away_sp_id": away_id,
        "away_sp_name": away_name,
    }


def _fill_missing_probables(slate: pd.DataFrame) -> pd.DataFrame:
    if slate.empty:
        return slate

    slate = slate.copy()
    missing = slate["home_sp_id"].isna() | slate["away_sp_id"].isna()
    for idx, row in slate.loc[missing].iterrows():
        probables = _fetch_probables_from_live_feed(int(row["game_pk"]))
        for col, value in probables.items():
            if value is not None and pd.isna(slate.at[idx, col]):
                slate.at[idx, col] = value
    return slate


def fetch_today_slate(
    target_date: date,
    fill_probables: bool = True,
    audit_path: Path | None = SCHEDULE_AUDIT_PATH,
) -> pd.DataFrame:
    """Scheduled games for a single date with probable starters and lineups.

    Returns DataFrame with: game_pk, game_date, official_date,
    home/away team_id, home/away sp_id, home/away sp_name,
    home/away lineup_ids, venue_id, venue_name, scheduled_start_utc, status.

    sp_id / sp_name are None when the probable has not been announced.
    Lineups are None until the card is posted (~60-90 min before first pitch).
    """
    r = _SESSION.get(
        f"{BASE_URL}/schedule",
        params={
            "date": target_date.strftime("%Y-%m-%d"),
            "sportId": 1,
            "gameType": "R",
            "hydrate": "probablePitcher,lineups,venue",
        },
        timeout=15,
    )
    r.raise_for_status()
    raw_slate = _schedule_frame(_parse_schedule_response(r.json()))
    slate = filter_schedule_slate(raw_slate, requested_date=target_date)
    if audit_path is not None:
        write_schedule_audit(target_date, raw_slate, slate, audit_path=audit_path)
    return _fill_missing_probables(slate) if fill_probables else slate


def fetch_slate_range(start_date: date, end_date: date, fill_probables: bool = True) -> pd.DataFrame:
    """Scheduled games for a date range with probable starters.

    Fetches all dates in [start_date, end_date] in a single API call.
    Probables are typically announced 1-2 days in advance; games further
    out will have sp_id / sp_name as None.

    Returns the same schema as fetch_today_slate plus an 'official_date'
    column (YYYY-MM-DD string) for grouping by day.
    """
    r = _SESSION.get(
        f"{BASE_URL}/schedule",
        params={
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "sportId": 1,
            "gameType": "R",
            "hydrate": "probablePitcher,lineups,venue",
        },
        timeout=15,
    )
    r.raise_for_status()
    games = _parse_schedule_response(r.json())
    if not games:
        return pd.DataFrame()
    df = _schedule_frame(games)
    df = filter_schedule_slate(df)
    if df.empty:
        return df
    df["official_date"] = pd.to_datetime(df["official_date"]).dt.date
    df = df.drop_duplicates(subset=["game_pk"]).reset_index(drop=True)
    return _fill_missing_probables(df) if fill_probables else df


def fetch_weather_for_slate(slate: pd.DataFrame) -> pd.DataFrame:
    """Forecast at first pitch for each outdoor venue in the slate.

    TODO: NWS api.weather.gov; require NWS_USER_AGENT env var.
    """
    raise NotImplementedError


def append_yesterday_results(yesterday: date, processed_dir: Path) -> None:
    """Append yesterday's final scores to the processed game log."""
    raise NotImplementedError

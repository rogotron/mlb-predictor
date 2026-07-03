"""Fetch and cache posted MLB lineups from the Stats API.

Cache layout:
    data/raw/lineups/{game_pk}.parquet

Each cached row contains the home/away batting order, starting pitcher IDs,
starter handedness, and lineup status. Historical games are treated as
confirmed when a nine-player batting order is present because the boxscore is
post-game ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1.1"
_SESSION = requests.Session()
_SESSION.trust_env = False

LINEUP_COLS = [
    "game_pk",
    "home_lineup_ids",
    "away_lineup_ids",
    "home_lineup_status",
    "away_lineup_status",
    "lineup_status",
    "home_sp_id",
    "away_sp_id",
    "home_sp_hand",
    "away_sp_hand",
]


def _lineup_path(raw_dir: Path, game_pk: int) -> Path:
    return raw_dir / "lineups" / f"{int(game_pk)}.parquet"


def _person_id(person: dict[str, Any] | None) -> int | None:
    if not person:
        return None
    value = person.get("id")
    return int(value) if value is not None else None


def _side_payload(payload: dict[str, Any], side: str) -> dict[str, Any]:
    return (
        payload.get("liveData", {})
        .get("boxscore", {})
        .get("teams", {})
        .get(side, {})
    )


def _starter_from_boxscore(team: dict[str, Any]) -> tuple[int | None, str | None]:
    pitchers = team.get("pitchers") or []
    players = team.get("players") or {}
    if not pitchers:
        probable = team.get("probablePitcher")
        return _person_id(probable), None

    starter_id = int(pitchers[0])
    player = players.get(f"ID{starter_id}", {}).get("person", {})
    hand = player.get("pitchHand", {}).get("code")
    return starter_id, hand


def _status_for_order(order: list[int], historical: bool, game_state: str | None) -> str:
    if len(order) >= 9:
        if historical or game_state in {"Live", "Final"}:
            return "confirmed"
        return "projected"
    return "missing"


def parse_lineup_payload(
    payload: dict[str, Any],
    game_pk: int | None = None,
    historical: bool = False,
) -> pd.DataFrame:
    """Parse a Stats API live-feed payload into one lineup cache row."""
    game_pk = int(game_pk or payload.get("gamePk") or payload.get("game_pk"))
    game_state = (
        payload.get("gameData", {})
        .get("status", {})
        .get("abstractGameState")
    )

    row: dict[str, Any] = {"game_pk": game_pk}
    statuses: list[str] = []
    for side in ("home", "away"):
        team = _side_payload(payload, side)
        order = [int(pid) for pid in team.get("battingOrder", []) if pid is not None]
        sp_id, sp_hand = _starter_from_boxscore(team)
        status = _status_for_order(order, historical=historical, game_state=game_state)
        row[f"{side}_lineup_ids"] = order[:9] if len(order) >= 9 else None
        row[f"{side}_lineup_status"] = status
        row[f"{side}_sp_id"] = sp_id
        row[f"{side}_sp_hand"] = sp_hand
        statuses.append(status)

    if all(status == "confirmed" for status in statuses):
        row["lineup_status"] = "confirmed"
    elif any(status == "projected" for status in statuses):
        row["lineup_status"] = "projected"
    else:
        row["lineup_status"] = "missing"

    return pd.DataFrame([row], columns=LINEUP_COLS)


def fetch_game_lineup(
    game_pk: int,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
    historical: bool = False,
) -> pd.DataFrame:
    """Fetch and cache one game lineup row keyed by gamePk."""
    path = _lineup_path(raw_dir, game_pk)
    if path.exists() and not force:
        return pd.read_parquet(path)

    logger.info("fetching lineup for game_pk=%s", game_pk)
    r = _SESSION.get(f"{BASE_URL}/game/{int(game_pk)}/feed/live", timeout=15)
    r.raise_for_status()
    df = parse_lineup_payload(r.json(), game_pk=game_pk, historical=historical)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_lineups_for_games(
    game_pks: list[int] | pd.Series,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
    historical: bool = False,
) -> pd.DataFrame:
    """Load cached lineups, fetching missing gamePk files through data ingest."""
    rows = []
    for raw_pk in pd.Series(game_pks).dropna().astype(int).drop_duplicates():
        try:
            rows.append(
                fetch_game_lineup(
                    int(raw_pk),
                    raw_dir=raw_dir,
                    force=force,
                    historical=historical,
                )
            )
        except Exception as exc:
            logger.warning("lineup unavailable game_pk=%s: %s", raw_pk, exc)
            rows.append(pd.DataFrame([{"game_pk": int(raw_pk)}], columns=LINEUP_COLS))

    if not rows:
        return pd.DataFrame(columns=LINEUP_COLS)
    return pd.concat(rows, ignore_index=True)

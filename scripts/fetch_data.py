"""Fetch historical MLB data and cache to data/raw/.

Example:
    python scripts/fetch_data.py --start 2018 --end 2024
    python scripts/fetch_data.py --start 2024 --end 2024 --force

Uses MLB Stats API. Cached parquet files mean subsequent runs are nearly instant.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd
import requests

from src.data.update import today_in_schedule_timezone
from src.utils.logging import configure_logging
from src.utils.paths import RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# MLB Stats API base URL
BASE_URL = "https://statsapi.mlb.com/api/v1"

# Season date ranges (approximate - MLB season runs April-October)
SEASON_START_MONTH = 4
SEASON_END_MONTH = 10


def fetch_mlb_teams() -> pd.DataFrame:
    """Fetch all active MLB teams."""
    r = requests.get(f"{BASE_URL}/teams?sportId=1")
    r.raise_for_status()
    data = r.json()
    teams = [t for t in data["teams"] if t.get("active")]
    return pd.DataFrame([
        {
            "team_id": t["id"],
            "team_name": t["teamName"],
            "abbreviation": t.get("abbreviation", ""),
            "league_id": t.get("league", {}).get("id"),
            "sport_id": t.get("sport", {}).get("id"),
        }
        for t in teams
    ])


def fetch_schedule_for_date(game_date: date) -> pd.DataFrame:
    """Fetch all games for a specific date."""
    date_str = game_date.strftime("%Y-%m-%d")
    r = requests.get(
        f"{BASE_URL}/schedule",
        params={"date": date_str, "sportId": 1, "gameType": "R"},
    )
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game = {
                "game_pk": g["gamePk"],
                "game_date": g["gameDate"],
                "official_date": g.get("officialDate"),
                "game_type": g.get("gameType"),
                "status": g.get("status", {}).get("abstractGameState"),
                "away_team_id": g["teams"]["away"]["team"]["id"],
                "away_team_name": g["teams"]["away"]["team"]["name"],
                "home_team_id": g["teams"]["home"]["team"]["id"],
                "home_team_name": g["teams"]["home"]["team"]["name"],
                "away_score": g["teams"]["away"].get("score"),
                "home_score": g["teams"]["home"].get("score"),
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "scheduled_innings": g.get("scheduledInnings"),
            }
            games.append(game)

    return pd.DataFrame(games)


def fetch_season_games(year: int, force: bool = False) -> pd.DataFrame:
    """Fetch all games for a season. Cached per year."""
    out_path = RAW_DIR / "schedule" / f"schedule_{year}.parquet"
    if out_path.exists() and not force:
        logger.debug("cache hit season %d", year)
        return pd.read_parquet(out_path)

    logger.info("fetching schedule for %d", year)

    # Fetch games for each day of the season (April to October)
    all_games = []
    start_date = date(year, SEASON_START_MONTH, 1)
    end_date = date(year, SEASON_END_MONTH, 30)

    current_date = start_date
    today = today_in_schedule_timezone()
    while current_date <= end_date and current_date <= today:
        try:
            df = fetch_schedule_for_date(current_date)
            if not df.empty:
                all_games.append(df)
                logger.debug("fetched %d games for %s", len(df), current_date)
        except Exception as exc:
            logger.warning("failed %s: %s", current_date, exc)

        current_date += timedelta(days=1)

    if all_games:
        result = pd.concat(all_games, ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("saved %d games for %d", len(result), year)
        return result
    else:
        logger.warning("no games fetched for %d", year)
        return pd.DataFrame()


def fetch_pitching_year(year: int, force: bool = False) -> pd.DataFrame:
    """Fetch league-wide pitching stats for a season.

    Note: FanGraphs is currently returning 403 errors, so this may fail.
    """
    out_path = RAW_DIR / "pitching" / f"fangraphs_{year}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    try:
        from pybaseball import pitching_stats

        logger.info("fetching fangraphs pitching %d", year)
        df = pitching_stats(year, qual=0)  # qual=0 = include everyone
        df["season"] = year
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return df
    except Exception as exc:
        logger.warning("failed pitching %d: %s", year, exc)
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, help="First season (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Last season (inclusive)")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached"
    )
    parser.add_argument(
        "--skip-pitching", action="store_true", help="Skip league pitching pull"
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    # First, fetch and cache team info
    teams_path = RAW_DIR / "teams.parquet"
    if not teams_path.exists() or args.force:
        logger.info("fetching team info")
        teams = fetch_mlb_teams()
        teams.to_parquet(teams_path, index=False)
        logger.info("cached %d teams", len(teams))

    years = range(args.start, args.end + 1)
    for year in years:
        try:
            fetch_season_games(year, force=args.force)
        except Exception as exc:
            logger.warning("failed season %d: %s", year, exc)

        if not args.skip_pitching:
            try:
                fetch_pitching_year(year, force=args.force)
            except Exception as exc:
                logger.warning("failed pitching %d: %s", year, exc)

    logger.info("done. cache at %s", RAW_DIR)


if __name__ == "__main__":
    main()

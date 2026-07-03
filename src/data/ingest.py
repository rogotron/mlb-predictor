"""Historical data ingest.

Loads cached raw data from data/raw/ and creates processed game logs
in data/processed/. Idempotent: re-running with the same range is a no-op
unless --force is passed.

Primary source: MLB Stats API (via fetch_data.py).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.logging import configure_logging
from src.utils.paths import RAW_DIR, PROCESSED_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def load_schedule(raw_dir: Path, year: int) -> pd.DataFrame:
    """Load cached schedule data for a given year."""
    schedule_path = raw_dir / "schedule" / f"schedule_{year}.parquet"
    if not schedule_path.exists():
        logger.warning("no schedule data for %d", year)
        return pd.DataFrame()
    return pd.read_parquet(schedule_path)


def load_teams(raw_dir: Path) -> pd.DataFrame:
    """Load cached team data."""
    teams_path = raw_dir / "teams.parquet"
    if not teams_path.exists():
        logger.warning("no team data found")
        return pd.DataFrame()
    return pd.read_parquet(teams_path)


def process_games(schedule: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Process raw schedule into clean game logs with targets.

    Returns a DataFrame with columns:
        game_pk, game_date, official_date,
        home_team_id, away_team_id,
        home_score, away_score,
        venue_id, venue_name,
        target_home_win (1 if home wins, 0 otherwise),
        target_total_runs (home_score + away_score)
    """
    if schedule.empty:
        return pd.DataFrame()

    # Filter to completed games with usable scores. MLB's schedule feed can
    # leave postponed/rescheduled records marked Final with blank scores.
    df = schedule[
        (schedule["status"] == "Final")
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()

    if df.empty:
        logger.warning("no completed games found")
        return df

    # Parse game date
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
    df["official_date"] = pd.to_datetime(df["official_date"])

    # Create target columns
    df["target_home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["target_total_runs"] = df["home_score"] + df["away_score"]

    # Select and order columns
    cols = [
        "game_pk",
        "game_date",
        "official_date",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "venue_id",
        "venue_name",
        "target_home_win",
        "target_total_runs",
    ]
    df = df[cols].copy()

    # Sort by date
    df = df.sort_values("game_date").reset_index(drop=True)

    return df


def build_team_lookup(teams: pd.DataFrame) -> pd.DataFrame:
    """Build team ID to info mapping."""
    if teams.empty:
        return pd.DataFrame()
    return teams[["team_id", "team_name", "abbreviation"]].copy()


def ingest_year(year: int, raw_dir: Path, processed_dir: Path, force: bool = False) -> pd.DataFrame:
    """Ingest a single year's data."""
    out_path = processed_dir / "games" / f"games_{year}.parquet"
    if out_path.exists() and not force:
        logger.debug("cache hit %d", year)
        return pd.read_parquet(out_path)

    logger.info("processing %d", year)

    # Load raw data
    schedule = load_schedule(raw_dir, year)
    teams = load_teams(raw_dir)

    if schedule.empty:
        logger.warning("no schedule data for %d", year)
        return pd.DataFrame()

    # Process games
    games = process_games(schedule, teams)

    if not games.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        games.to_parquet(out_path, index=False)
        logger.info("saved %d games for %d", len(games), year)

    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, help="First season (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Last season (inclusive)")
    parser.add_argument(
        "--force", action="store_true", help="Re-process even if cached"
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    years = range(args.start, args.end + 1)
    for year in years:
        try:
            ingest_year(year, RAW_DIR, PROCESSED_DIR, force=args.force)
        except Exception as exc:
            logger.warning("failed %d: %s", year, exc)

    logger.info("done. processed data at %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()

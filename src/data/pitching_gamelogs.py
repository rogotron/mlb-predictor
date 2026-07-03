"""Per-start pitching stats from the MLB Stats API boxscore endpoint.

Uses the same host as the schedule fetcher (statsapi.mlb.com) — no
reliability issues with Baseball Savant.

Cache layout:
    data/raw/pitching_gamelogs/{year}.parquet

Schema: game_pk, game_date, pitcher_id, side (home/away),
        ip, h, er, bb, k, hr, bf
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import time

import pandas as pd
import requests

from src.utils.paths import RAW_DIR

# Polite delay between boxscore requests to avoid triggering rate limits.
_REQUEST_DELAY = 0.15  # seconds

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Stats computed per start then rolled — mirrors _SP_STAT_COLS in statcast.py
GAMELOG_STAT_COLS = [
    "era",
    "whip",
    "k_per_9",
    "bb_per_9",
    "k_minus_bb_pct",
    "hr_per_9",
    "ip_per_start",
]

# Season-to-date stats derived from cumulative counting stats (more stable
# than per-start rates, grows in sample size through the season).
STD_STAT_COLS = [
    "era_std",
    "whip_std",
    "k_per_9_std",
    "bb_per_9_std",
    "k_minus_bb_pct_std",
    "hr_per_9_std",
    "ip_total_std",  # total IP so far this season (sample size proxy)
]


def _parse_ip(ip_str) -> float:
    """Convert '6.2' (6 and 2/3 innings) to decimal 6.667."""
    try:
        parts = str(ip_str).split(".")
        full = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return full + thirds / 3.0
    except (ValueError, AttributeError):
        return 0.0


def _extract_sp_line(
    team_data: dict, game_pk: int, game_date: str, side: str
) -> dict | None:
    """Pull the starting pitcher's line from one side of a boxscore."""
    pitchers = team_data.get("pitchers", [])
    if not pitchers:
        return None

    sp_id = pitchers[0]
    player = team_data.get("players", {}).get(f"ID{sp_id}", {})
    stats = player.get("stats", {}).get("pitching", {})
    if not stats:
        return None

    ip = _parse_ip(stats.get("inningsPitched", "0.0"))
    if ip == 0.0:
        return None

    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "pitcher_id": sp_id,
        "side": side,
        "ip": ip,
        "h": int(stats.get("hits", 0) or 0),
        "er": int(stats.get("earnedRuns", 0) or 0),
        "bb": int(stats.get("baseOnBalls", 0) or 0),
        "k": int(stats.get("strikeOuts", 0) or 0),
        "hr": int(stats.get("homeRunsAllowed", 0) or 0),
        "bf": int(stats.get("battersFaced", 0) or 0),
    }


def fetch_boxscore_pitching(game_pk: int, game_date: str = "") -> list[dict]:
    """Return [home_sp_row, away_sp_row] (or fewer) for one game.

    game_date should be passed in ISO format (YYYY-MM-DD) from the schedule
    cache — the boxscore endpoint does not include date information.
    """
    try:
        r = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.debug("boxscore failed game_pk=%d: %s", game_pk, exc)
        return []

    teams = data.get("teams", {})

    rows = []
    for side in ("home", "away"):
        row = _extract_sp_line(teams.get(side, {}), game_pk, game_date, side)
        if row:
            rows.append(row)
    return rows


def fetch_season_pitching_logs(
    year: int,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch all SP lines for one season, cached to parquet.

    Loads the existing schedule cache to get game_pks, then fetches each
    boxscore. Already-fetched game_pks are skipped unless force=True.
    """
    out_path = raw_dir / "pitching_gamelogs" / f"{year}.parquet"
    schedule_path = raw_dir / "schedule" / f"schedule_{year}.parquet"

    if not schedule_path.exists():
        logger.warning("no schedule cache for %d; run fetch_data.py first", year)
        return pd.DataFrame()

    schedule = pd.read_parquet(schedule_path)
    completed = schedule[schedule["status"] == "Final"]
    all_pks = set(completed["game_pk"].dropna().astype(int))

    existing = pd.DataFrame()
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path)
        done_pks = set(existing["game_pk"].unique())
        to_fetch = sorted(all_pks - done_pks)
        logger.info(
            "%d: %d completed games, %d already cached, %d to fetch",
            year, len(all_pks), len(done_pks), len(to_fetch),
        )
    else:
        to_fetch = sorted(all_pks)
        logger.info("%d: fetching %d games", year, len(to_fetch))

    if not to_fetch:
        return existing

    # Build a game_pk -> official_date lookup from the schedule cache
    date_map: dict[int, str] = {}
    if "official_date" in completed.columns:
        date_map = dict(zip(completed["game_pk"].astype(int), completed["official_date"].astype(str)))

    new_rows: list[dict] = []
    for i, game_pk in enumerate(to_fetch):
        gdate = date_map.get(int(game_pk), "")
        new_rows.extend(fetch_boxscore_pitching(game_pk, game_date=gdate))
        time.sleep(_REQUEST_DELAY)
        if (i + 1) % 200 == 0:
            logger.info("  %d / %d games fetched", i + 1, len(to_fetch))

    if not new_rows:
        logger.warning("%d: no new pitching lines fetched", year)
        return existing

    new_df = pd.DataFrame(new_rows)
    new_df["game_date"] = pd.to_datetime(new_df["game_date"])

    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        if not existing.empty
        else new_df
    )
    combined = (
        combined.drop_duplicates(subset=["game_pk", "side"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info("%d: saved %d SP lines -> %s", year, len(combined), out_path)
    return combined


def load_pitching_gamelogs(
    start_year: int,
    end_year: int,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load cached pitching gamelogs for a year range."""
    chunks = []
    for year in range(start_year, end_year + 1):
        path = raw_dir / "pitching_gamelogs" / f"{year}.parquet"
        if path.exists():
            chunks.append(pd.read_parquet(path))
        else:
            logger.debug("no pitching gamelogs for %d (run fetch_pitching_gamelogs.py)", year)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Back-fill game_date from the schedule cache for rows where the boxscore
    # fetch stored NaT (the boxscore endpoint doesn't include the game date).
    if df["game_date"].isna().any():
        sch_chunks = []
        for year in range(start_year, end_year + 1):
            sch_path = raw_dir / "schedule" / f"schedule_{year}.parquet"
            if sch_path.exists():
                sch = pd.read_parquet(sch_path, columns=["game_pk", "official_date"])
                sch_chunks.append(sch)
        if sch_chunks:
            schedule = (
                pd.concat(sch_chunks, ignore_index=True)
                .drop_duplicates("game_pk")
                .assign(
                    game_pk=lambda d: d["game_pk"].astype(int),
                    _sched_date=lambda d: pd.to_datetime(d["official_date"]),
                )
            )
            df["game_pk"] = df["game_pk"].astype(int)
            df = df.merge(schedule[["game_pk", "_sched_date"]], on="game_pk", how="left")
            mask = df["game_date"].isna()
            df.loc[mask, "game_date"] = df.loc[mask, "_sched_date"]
            df = df.drop(columns=["_sched_date"])
            logger.info("back-filled %d game_dates from schedule cache", mask.sum())

    return df


def _add_per_start_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate stats from counting stats for each start."""
    df = df.copy()
    ip = df["ip"].clip(lower=0.01)
    bf = df["bf"].clip(lower=1)

    df["era"] = df["er"] * 9.0 / ip
    df["whip"] = (df["h"] + df["bb"]) / ip
    df["k_per_9"] = df["k"] * 9.0 / ip
    df["bb_per_9"] = df["bb"] * 9.0 / ip
    df["hr_per_9"] = df["hr"] * 9.0 / ip
    df["k_minus_bb_pct"] = (df["k"] - df["bb"]) / bf
    df["ip_per_start"] = df["ip"]
    return df


def compute_starter_season_to_date(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date cumulative pitching stats per pitcher, shift(1) to prevent leakage.

    Rates are computed from cumulative counting stats (not averaged per-start
    rates), giving more stable estimates as the season progresses.

    Returns gamelogs with extra columns for each stat in STD_STAT_COLS.
    """
    if gamelogs.empty:
        return gamelogs

    df = gamelogs.copy()
    df["_year"] = pd.to_datetime(df["game_date"]).dt.year
    df = df.sort_values(["pitcher_id", "_year", "game_date"]).reset_index(drop=True)

    grouped = df.groupby(["pitcher_id", "_year"], group_keys=False)

    # Cumulative sums of counting stats, shifted so current game is excluded
    for col in ["er", "ip", "h", "bb", "k", "hr", "bf"]:
        df[f"_cum_{col}"] = grouped[col].transform(
            lambda x: x.shift(1).expanding().sum()
        )

    cum_ip = df["_cum_ip"].clip(lower=0.01)
    cum_bf = df["_cum_bf"].clip(lower=1)

    df["era_std"] = df["_cum_er"] * 9.0 / cum_ip
    df["whip_std"] = (df["_cum_h"] + df["_cum_bb"]) / cum_ip
    df["k_per_9_std"] = df["_cum_k"] * 9.0 / cum_ip
    df["bb_per_9_std"] = df["_cum_bb"] * 9.0 / cum_ip
    df["hr_per_9_std"] = df["_cum_hr"] * 9.0 / cum_ip
    df["k_minus_bb_pct_std"] = (df["_cum_k"] - df["_cum_bb"]) / cum_bf
    df["ip_total_std"] = df["_cum_ip"]

    df = df.drop(columns=[f"_cum_{c}" for c in ["er", "ip", "h", "bb", "k", "hr", "bf"]] + ["_year"])
    return df


def compute_starter_rolling_features(
    gamelogs: pd.DataFrame,
    n_starts: int = 3,
) -> pd.DataFrame:
    """Rolling n-start averages per pitcher, shift(1) to prevent leakage.

    Returns gamelogs with extra columns {stat}_l{n_starts} for each stat
    in GAMELOG_STAT_COLS.
    """
    if gamelogs.empty:
        return gamelogs

    df = _add_per_start_rates(gamelogs)
    df = df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    grouped = df.groupby("pitcher_id", group_keys=False)

    for col in GAMELOG_STAT_COLS:
        if col not in df.columns:
            df[f"{col}_l{n_starts}"] = float("nan")
            continue
        df[f"{col}_l{n_starts}"] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=n_starts, min_periods=1).mean()
        )

    return df

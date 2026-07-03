"""Fetch and cache park factors from FanGraphs.

Cache: data/raw/park_factors/park_factors_{year}.parquet

Output schema (one row per venue):
    venue_id  (int)     — MLB MLBAM venue ID
    team_id   (int)     — home team's MLB ID
    pf_runs   (float)   — runs park factor (100 = neutral)
    pf_hr     (float)   — home-run park factor

Falls back to computing a basic runs park factor from the processed game
logs when FanGraphs is unreachable (e.g. 403).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

_FG_API = (
    "https://www.fangraphs.com/api/park-factors"
    "?season={year}&type=runs&condition=all&role=all&teamid=0"
)
_FG_HR_API = (
    "https://www.fangraphs.com/api/park-factors"
    "?season={year}&type=hr&condition=all&role=all&teamid=0"
)

# FanGraphs team abbreviations that differ from the MLB abbreviations stored
# in data/raw/teams.parquet.  Only mismatches are listed here.
_FG_TO_MLB_ABBREV: dict[str, str] = {
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
    "LAA": "LAA",  # same, listed for clarity
}


def _fetch_fg_pf(year: int, pf_type: str = "runs") -> dict[str, float]:
    """Return {fg_abbrev: park_factor} from FanGraphs API, or {} on failure."""
    url = _FG_API if pf_type == "runs" else _FG_HR_API
    try:
        r = requests.get(
            url.format(year=year),
            headers={"User-Agent": "mlb-predictor/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        # FanGraphs returns {"data": [{"Team": "NYY", "Basic_pf": 103, ...}, ...]}
        result: dict[str, float] = {}
        for row in data.get("data", []):
            team = row.get("Team") or row.get("team", "")
            pf = row.get("Basic_pf") or row.get("basic_pf")
            if team and pf is not None:
                result[str(team)] = float(pf)
        return result
    except Exception as exc:
        logger.warning(
            "FanGraphs park factors unavailable (API returned non-JSON, likely a bot "
            "block or redirect). Will fall back to game-log computation. Detail: %s", exc
        )
        return {}


def _compute_from_game_logs(year: int, processed_dir: Path) -> pd.DataFrame:
    """Derive a basic runs park factor from the processed game log.

    PF = (runs_per_game_at_venue / league_avg_runs_per_game) * 100
    Keyed by (venue_id, team_id).
    """
    from src.features.build import load_processed_games  # avoid circular at module level

    games = load_processed_games(processed_dir, max(2018, year - 2), year)
    if games.empty:
        return pd.DataFrame()

    games = games.dropna(subset=["home_score", "away_score", "venue_id"])
    games["total_runs"] = games["home_score"] + games["away_score"]

    league_avg = games["total_runs"].mean()
    if league_avg == 0:
        return pd.DataFrame()

    venue_avg = (
        games.groupby(["venue_id", "home_team_id"])["total_runs"]
        .mean()
        .reset_index()
        .rename(columns={"home_team_id": "team_id", "total_runs": "avg_runs"})
    )
    venue_avg["pf_runs"] = (venue_avg["avg_runs"] / league_avg * 100).round(1)
    venue_avg["pf_hr"] = float("nan")
    venue_avg["venue_id"] = venue_avg["venue_id"].astype("Int64")
    venue_avg["team_id"] = venue_avg["team_id"].astype("Int64")
    logger.info("computed park factors from game logs for %d venues", len(venue_avg))
    return venue_avg[["venue_id", "team_id", "pf_runs", "pf_hr"]]


def fetch_park_factors(
    year: int,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Return park factors keyed by venue_id for the given season.

    Tries FanGraphs first; falls back to a game-log computation if the API
    is unavailable.  Results are cached to data/raw/park_factors/.
    """
    cache_path = raw_dir / "park_factors" / f"park_factors_{year}.parquet"
    if cache_path.exists() and not force:
        logger.debug("park factor cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    # Load team info so we can map abbreviations → team_id → venue_id
    teams_path = raw_dir / "teams.parquet"
    if not teams_path.exists():
        logger.warning("teams.parquet not found; skipping FanGraphs lookup")
        teams = pd.DataFrame(columns=["team_id", "team_name", "abbreviation"])
    else:
        teams = pd.read_parquet(teams_path)[["team_id", "team_name", "abbreviation"]]

    # Build venue → team map from the processed game log
    from src.features.build import load_processed_games
    games = load_processed_games(processed_dir, year, year)
    if games.empty:
        games = load_processed_games(processed_dir, year - 1, year - 1)

    if not games.empty and "venue_id" in games.columns:
        venue_team = (
            games.groupby("home_team_id")["venue_id"]
            .agg(lambda x: x.mode()[0])
            .reset_index()
            .rename(columns={"home_team_id": "team_id"})
        )
    else:
        venue_team = pd.DataFrame(columns=["team_id", "venue_id"])

    pf_runs = _fetch_fg_pf(year, "runs")
    pf_hr = _fetch_fg_pf(year, "hr")

    if pf_runs:
        rows = []
        for fg_abbrev, pf in pf_runs.items():
            mlb_abbrev = _FG_TO_MLB_ABBREV.get(fg_abbrev, fg_abbrev)
            team_row = teams[teams["abbreviation"] == mlb_abbrev]
            if team_row.empty:
                continue
            team_id = int(team_row["team_id"].iloc[0])
            vt = venue_team[venue_team["team_id"] == team_id]
            venue_id = int(vt["venue_id"].iloc[0]) if not vt.empty else None
            rows.append({
                "venue_id": venue_id,
                "team_id": team_id,
                "pf_runs": pf,
                "pf_hr": float(pf_hr.get(fg_abbrev, float("nan"))),
            })
        df = pd.DataFrame(rows)
        logger.info("fetched FanGraphs park factors for %d teams", len(df))
    else:
        logger.info("FanGraphs unavailable; computing park factors from game logs")
        df = _compute_from_game_logs(year, processed_dir)

    if df.empty:
        return df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df

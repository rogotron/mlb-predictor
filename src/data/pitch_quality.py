"""Fetch and cache Baseball Savant pitch-arsenal quality metrics.

Source:
    https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats

The endpoint returns one row per pitcher/pitch type. We aggregate to pitcher
level by weighting empirical run value per 100 pitches, xwOBA, and whiff rate
by pitch usage. Run value per 100 is used as a public Stuff+ substitute now
that FanGraphs leaderboards block scraper access.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from src.data.park_factors import _FG_TO_MLB_ABBREV
from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

QUALITY_COLS = ["rv_per_100", "xwoba_arsenal", "whiff_arsenal"]
_CACHE_TTL = timedelta(hours=24)
_SAVANT_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mlb-predictor/1.0)"}


def _cache_path(raw_dir: Path, season: int, as_of: date | None) -> Path:
    suffix = as_of.isoformat() if as_of else "latest"
    return raw_dir / "pitch_quality" / f"pitch_arsenal_{season}_{suffix}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - modified < _CACHE_TTL


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalised = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for alias in aliases:
        key = alias.lower().replace(" ", "").replace("_", "")
        if key in normalised:
            return normalised[key]
    return None


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = vals.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((vals[mask] * w[mask]).sum() / w[mask].sum())


def normalize_pitch_quality(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Savant pitch-type rows to one pitcher-level row."""
    if raw.empty:
        return pd.DataFrame(columns=["player_id", "team", "pitches"] + QUALITY_COLS)

    player_col = _find_col(raw, ["player_id", "pitcher", "mlbam"])
    team_col = _find_col(raw, ["team_name_alt", "team", "team_name"])
    pitches_col = _find_col(raw, ["pitches", "pitch_count"])
    usage_col = _find_col(raw, ["pitch_usage", "usage"])
    rv_col = _find_col(raw, ["run_value_per_100", "rv_per_100"])
    xwoba_col = _find_col(raw, ["est_woba", "xwoba", "woba"])
    whiff_col = _find_col(raw, ["whiff_percent", "whiff_rate"])

    if player_col is None:
        return pd.DataFrame(columns=["player_id", "team", "pitches"] + QUALITY_COLS)

    df = pd.DataFrame()
    df["player_id"] = pd.to_numeric(raw[player_col], errors="coerce")
    df["team"] = raw[team_col].astype(str) if team_col else pd.NA
    df["pitches"] = pd.to_numeric(raw[pitches_col], errors="coerce") if pitches_col else 0
    if usage_col:
        df["_weight"] = pd.to_numeric(raw[usage_col], errors="coerce")
    else:
        df["_weight"] = df["pitches"]
    df["rv_per_100"] = pd.to_numeric(raw[rv_col], errors="coerce") if rv_col else pd.NA
    df["xwoba_arsenal"] = pd.to_numeric(raw[xwoba_col], errors="coerce") if xwoba_col else pd.NA
    df["whiff_arsenal"] = pd.to_numeric(raw[whiff_col], errors="coerce") if whiff_col else pd.NA
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)

    rows = []
    for player_id, grp in df.groupby("player_id"):
        row = {
            "player_id": int(player_id),
            "team": grp["team"].dropna().mode().iloc[0] if grp["team"].notna().any() else None,
            "pitches": float(grp["pitches"].sum()),
        }
        for col in QUALITY_COLS:
            row[col] = _weighted_mean(grp[col], grp["_weight"])
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_pitch_quality(
    season: int,
    raw_dir: Path = RAW_DIR,
    as_of: date | None = None,
    force: bool = False,
    min_pitches: int = 50,
) -> pd.DataFrame:
    """Fetch Baseball Savant pitch-arsenal metrics for a season."""
    path = _cache_path(raw_dir, season, as_of)
    current_latest = as_of is None and season == date.today().year
    if path.exists() and not force and (not current_latest or _is_fresh(path)):
        return pd.read_parquet(path)

    logger.info("fetching Savant pitch arsenal season=%s min=%s", season, min_pitches)
    try:
        r = requests.get(
            _SAVANT_URL,
            params={"type": "pitcher", "season": season, "min": min_pitches, "csv": "true"},
            headers=_HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text.lstrip("\ufeff")))
    except Exception as exc:
        logger.warning("Savant pitch arsenal unavailable: %s", exc)
        return pd.DataFrame(columns=["player_id", "team", "pitches"] + QUALITY_COLS)

    df = normalize_pitch_quality(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def team_abbrev_to_id(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Map Baseball Savant/FanGraphs team abbreviations to MLBAM team IDs."""
    teams_path = raw_dir / "teams.parquet"
    if not teams_path.exists():
        return {}
    teams = pd.read_parquet(teams_path)
    out: dict[str, int] = {}
    for _, row in teams.iterrows():
        mlb = str(row.get("abbreviation", ""))
        fg = next((k for k, v in _FG_TO_MLB_ABBREV.items() if v == mlb), mlb)
        out[fg] = int(row["team_id"])
        out[mlb] = int(row["team_id"])
    return out


def bullpen_pitch_quality_by_team(
    quality: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    top_n: int = 5,
) -> pd.DataFrame:
    """Top-N bullpen arsenal averages by team, weighted by pitch count."""
    if quality.empty or "team" not in quality.columns:
        return pd.DataFrame()
    team_map = team_abbrev_to_id(raw_dir)
    df = quality.copy()
    df["team_id"] = df["team"].map(team_map)
    df = df.dropna(subset=["team_id"])
    df["team_id"] = df["team_id"].astype(int)
    df["pitches"] = pd.to_numeric(df["pitches"], errors="coerce").fillna(0)
    df = df.sort_values(["team_id", "pitches"], ascending=[True, False])

    rows = []
    for team_id, grp in df.groupby("team_id"):
        top = grp.head(top_n)
        weights = top["pitches"].clip(lower=0)
        row = {"team_id": int(team_id)}
        for col in QUALITY_COLS:
            row[f"bp_{col}_weighted"] = _weighted_mean(top[col], weights)
        rows.append(row)
    return pd.DataFrame(rows)

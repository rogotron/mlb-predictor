"""Fetch and cache Statcast pitch data from Baseball Savant via pybaseball.

Cache layout (one file per calendar month so incremental refreshes only
re-fetch the current incomplete month):

    data/raw/statcast/pitches_{year}_{month:02d}.parquet

Aggregated tables derived from the pitch cache:

    pitcher_starts  — one row per (pitcher, game_pk): xwOBA against, whiff
                      rate, barrel rate, and platoon splits.
    batter_pa       — one row per (batter, pitcher): PA count and mean xwOBA,
                      used for BvP lookups.
    batter_season   — one row per (batter, year): season-level Statcast rates
                      by handedness split, used for lineup aggregation.

Note on barrel rate: pybaseball does not always return a dedicated `barrel`
column.  We fall back to `launch_speed_angle == 6` (Baseball Savant's barrel
zone classification) when `barrel` is absent.  If neither column is present,
barrel_rate is NaN.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pybaseball

from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)
_MONTH_CACHE: dict[tuple[int, int, str], pd.DataFrame] = {}

# Columns we keep from the full ~90-col Statcast feed to limit parquet size.
# `inning` and `inning_topbot` are needed to identify which pitcher is the
# home vs away starter. `launch_speed_angle` is the barrel-zone fallback.
_KEEP_COLS = [
    "game_pk",
    "game_date",
    "inning",
    "inning_topbot",
    "pitcher",
    "batter",
    "p_throws",
    "stand",
    "events",
    "description",
    "type",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "barrel",
    "launch_speed_angle",
    "launch_speed",
    "bb_type",
    "at_bat_number",
]

_SWING_DESCS = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
})
_WHIFF_DESCS = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
})

# Baseball Savant launch_speed_angle categories; 6 = barrel.
_BARREL_ZONE = 6


# ---------------------------------------------------------------------------
# Raw fetch / cache helpers
# ---------------------------------------------------------------------------

def _month_path(raw_dir: Path, year: int, month: int) -> Path:
    return raw_dir / "statcast" / f"pitches_{year}_{month:02d}.parquet"


def _is_month_complete(year: int, month: int) -> bool:
    today = date.today()
    if year < today.year:
        return True
    return year == today.year and month < today.month


def fetch_statcast_month(
    year: int,
    month: int,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch one calendar month of Statcast pitch data and cache it.

    Complete (past) months are fetched once and never re-fetched unless
    force=True.  The current month is always re-fetched to stay current.
    """
    import calendar as _cal

    path = _month_path(raw_dir, year, month)
    complete = _is_month_complete(year, month)
    cache_key = (year, month, str(raw_dir.resolve()))

    if not force and cache_key in _MONTH_CACHE:
        return _MONTH_CACHE[cache_key].copy()

    if path.exists() and complete and not force:
        logger.debug("statcast cache hit: %s", path)
        df = pd.read_parquet(path)
        _MONTH_CACHE[cache_key] = df
        return df.copy()

    last_day = _cal.monthrange(year, month)[1]
    start = date(year, month, 1)
    end = min(date(year, month, last_day), date.today())

    if start > date.today():
        return pd.DataFrame()

    logger.info("fetching statcast %s -> %s", start, end)
    pybaseball.cache.enable()
    try:
        df = pybaseball.statcast(
            start_dt=start.isoformat(),
            end_dt=end.isoformat(),
            verbose=False,
        )
    except Exception as exc:
        logger.warning("statcast fetch failed for %d-%02d: %s", year, month, exc)
        return pd.DataFrame()

    # Cache even empty results so re-runs skip the API call for off-season months
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        pd.DataFrame(columns=_KEEP_COLS).to_parquet(path, index=False)
        logger.debug("cached empty parquet for %d-%02d (off-season)", year, month)
        _MONTH_CACHE[cache_key] = df
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    available = [c for c in _KEEP_COLS if c in df.columns]
    df = df[available].copy()

    df.to_parquet(path, index=False)
    logger.info("cached %d pitches -> %s", len(df), path)
    _MONTH_CACHE[cache_key] = df
    return df.copy()


def load_statcast(
    start_date: date,
    end_date: date,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Load (and fetch if missing) Statcast data for a date range.

    Iterates over calendar months that overlap [start_date, end_date].
    """
    chunks: list[pd.DataFrame] = []
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        df = fetch_statcast_month(cur.year, cur.month, raw_dir=raw_dir, force=force)
        if not df.empty:
            mask = (df["game_date"] >= pd.Timestamp(start_date)) & (
                df["game_date"] <= pd.Timestamp(end_date)
            )
            chunks.append(df.loc[mask])
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    if not chunks:
        return pd.DataFrame(columns=_KEEP_COLS)

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-pitch helpers
# ---------------------------------------------------------------------------

def _is_pa(pitches: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows that are the final pitch of a plate appearance."""
    return pitches["woba_denom"].fillna(0).astype(float) == 1


def _whiff_rate(group: pd.DataFrame) -> float:
    if "description" not in group.columns:
        return float("nan")
    swings = group["description"].isin(_SWING_DESCS).sum()
    whiffs = group["description"].isin(_WHIFF_DESCS).sum()
    return float(whiffs / swings) if swings > 0 else float("nan")


def _barrel_rate(group: pd.DataFrame) -> float:
    """Barrel rate with fallback: barrel col -> launch_speed_angle==6 -> NaN."""
    if "type" not in group.columns:
        return float("nan")
    bip = group["type"].eq("X").sum()
    if bip == 0:
        return float("nan")

    if "barrel" in group.columns:
        barrels = group["barrel"].fillna(0).astype(float).sum()
    elif "launch_speed_angle" in group.columns:
        barrels = (group["launch_speed_angle"] == _BARREL_ZONE).sum()
    else:
        return float("nan")

    return float(barrels / bip)


def _mean_xwoba(group: pd.DataFrame) -> float:
    if "estimated_woba_using_speedangle" not in group.columns:
        return float("nan")
    vals = group.loc[_is_pa(group), "estimated_woba_using_speedangle"].dropna()
    return float(vals.mean()) if len(vals) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-start aggregation
# ---------------------------------------------------------------------------

def aggregate_pitcher_starts(pitches: pd.DataFrame) -> pd.DataFrame:
    """Summarise Statcast pitches into one row per (pitcher, game_pk).

    Output columns:
        pitcher, game_pk, game_date, p_throws,
        xwoba_against, whiff_rate, barrel_rate,
        xwoba_against_vs_L, xwoba_against_vs_R,
        whiff_rate_vs_L, whiff_rate_vs_R,
        pa_count
    """
    if pitches.empty:
        return pd.DataFrame()

    rows = []
    for (pitcher_id, game_pk), grp in pitches.groupby(["pitcher", "game_pk"]):
        row: dict = {
            "pitcher": int(pitcher_id),
            "game_pk": int(game_pk),
            "game_date": grp["game_date"].iloc[0],
            "p_throws": grp["p_throws"].mode()[0] if "p_throws" in grp.columns else None,
            "pa_count": int(_is_pa(grp).sum()),
            "xwoba_against": _mean_xwoba(grp),
            "whiff_rate": _whiff_rate(grp),
            "barrel_rate": _barrel_rate(grp),
        }
        for hand in ("L", "R"):
            sub = grp[grp["stand"] == hand] if "stand" in grp.columns else grp.iloc[0:0]
            row[f"xwoba_against_vs_{hand}"] = _mean_xwoba(sub)
            row[f"whiff_rate_vs_{hand}"] = _whiff_rate(sub)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_game_starters(pitches: pd.DataFrame) -> pd.DataFrame:
    """Identify home and away starting pitcher for each game_pk.

    The starter is the pitcher who appears in the first inning on each side.
    Top of 1st = home team pitcher facing away batters.
    Bot of 1st = away team pitcher facing home batters.

    Requires `inning` and `inning_topbot` columns (added to _KEEP_COLS).
    Existing cached files that pre-date this change won't have them; re-fetch
    with --force to regenerate.

    Returns DataFrame with: game_pk, home_sp_id, away_sp_id.
    """
    if pitches.empty:
        return pd.DataFrame(columns=["game_pk", "home_sp_id", "away_sp_id"])

    if "inning" not in pitches.columns or "inning_topbot" not in pitches.columns:
        logger.warning(
            "inning/inning_topbot columns missing from Statcast cache; "
            "re-fetch with --force to enable starter identification"
        )
        return pd.DataFrame(columns=["game_pk", "home_sp_id", "away_sp_id"])

    first = pitches[pitches["inning"] == 1]
    rows = []
    for game_pk, grp in first.groupby("game_pk"):
        top = grp[grp["inning_topbot"] == "Top"]["pitcher"]
        bot = grp[grp["inning_topbot"] == "Bot"]["pitcher"]
        rows.append({
            "game_pk": int(game_pk),
            "home_sp_id": int(top.mode()[0]) if not top.empty else None,
            "away_sp_id": int(bot.mode()[0]) if not bot.empty else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vectorised rolling pitcher features (for training set joins)
# ---------------------------------------------------------------------------

_SP_STAT_COLS = [
    "xwoba_against",
    "whiff_rate",
    "barrel_rate",
    "xwoba_against_vs_L",
    "xwoba_against_vs_R",
    "whiff_rate_vs_L",
    "whiff_rate_vs_R",
]


def compute_pitcher_rolling_features(
    pitcher_starts: pd.DataFrame,
    n_starts: int = 3,
) -> pd.DataFrame:
    """Pre-compute rolling n-start averages for every pitcher start.

    Uses shift(1) so a start's rolling stats only see previous starts
    (no same-game leakage).  Suitable for a merge_asof join against the
    training game log.

    Returns pitcher_starts with additional columns:
        {stat}_l{n_starts}  for each stat in _SP_STAT_COLS
        sp_starts_available  (int) — how many prior starts were in the window
    """
    if pitcher_starts.empty:
        return pitcher_starts

    n = n_starts
    df = pitcher_starts.sort_values(["pitcher", "game_date"]).copy()

    grouped = df.groupby("pitcher", group_keys=False)

    for col in _SP_STAT_COLS:
        if col not in df.columns:
            df[f"{col}_l{n}"] = float("nan")
            continue
        df[f"{col}_l{n}"] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
        )

    # Count how many prior starts were actually available in the window.
    ref_col = next((c for c in _SP_STAT_COLS if c in df.columns), None)
    if ref_col:
        df["sp_starts_available"] = grouped[ref_col].transform(
            lambda x: x.shift(1).rolling(window=n, min_periods=1).count()
        ).fillna(0).astype(int)
    else:
        df["sp_starts_available"] = 0

    return df


# ---------------------------------------------------------------------------
# Batter aggregations
# ---------------------------------------------------------------------------

def aggregate_batter_season(pitches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Season-level Statcast rates per batter.

    Output columns:
        batter, year, stand,
        xwoba, barrel_rate,
        xwoba_vs_R, xwoba_vs_L,
        barrel_rate_vs_R, barrel_rate_vs_L,
        pa_count
    """
    if pitches.empty:
        return pd.DataFrame()

    rows = []
    for batter_id, grp in pitches.groupby("batter"):
        row: dict = {
            "batter": int(batter_id),
            "year": year,
            "stand": grp["stand"].mode()[0] if "stand" in grp.columns else None,
            "pa_count": int(_is_pa(grp).sum()),
            "xwoba": _mean_xwoba(grp),
            "barrel_rate": _barrel_rate(grp),
        }
        for hand in ("R", "L"):
            sub = grp[grp["p_throws"] == hand] if "p_throws" in grp.columns else grp.iloc[0:0]
            row[f"xwoba_vs_{hand}"] = _mean_xwoba(sub)
            row[f"barrel_rate_vs_{hand}"] = _barrel_rate(sub)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_bvp(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (batter, pitcher) career PA count and mean xwOBA from Statcast.

    Output columns: batter, pitcher, pa_count, xwoba_bvp

    Statcast goes back to 2015, covering active-player BvP history.
    Filter to PA >= 20 and apply min(PA, 60)/60 weight when consuming
    (see src/features/bvp.py).
    """
    if pitches.empty:
        return pd.DataFrame(columns=["batter", "pitcher", "pa_count", "xwoba_bvp"])

    pa = pitches.loc[_is_pa(pitches), ["batter", "pitcher", "estimated_woba_using_speedangle"]].copy()
    pa["batter"] = pd.to_numeric(pa["batter"], errors="coerce")
    pa["pitcher"] = pd.to_numeric(pa["pitcher"], errors="coerce")
    pa = pa.dropna(subset=["batter", "pitcher"])

    result = (
        pa.groupby(["batter", "pitcher"], as_index=False)
        .agg(pa_count=("batter", "count"), xwoba_bvp=("estimated_woba_using_speedangle", "mean"))
    )
    result["batter"] = result["batter"].astype(int)
    result["pitcher"] = result["pitcher"].astype(int)
    return result


# ---------------------------------------------------------------------------
# Team batting aggregations (for rolling team Statcast features)
# ---------------------------------------------------------------------------

# Stats computed per (game, batting side) then rolled across games.
_TEAM_HIT_COLS = ["xwoba_off", "barrel_rate_off"]

_BULLPEN_STAT_COLS = [
    "bullpen_xwoba_against",
    "bullpen_whiff_rate",
    "bullpen_barrel_rate",
]


def aggregate_team_game_hitting(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (game_pk, side) batting Statcast stats.

    'home' = Bot innings (home team at bat against away pitchers).
    'away' = Top innings (away team at bat against home pitchers).

    Uses woba_denom==1 to identify plate-appearance-ending pitches for xwOBA,
    and type=='X' (ball in play) with launch_speed_angle==6 for barrel rate
    (same barrel fallback as _barrel_rate()).

    Returns DataFrame with: game_pk, game_date, side, xwoba_off, barrel_rate_off.
    """
    if pitches.empty or "inning_topbot" not in pitches.columns:
        return pd.DataFrame(columns=["game_pk", "game_date", "side"] + _TEAM_HIT_COLS)

    pitches = pitches.copy()
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    # xwOBA: mean over PA-ending pitches
    pa = pitches[pitches["woba_denom"].fillna(0).astype(float) == 1].copy()
    xwoba_agg = (
        pa.groupby(["game_pk", "inning_topbot"])["estimated_woba_using_speedangle"]
        .mean()
        .reset_index()
        .rename(columns={"estimated_woba_using_speedangle": "xwoba_off"})
    )

    # Barrel rate: barrels / BIP, using launch_speed_angle fallback
    bip = pitches[pitches["type"].eq("X")].copy() if "type" in pitches.columns else pd.DataFrame()
    if not bip.empty:
        if "barrel" in bip.columns:
            bip["_barrel"] = bip["barrel"].fillna(0).astype(float).clip(upper=1)
        elif "launch_speed_angle" in bip.columns:
            bip["_barrel"] = (bip["launch_speed_angle"] == _BARREL_ZONE).astype(float)
        else:
            bip["_barrel"] = float("nan")

        barrel_agg = (
            bip.groupby(["game_pk", "inning_topbot"])
            .agg(bip_count=("_barrel", "count"), barrels=("_barrel", "sum"))
            .reset_index()
            .assign(barrel_rate_off=lambda d: d["barrels"] / d["bip_count"].clip(lower=1))
            [["game_pk", "inning_topbot", "barrel_rate_off"]]
        )
    else:
        barrel_agg = pd.DataFrame(columns=["game_pk", "inning_topbot", "barrel_rate_off"])

    # Game date lookup (one row per game_pk)
    date_lookup = pitches.groupby("game_pk")["game_date"].first().reset_index()

    # Merge all stats and map inning_topbot → side
    result = xwoba_agg.merge(barrel_agg, on=["game_pk", "inning_topbot"], how="outer")
    result = result.merge(date_lookup, on="game_pk", how="left")
    result["side"] = result["inning_topbot"].map({"Bot": "home", "Top": "away"})
    result = result.dropna(subset=["side"])
    return result[["game_pk", "game_date", "side", "xwoba_off", "barrel_rate_off"]]


def aggregate_team_game_bullpen(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (game_pk, pitching side) bullpen Statcast stats.

    The starter is identified as the first-inning pitcher for each side and
    excluded from that team's pitching totals. This creates one bullpen row per
    game/team using only reliever pitches from that game.

    Returns DataFrame with: game_pk, game_date, side,
    bullpen_xwoba_against, bullpen_whiff_rate, bullpen_barrel_rate,
    bullpen_pa, bullpen_pitches.
    """
    if pitches.empty or "inning_topbot" not in pitches.columns or "inning" not in pitches.columns:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "side"] + _BULLPEN_STAT_COLS
            + ["bullpen_pa", "bullpen_pitches"]
        )

    p = pitches.copy()
    p["game_date"] = pd.to_datetime(p["game_date"])
    p["side"] = p["inning_topbot"].map({"Top": "home", "Bot": "away"})
    p = p.dropna(subset=["side", "pitcher"])
    p["pitcher"] = pd.to_numeric(p["pitcher"], errors="coerce")
    p = p.dropna(subset=["pitcher"])

    first = p[p["inning"] == 1]
    if first.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "side"] + _BULLPEN_STAT_COLS
            + ["bullpen_pa", "bullpen_pitches"]
        )

    starter_counts = (
        first.groupby(["game_pk", "side", "pitcher"], as_index=False)
        .size()
        .sort_values(["game_pk", "side", "size"], ascending=[True, True, False])
        .drop_duplicates(["game_pk", "side"])
        .rename(columns={"pitcher": "_starter"})
        [["game_pk", "side", "_starter"]]
    )
    p = p.merge(starter_counts, on=["game_pk", "side"], how="left")
    pen = p[p["pitcher"] != p["_starter"]].copy()
    if pen.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "side"] + _BULLPEN_STAT_COLS
            + ["bullpen_pa", "bullpen_pitches"]
        )

    pen["_is_pa"] = _is_pa(pen).astype(float)
    pen["_is_swing"] = pen["description"].isin(_SWING_DESCS).astype(float)
    pen["_is_whiff"] = pen["description"].isin(_WHIFF_DESCS).astype(float)
    pen["_is_bip"] = pen["type"].eq("X").astype(float)
    if "barrel" in pen.columns:
        pen["_is_barrel"] = pen["barrel"].fillna(0).astype(float).clip(upper=1)
    elif "launch_speed_angle" in pen.columns:
        pen["_is_barrel"] = (pen["launch_speed_angle"] == _BARREL_ZONE).astype(float)
    else:
        pen["_is_barrel"] = float("nan")
    pen["_xwoba_sum"] = (
        pen["estimated_woba_using_speedangle"].where(pen["_is_pa"].eq(1), 0).fillna(0)
    )

    agg = (
        pen.groupby(["game_pk", "game_date", "side"], as_index=False)
        .agg(
            bullpen_pa=("_is_pa", "sum"),
            bullpen_pitches=("pitcher", "size"),
            _xwoba_sum=("_xwoba_sum", "sum"),
            _swings=("_is_swing", "sum"),
            _whiffs=("_is_whiff", "sum"),
            _bip=("_is_bip", "sum"),
            _barrels=("_is_barrel", "sum"),
        )
    )
    agg["bullpen_xwoba_against"] = agg["_xwoba_sum"] / agg["bullpen_pa"].clip(lower=1)
    agg["bullpen_whiff_rate"] = agg["_whiffs"] / agg["_swings"].clip(lower=1)
    agg["bullpen_barrel_rate"] = agg["_barrels"] / agg["_bip"].clip(lower=1)
    return agg[
        ["game_pk", "game_date", "side"]
        + _BULLPEN_STAT_COLS
        + ["bullpen_pa", "bullpen_pitches"]
    ]


# ---------------------------------------------------------------------------
# Lineup derivation and batter handedness splits
# ---------------------------------------------------------------------------

def aggregate_game_lineups(pitches: pd.DataFrame) -> pd.DataFrame:
    """Derive per-game effective lineups from Statcast pitch data.

    Bot innings → home team batters; Top innings → away team batters.

    Returns a wide DataFrame with one row per game_pk:
        game_pk, home_lineup_ids (list[int]), away_lineup_ids (list[int])
    """
    if pitches.empty or "inning_topbot" not in pitches.columns:
        return pd.DataFrame(columns=["game_pk", "home_lineup_ids", "away_lineup_ids"])

    p = pitches[["game_pk", "inning_topbot", "batter"]].copy()
    p["batter"] = pd.to_numeric(p["batter"], errors="coerce")
    p = p.dropna(subset=["batter"])
    p["batter"] = p["batter"].astype(int)

    home = (
        p[p["inning_topbot"] == "Bot"]
        .groupby("game_pk")["batter"]
        .apply(lambda x: list(x.unique()))
        .rename("home_lineup_ids")
    )
    away = (
        p[p["inning_topbot"] == "Top"]
        .groupby("game_pk")["batter"]
        .apply(lambda x: list(x.unique()))
        .rename("away_lineup_ids")
    )
    return pd.DataFrame({"home_lineup_ids": home, "away_lineup_ids": away}).reset_index()


def compute_batter_season_splits(pitches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Vectorised batter season xwOBA split by opposing pitcher handedness.

    Returns one row per batter with columns:
        batter, year, xwoba_vs_R, xwoba_vs_L, pa_vs_R, pa_vs_L
    """
    if pitches.empty or "p_throws" not in pitches.columns:
        return pd.DataFrame(columns=["batter", "year", "xwoba_vs_R", "xwoba_vs_L", "pa_vs_R", "pa_vs_L"])

    pa = pitches.loc[_is_pa(pitches), ["batter", "p_throws", "estimated_woba_using_speedangle"]].copy()
    pa["batter"] = pd.to_numeric(pa["batter"], errors="coerce")
    pa = pa.dropna(subset=["batter", "p_throws"])
    pa["batter"] = pa["batter"].astype(int)
    pa = pa[pa["p_throws"].isin(["L", "R"])]

    if pa.empty:
        return pd.DataFrame(columns=["batter", "year", "xwoba_vs_R", "xwoba_vs_L", "pa_vs_R", "pa_vs_L"])

    agg = (
        pa.groupby(["batter", "p_throws"], as_index=False)
        .agg(pa_count=("batter", "count"), xwoba=("estimated_woba_using_speedangle", "mean"))
    )
    # Pivot: rows=batter, columns=(stat, hand)
    wide = agg.pivot(index="batter", columns="p_throws", values=["xwoba", "pa_count"])
    # Flatten MultiIndex: ("xwoba","L") → "xwoba_vs_L", ("pa_count","R") → "pa_vs_R"
    wide.columns = [
        f"xwoba_vs_{hand}" if stat == "xwoba" else f"pa_vs_{hand}"
        for stat, hand in wide.columns
    ]
    wide = wide.reset_index()
    wide["year"] = year
    for col in ["xwoba_vs_R", "xwoba_vs_L", "pa_vs_R", "pa_vs_L"]:
        if col not in wide.columns:
            wide[col] = float("nan")
    return wide[["batter", "year", "xwoba_vs_R", "xwoba_vs_L", "pa_vs_R", "pa_vs_L"]]

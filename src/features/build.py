"""Assemble the final modeling DataFrame.

Joins all feature functions into one row per (game_pk, side) where side is
'home' or 'away', then pivots to one row per game with home_* and away_*
columns plus a target.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.features.team import days_rest, home_away_split, season_to_date
from src.utils.logging import configure_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def load_processed_games(processed_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Load all processed game data for the given year range."""
    all_games = []
    for year in range(start_year, end_year + 1):
        path = processed_dir / "games" / f"games_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            all_games.append(df)
            logger.debug("loaded %d games from %d", len(df), year)
        else:
            logger.warning("no processed data for %d", year)

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk"]).reset_index(drop=True)
    if len(combined) < n_before:
        logger.warning(
            "dropped %d duplicate game_pk rows from processed games",
            n_before - len(combined),
        )
    return combined


def compute_team_rolling_features(games: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Compute rolling team features using only past games (no leakage).

    Args:
        games: DataFrame with game data
        n_games: number of games for rolling window

    Returns:
        DataFrame with rolling features per (game_pk, team_id)
    """
    # Create team-game view (one row per team per game)
    home_games = games[[
        "game_pk", "game_date", "home_team_id", "home_score", "away_score", "venue_id"
    ]].copy()
    home_games["team_id"] = home_games["home_team_id"]
    home_games["runs_for"] = home_games["home_score"]
    home_games["runs_against"] = home_games["away_score"]
    home_games["is_home"] = 1

    away_games = games[[
        "game_pk", "game_date", "away_team_id", "home_score", "away_score", "venue_id"
    ]].copy()
    away_games["team_id"] = away_games["away_team_id"]
    away_games["runs_for"] = away_games["away_score"]
    away_games["runs_against"] = away_games["home_score"]
    away_games["is_home"] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # Compute win indicator
    team_games["won"] = (team_games["runs_for"] > team_games["runs_against"]).astype(int)
    team_games["run_diff"] = team_games["runs_for"] - team_games["runs_against"]

    # Rolling features (shift by 1 to avoid leakage - only use past games)
    for n in [5, 10, 20]:
        window = n
        # Group by team and compute rolling stats
        grouped = team_games.groupby("team_id", group_keys=False)
        
        # Rolling sum of wins
        team_games[f"wins_l{n}"] = grouped["won"].transform(
            lambda x, window=window: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
        
        # Rolling sum of run differential
        team_games[f"run_diff_l{n}"] = grouped["run_diff"].transform(
            lambda x, window=window: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
        
        # Rolling average of runs scored
        team_games[f"avg_runs_for_l{n}"] = grouped["runs_for"].transform(
            lambda x, window=window: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling average of runs allowed
        team_games[f"avg_runs_against_l{n}"] = grouped["runs_against"].transform(
            lambda x, window=window: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Win percentage
        team_games[f"win_pct_l{n}"] = team_games[f"wins_l{n}"] / n

    return team_games


def join_pitcher_features(
    games: pd.DataFrame,
    raw_dir: Path,
    n_starts: int = 3,
) -> pd.DataFrame:
    """Join rolling SP Statcast features to a game-level DataFrame.

    Loads all cached Statcast monthly files that overlap the games' date range,
    identifies the home and away starter per game (via inning/inning_topbot),
    pre-computes rolling n-start averages (vectorised, shift(1) anti-leakage),
    then merges to games with a pd.merge_asof on game_date.

    Games with no cached Statcast data get NaN for all pitcher columns; the
    model fills those with 0 at training time (neutral prior).

    Requires Statcast cache to have been built with the current _KEEP_COLS
    (includes inning / inning_topbot).  Run fetch_game_features.py --force to
    regenerate stale cache files.
    """
    from src.data.statcast import (
        _SP_STAT_COLS,
        aggregate_game_starters,
        aggregate_pitcher_starts,
        compute_pitcher_rolling_features,
        load_statcast,
    )

    if games.empty:
        return games

    gmin = pd.Timestamp(games["game_date"].min())
    gmax = pd.Timestamp(games["game_date"].max())

    logger.info(
        "loading statcast for pitcher features (%s -> %s)", gmin.date(), gmax.date()
    )
    pitches = load_statcast(gmin.date(), gmax.date(), raw_dir=raw_dir)

    if pitches.empty:
        logger.warning("no statcast data found; pitcher features will be NaN")
        return games

    starts = aggregate_pitcher_starts(pitches)
    if starts.empty:
        return games

    starters = aggregate_game_starters(pitches)
    rolling = compute_pitcher_rolling_features(starts, n_starts=n_starts)

    # Column names we want from the rolling table
    roll_cols = [f"{s}_l{n_starts}" for s in _SP_STAT_COLS]
    keep = ["pitcher", "game_date"] + [c for c in roll_cols if c in rolling.columns]
    # merge_asof requires the right side to be globally sorted by the on-key (game_date),
    # not just sorted within each by-group (pitcher). Sort by game_date only.
    rolling_clean = (
        rolling[keep]
        .assign(game_date=lambda d: pd.to_datetime(d["game_date"]).astype("datetime64[us]"))
        .sort_values("game_date")
    )

    games = games.copy()
    # Normalise to us to match Statcast parquet datetime precision
    games["game_date"] = pd.to_datetime(games["game_date"]).astype("datetime64[us]")

    # Merge starters only if SP IDs not already present (e.g. from gamelog join)
    if "home_sp_id" not in games.columns or "away_sp_id" not in games.columns:
        if not starters.empty:
            games = games.merge(starters, on="game_pk", how="left")
        else:
            games["home_sp_id"] = None
            games["away_sp_id"] = None

    # For each side, use merge_asof to find each SP's rolling stats as of game_date.
    # shift(1) in compute_pitcher_rolling_features means the stats at a start row
    # already exclude that start — so matching on game_date <= game is correct.
    for side in ("home", "away"):
        sp_col = f"{side}_sp_id"
        if sp_col not in games.columns:
            for c in roll_cols:
                games[f"{side}_sp_{c}"] = float("nan")
            continue

        join_left = (
            games[["game_pk", "game_date", sp_col]]
            .rename(columns={sp_col: "pitcher"})
            .dropna(subset=["pitcher"])
            .assign(pitcher=lambda d: d["pitcher"].astype(int))
            .sort_values("game_date")
        )

        merged = pd.merge_asof(
            join_left,
            rolling_clean,
            on="game_date",
            by="pitcher",
            direction="backward",
        )
        merged = merged.rename(columns={c: f"{side}_sp_{c}" for c in roll_cols})
        merged = merged[["game_pk"] + [f"{side}_sp_{c}" for c in roll_cols if f"{side}_sp_{c}" in merged.columns]]

        games = games.merge(merged, on="game_pk", how="left")

    logger.info("joined pitcher features; columns now %d", len(games.columns))
    return games


def build_pitcher_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    n_starts: int = 3,
    lookback_days: int = 60,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Compute rolling SP features for today's probable starters.

    Used by predict_today.py to add pitcher columns before scoring.
    Returns a DataFrame with game_pk + home_sp_*/away_sp_* feature columns.
    """
    from src.data.statcast import (
        _SP_STAT_COLS,
        aggregate_pitcher_starts,
        compute_pitcher_rolling_features,
        load_statcast,
    )

    if slate.empty:
        return pd.DataFrame()

    game_dates = _slate_dates(slate, target_date)
    min_date = min(game_dates.values())
    max_date = max(game_dates.values())
    start_date = min_date - timedelta(days=lookback_days)

    pitches = load_statcast(start_date, max_date - timedelta(days=1), raw_dir=raw_dir)
    if pitches.empty:
        return pd.DataFrame({"game_pk": slate["game_pk"]})

    starts = aggregate_pitcher_starts(pitches)
    if starts.empty:
        return pd.DataFrame({"game_pk": slate["game_pk"]})

    rolling = compute_pitcher_rolling_features(starts, n_starts=n_starts)
    roll_cols = [f"{s}_l{n_starts}" for s in _SP_STAT_COLS]
    keep = ["pitcher", "game_date"] + [c for c in roll_cols if c in rolling.columns]
    rolling_clean = rolling[keep].sort_values(["pitcher", "game_date"])

    rows = []
    for game_idx, game in slate.iterrows():
        target_ts = pd.Timestamp(game_dates[game_idx])
        row: dict = {"game_pk": game["game_pk"]}
        for side in ("home", "away"):
            sp_id = game.get(f"{side}_sp_id")
            if sp_id is None or pd.isna(sp_id):
                for c in roll_cols:
                    row[f"{side}_sp_{c}"] = float("nan")
                continue
            sp_rows = rolling_clean[
                (rolling_clean["pitcher"] == int(sp_id))
                & (rolling_clean["game_date"] < target_ts)
            ]
            if sp_rows.empty:
                for c in roll_cols:
                    row[f"{side}_sp_{c}"] = float("nan")
            else:
                latest = sp_rows.sort_values("game_date").iloc[-1]
                for c in roll_cols:
                    row[f"{side}_sp_{c}"] = latest.get(c, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


def join_gamelog_pitcher_features(
    games: pd.DataFrame,
    raw_dir: Path,
    n_starts: int = 3,
    recent_days: int = 60,
) -> pd.DataFrame:
    """Join rolling SP features from Stats API gamelogs to a game-level DataFrame.

    Reliable alternative to join_pitcher_features (Statcast); uses the same
    boxscore endpoint as the schedule fetcher so no Baseball Savant dependency.
    """
    from src.data.pitching_gamelogs import (
        GAMELOG_STAT_COLS,
        STD_STAT_COLS,
        compute_starter_rolling_features,
        compute_starter_season_to_date,
        load_pitching_gamelogs,
    )

    if games.empty:
        return games

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    gmin = games["game_date"].min()
    gmax = games["game_date"].max()

    gamelogs = load_pitching_gamelogs(gmin.year, gmax.year, raw_dir=raw_dir)
    if gamelogs.empty:
        logger.warning("no pitching gamelogs found; pitcher features will be NaN")
        return games

    rolling = compute_starter_rolling_features(gamelogs, n_starts=n_starts)
    std = compute_starter_season_to_date(gamelogs)

    roll_cols = [f"{s}_l{n_starts}" for s in GAMELOG_STAT_COLS]
    std_cols = STD_STAT_COLS

    # SP days rest: days since each pitcher's previous start (shift-1 per pitcher)
    gl_rest = gamelogs.copy()
    gl_rest["game_date"] = pd.to_datetime(gl_rest["game_date"])
    gl_rest["_year"] = gl_rest["game_date"].dt.year
    gl_rest = gl_rest.sort_values(["pitcher_id", "game_date"])
    gl_rest["_prev_start"] = gl_rest.groupby("pitcher_id")["game_date"].shift(1)
    gl_rest["sp_days_rest"] = (gl_rest["game_date"] - gl_rest["_prev_start"]).dt.days
    gl_rest["sp_season_starts_prior"] = gl_rest.groupby(["pitcher_id", "_year"]).cumcount()
    gl_rest[f"sp_recent_starts_l{recent_days}d"] = 0
    for _, grp in gl_rest.groupby("pitcher_id"):
        grp = grp.sort_values("game_date")
        for row_idx, row in grp.iterrows():
            start = row["game_date"] - pd.Timedelta(days=recent_days)
            prior = grp[(grp["game_date"] < row["game_date"]) & (grp["game_date"] >= start)]
            gl_rest.loc[row_idx, f"sp_recent_starts_l{recent_days}d"] = len(prior)
    gl_rest["sp_short_history"] = (
        gl_rest[f"sp_recent_starts_l{recent_days}d"] < n_starts
    ).astype("int8")
    for side in ("home", "away"):
        side_rest = (
            gl_rest[gl_rest["side"] == side][[
                "game_pk",
                "sp_days_rest",
                "sp_season_starts_prior",
                f"sp_recent_starts_l{recent_days}d",
                "sp_short_history",
            ]]
            .rename(columns={
                "sp_days_rest": f"{side}_sp_days_rest",
                "sp_season_starts_prior": f"{side}_sp_season_starts_prior",
                f"sp_recent_starts_l{recent_days}d": f"{side}_sp_recent_starts_l{recent_days}d",
                "sp_short_history": f"{side}_sp_short_history",
            })
        )
        games = games.merge(side_rest, on="game_pk", how="left")

    # Identify home/away SP per game from the gamelogs
    for side in ("home", "away"):
        sp_map = (
            rolling[rolling["side"] == side][["game_pk", "pitcher_id"]]
            .rename(columns={"pitcher_id": f"{side}_sp_id"})
        )
        games = games.merge(sp_map, on="game_pk", how="left")
        games[f"{side}_sp_unknown"] = games[f"{side}_sp_id"].isna().astype("int8")

    def _make_lookup(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        keep = ["pitcher_id", "game_date"] + [c for c in feature_cols if c in df.columns]
        return (
            df[keep]
            .dropna(subset=["pitcher_id", "game_date"])
            .assign(
                pitcher_id=lambda d: d["pitcher_id"].astype("int64"),
                game_date=lambda d: d["game_date"].astype("datetime64[ms]"),
            )
            .sort_values("game_date")  # merge_asof requires right sorted by on-key globally
        )

    games["game_date"] = games["game_date"].astype("datetime64[ms]")
    rolling_clean = _make_lookup(rolling, roll_cols)
    std_clean = _make_lookup(std, std_cols)

    for side in ("home", "away"):
        sp_col = f"{side}_sp_id"
        # Deduplicate to one row per game_pk before merge_asof so the result
        # stays unique on game_pk and avoids a many-to-many explosion when
        # merging back onto games (which may itself have duplicate game_pks).
        join_left = (
            games[["game_pk", "game_date", sp_col]]
            .drop_duplicates(subset=["game_pk"])
            .rename(columns={sp_col: "pitcher_id"})
            .dropna(subset=["pitcher_id"])
            .assign(pitcher_id=lambda d: d["pitcher_id"].astype(int))
            .sort_values("game_date")
        )

        for lookup, feat_cols in [
            (rolling_clean, roll_cols),
            (std_clean, std_cols),
        ]:
            merged = pd.merge_asof(
                join_left,
                lookup,
                on="game_date",
                by="pitcher_id",
                direction="backward",
            )
            merged = merged.rename(columns={c: f"{side}_sp_{c}" for c in feat_cols})
            out_cols = ["game_pk"] + [
                f"{side}_sp_{c}" for c in feat_cols if f"{side}_sp_{c}" in merged.columns
            ]
            games = games.merge(merged[out_cols], on="game_pk", how="left")

    logger.info("joined gamelog pitcher features; columns now %d", len(games.columns))
    return games


def build_gamelog_pitcher_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    n_starts: int = 3,
    lookback_days: int = 60,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Rolling SP features for today's probable starters using Stats API gamelog data."""
    from src.data.pitching_gamelogs import (
        GAMELOG_STAT_COLS,
        STD_STAT_COLS,
        _add_per_start_rates,
        load_pitching_gamelogs,
    )

    if slate.empty:
        return pd.DataFrame()

    slate = slate.copy()
    game_dates = _slate_dates(slate, target_date)
    min_date = min(game_dates.values())
    max_date = max(game_dates.values())
    start_year = (min_date - timedelta(days=lookback_days)).year

    gamelogs = load_pitching_gamelogs(start_year, max_date.year, raw_dir=raw_dir)
    if gamelogs.empty:
        return pd.DataFrame({"game_pk": slate["game_pk"]})

    gamelogs = _add_per_start_rates(gamelogs)
    gamelogs["game_date"] = pd.to_datetime(gamelogs["game_date"])
    roll_cols = [f"{s}_l{n_starts}" for s in GAMELOG_STAT_COLS]
    std_cols = STD_STAT_COLS

    rows = []
    for game_idx, game in slate.iterrows():
        target_ts = pd.Timestamp(game_dates[game_idx])
        row: dict = {"game_pk": game["game_pk"]}
        for side in ("home", "away"):
            sp_id = game.get(f"{side}_sp_id")
            if sp_id is None or pd.isna(sp_id):
                for c in roll_cols + std_cols:
                    row[f"{side}_sp_{c}"] = float("nan")
                row[f"{side}_sp_days_rest"] = float("nan")
                row[f"{side}_sp_season_starts_prior"] = 0
                row[f"{side}_sp_recent_starts_l{lookback_days}d"] = 0
                row[f"{side}_sp_short_history"] = 1
                row[f"{side}_sp_unknown"] = 1
                continue
            pid = int(sp_id)
            prior = gamelogs[
                (gamelogs["pitcher_id"] == pid) & (gamelogs["game_date"] < target_ts)
            ].sort_values("game_date")
            row[f"{side}_sp_unknown"] = 0

            # Days since most recent start before this game
            if not prior.empty:
                row[f"{side}_sp_days_rest"] = (target_ts - prior.iloc[-1]["game_date"]).days
            else:
                row[f"{side}_sp_days_rest"] = float("nan")
            recent_start = target_ts - pd.Timedelta(days=lookback_days)
            recent_prior = prior[prior["game_date"] >= recent_start]
            row[f"{side}_sp_recent_starts_l{lookback_days}d"] = len(recent_prior)
            row[f"{side}_sp_short_history"] = int(len(recent_prior) < n_starts)

            recent = prior.tail(n_starts)
            for stat in GAMELOG_STAT_COLS:
                value = recent[stat].mean() if not recent.empty and stat in recent else float("nan")
                row[f"{side}_sp_{stat}_l{n_starts}"] = value

            season_prior = prior[prior["game_date"].dt.year == target_ts.year]
            row[f"{side}_sp_season_starts_prior"] = len(season_prior)
            _add_pitcher_season_to_date(row, side, season_prior)
        rows.append(row)

    return pd.DataFrame(rows)


def join_lineup_matchup_features(
    games: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """Join lineup-vs-starter handedness and BvP features.

    For each game computes:
      home_lineup_xwoba_vs_sp  — home batters' mean prior-year xwOBA vs away SP's hand
      away_lineup_xwoba_vs_sp  — away batters' mean prior-year xwOBA vs home SP's hand
      home_bvp_xwoba           — home lineup PA-weighted BvP xwOBA vs away SP
      away_bvp_xwoba           — away lineup PA-weighted BvP xwOBA vs home SP

    Leakage controls:
      Batter splits: uses year Y-1 season stats for games in year Y.
      BvP: uses all pitches in the loaded range (minor leakage for early-season
           games offset by the ≥20 PA filter — most pairs won't clear it within
           a single season).
    """
    from src.data.statcast import (
        aggregate_game_lineups,
        aggregate_pitcher_starts,
        compute_batter_season_splits,
        load_statcast,
    )

    if games.empty:
        return games

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    gmin = games["game_date"].min()
    gmax = games["game_date"].max()

    logger.info("loading statcast for lineup matchup features (%s -> %s)", gmin.date(), gmax.date())
    pitches = load_statcast(gmin.date(), gmax.date(), raw_dir=raw_dir)

    _nan_cols = ["home_lineup_xwoba_vs_sp", "away_lineup_xwoba_vs_sp",
                 "home_bvp_xwoba", "away_bvp_xwoba"]
    if pitches.empty or "inning_topbot" not in pitches.columns:
        for c in _nan_cols:
            games[c] = float("nan")
        return games

    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    # --- Effective lineups per game (from actual PA data) ---
    game_lineups = aggregate_game_lineups(pitches)

    # --- SP handedness: p_throws for the pitcher in inning 1 per game/side ---
    starts = aggregate_pitcher_starts(pitches)
    sp_throws = (
        starts[["pitcher", "game_pk", "p_throws"]]
        .rename(columns={"pitcher": "pitcher_id"})
        .drop_duplicates(subset=["game_pk", "pitcher_id"])
    )

    # Map SP handedness onto games using already-joined home_sp_id / away_sp_id
    if "home_sp_id" in games.columns and "away_sp_id" in games.columns:
        home_throws = (
            games[["game_pk", "home_sp_id"]].dropna(subset=["home_sp_id"])
            .assign(pitcher_id=lambda d: d["home_sp_id"].astype(int))
            .merge(sp_throws, on=["game_pk", "pitcher_id"], how="left")
            .rename(columns={"p_throws": "home_sp_throws"})[["game_pk", "home_sp_throws"]]
        )
        away_throws = (
            games[["game_pk", "away_sp_id"]].dropna(subset=["away_sp_id"])
            .assign(pitcher_id=lambda d: d["away_sp_id"].astype(int))
            .merge(sp_throws, on=["game_pk", "pitcher_id"], how="left")
            .rename(columns={"p_throws": "away_sp_throws"})[["game_pk", "away_sp_throws"]]
        )
    else:
        home_throws = pd.DataFrame(columns=["game_pk", "home_sp_throws"])
        away_throws = pd.DataFrame(columns=["game_pk", "away_sp_throws"])

    # --- Batter season splits (prior-year Y-1 for games in year Y) ---
    batter_splits_by_year: dict[int, pd.DataFrame] = {}
    for year in sorted(pitches["game_date"].dt.year.unique()):
        yr_pitches = pitches[pitches["game_date"].dt.year == year]
        batter_splits_by_year[year] = compute_batter_season_splits(yr_pitches, year)

    batter_splits_all = (
        pd.concat(batter_splits_by_year.values(), ignore_index=True)
        if batter_splits_by_year else pd.DataFrame()
    )

    # --- BvP plate appearances, filtered as-of per game below ---
    bvp_pa = (
        pitches.loc[
            pitches["woba_denom"].fillna(0).astype(float).eq(1),
            ["game_date", "batter", "pitcher", "estimated_woba_using_speedangle"],
        ]
        .dropna(subset=["game_date", "batter", "pitcher"])
        .assign(
            batter=lambda d: d["batter"].astype(int),
            pitcher=lambda d: d["pitcher"].astype(int),
        )
        .sort_values("game_date")
    )

    # --- Vectorised feature computation ---
    game_year = games[["game_pk", "game_date"]].copy()
    game_year["game_year"] = game_year["game_date"].dt.year

    lineup_wide = game_lineups.merge(game_year[["game_pk", "game_date", "game_year"]], on="game_pk", how="left")
    lineup_wide = lineup_wide.merge(home_throws, on="game_pk", how="left")
    lineup_wide = lineup_wide.merge(away_throws, on="game_pk", how="left")
    if "home_sp_id" in games.columns:
        lineup_wide = lineup_wide.merge(
            games[["game_pk", "home_sp_id", "away_sp_id"]], on="game_pk", how="left"
        )
    else:
        lineup_wide["home_sp_id"] = None
        lineup_wide["away_sp_id"] = None

    def _lineup_xwoba(row_iter, lineup_col, sp_throws_col, splits_all):
        """Expand lineups and join prior-year batter splits."""
        rows = []
        for _, r in row_iter.iterrows():
            ids = r.get(lineup_col) or []
            hand = r.get(sp_throws_col)
            game_year_raw = r.get("game_year")
            if pd.isna(game_year_raw):
                rows.append({"game_pk": r["game_pk"], "val": float("nan")})
                continue
            yr = int(game_year_raw) - 1
            xwoba_col = f"xwoba_vs_{hand}" if hand in ("L", "R") else None
            splits = splits_all[splits_all["year"] == yr] if not splits_all.empty else pd.DataFrame()
            vals = []
            for bid in ids:
                if xwoba_col is None or splits.empty:
                    continue
                match = splits.loc[splits["batter"] == bid, xwoba_col]
                if not match.empty and not pd.isna(match.iloc[0]):
                    vals.append(match.iloc[0])
            rows.append({"game_pk": r["game_pk"],
                         "val": float(np.mean(vals)) if vals else float("nan")})
        return pd.DataFrame(rows).set_index("game_pk")["val"]

    import numpy as np

    home_xwoba = _lineup_xwoba(lineup_wide, "home_lineup_ids", "away_sp_throws", batter_splits_all)
    away_xwoba = _lineup_xwoba(lineup_wide, "away_lineup_ids", "home_sp_throws", batter_splits_all)

    def _bvp(row_iter, lineup_col, sp_id_col, bvp_source):
        if bvp_source.empty:
            return pd.Series(dtype=float, index=pd.Index([], name="game_pk"))

        lookup = bvp_source.rename(
            columns={"estimated_woba_using_speedangle": "_xwoba"}
        ).sort_values(["batter", "pitcher", "game_date"])
        grouped = lookup.groupby(["batter", "pitcher"], group_keys=False)
        lookup["pa_count"] = grouped.cumcount() + 1
        lookup["_xwoba_sum"] = grouped["_xwoba"].cumsum()
        lookup["xwoba_bvp"] = lookup["_xwoba_sum"] / lookup["pa_count"].clip(lower=1)
        lookup = lookup[["game_date", "batter", "pitcher", "pa_count", "xwoba_bvp"]]

        left_rows = []
        for _, r in row_iter.iterrows():
            ids = r.get(lineup_col) or []
            raw_sp = r.get(sp_id_col)
            game_date = r.get("game_date")
            if raw_sp is None or pd.isna(raw_sp) or pd.isna(game_date):
                continue
            for bid in ids:
                left_rows.append({
                    "game_pk": r["game_pk"],
                    "game_date": pd.Timestamp(game_date),
                    "batter": int(bid),
                    "pitcher": int(raw_sp),
                })

        left = pd.DataFrame(left_rows)
        if left.empty:
            return pd.Series(dtype=float, index=pd.Index([], name="game_pk"))

        merged = pd.merge_asof(
            left.sort_values("game_date"),
            lookup.sort_values("game_date"),
            on="game_date",
            by=["batter", "pitcher"],
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged[merged["pa_count"].fillna(0) >= 20].copy()
        if merged.empty:
            return pd.Series(dtype=float, index=pd.Index([], name="game_pk"))
        merged["_weight"] = merged["pa_count"].clip(upper=60) / 60
        merged["_weighted_xwoba"] = merged["_weight"] * merged["xwoba_bvp"]
        sums = merged.groupby("game_pk")[["_weighted_xwoba", "_weight"]].sum()
        return (sums["_weighted_xwoba"] / sums["_weight"]).rename("val")

    home_bvp = _bvp(lineup_wide, "home_lineup_ids", "away_sp_id", bvp_pa)
    away_bvp = _bvp(lineup_wide, "away_lineup_ids", "home_sp_id", bvp_pa)

    games = games.merge(home_xwoba.rename("home_lineup_xwoba_vs_sp").reset_index(), on="game_pk", how="left")
    games = games.merge(away_xwoba.rename("away_lineup_xwoba_vs_sp").reset_index(), on="game_pk", how="left")
    games = games.merge(home_bvp.rename("home_bvp_xwoba").reset_index(), on="game_pk", how="left")
    games = games.merge(away_bvp.rename("away_bvp_xwoba").reset_index(), on="game_pk", how="left")

    logger.info("joined lineup matchup features; home xwoba fill=%.1f%% bvp fill=%.1f%%",
                games["home_lineup_xwoba_vs_sp"].notna().mean() * 100,
                games["home_bvp_xwoba"].notna().mean() * 100)
    return games


def build_lineup_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    lookback_days: int = 400,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Lineup matchup features for today's slate.

    Uses day-of lineup_ids from the slate (posted ~90 min before first pitch).
    Falls back to NaN when lineups aren't posted yet.
    BvP uses all recent Statcast history; batter splits use the prior full season.
    """
    from src.data.statcast import (
        aggregate_bvp,
        aggregate_pitcher_starts,
        compute_batter_season_splits,
        load_statcast,
    )

    if slate.empty:
        return pd.DataFrame()

    game_dates = _slate_dates(slate, target_date)
    min_date = min(game_dates.values())
    max_date = max(game_dates.values())
    start = min_date - timedelta(days=lookback_days)
    pitches = load_statcast(start, max_date - timedelta(days=1), raw_dir=raw_dir)

    nan_row_base = {
        "home_lineup_xwoba_vs_sp": float("nan"),
        "away_lineup_xwoba_vs_sp": float("nan"),
        "home_bvp_xwoba": float("nan"),
        "away_bvp_xwoba": float("nan"),
    }

    if pitches.empty:
        return pd.DataFrame([{**{"game_pk": r["game_pk"]}, **nan_row_base}
                              for _, r in slate.iterrows()])

    pitches["game_date"] = pd.to_datetime(pitches["game_date"])

    # Prior-season batter splits
    prior_year = max_date.year - 1
    prior_pitches = pitches[pitches["game_date"].dt.year == prior_year]
    batter_splits = compute_batter_season_splits(prior_pitches, prior_year)

    # BvP
    bvp_table = aggregate_bvp(pitches)
    bvp_indexed = bvp_table.set_index(["batter", "pitcher"]) if not bvp_table.empty else pd.DataFrame()

    # SP handedness from recent starts
    starts = aggregate_pitcher_starts(pitches)
    sp_throw_map = dict(zip(starts["pitcher"].astype(int), starts["p_throws"]))

    rows = []
    for _, game in slate.iterrows():
        row: dict = {"game_pk": game["game_pk"], **nan_row_base}

        home_ids_raw = game.get("home_lineup_ids")
        away_ids_raw = game.get("away_lineup_ids")
        home_sp_id_raw = game.get("home_sp_id")
        away_sp_id_raw = game.get("away_sp_id")

        home_ids = [int(x) for x in home_ids_raw] if home_ids_raw else []
        away_ids = [int(x) for x in away_ids_raw] if away_ids_raw else []
        home_sp = int(home_sp_id_raw) if home_sp_id_raw and not pd.isna(home_sp_id_raw) else None
        away_sp = int(away_sp_id_raw) if away_sp_id_raw and not pd.isna(away_sp_id_raw) else None

        away_hand = sp_throw_map.get(away_sp) if away_sp else None
        home_hand = sp_throw_map.get(home_sp) if home_sp else None

        def _xwoba_vs(lineup, hand):
            col = f"xwoba_vs_{hand}" if hand in ("L", "R") else None
            if not col or batter_splits.empty:
                return float("nan")
            vals = [
                batter_splits.loc[batter_splits["batter"] == bid, col].iloc[0]
                for bid in lineup
                if not batter_splits.loc[batter_splits["batter"] == bid, col].empty
                and not pd.isna(batter_splits.loc[batter_splits["batter"] == bid, col].iloc[0])
            ]
            return float(np.mean(vals)) if vals else float("nan")

        def _bvp_score(lineup, sp_id):
            if not sp_id or bvp_indexed.empty:
                return float("nan")
            weights, vals = [], []
            for bid in lineup:
                try:
                    br = bvp_indexed.loc[(bid, sp_id)]
                    if br["pa_count"] >= 20:
                        w = min(br["pa_count"], 60) / 60
                        weights.append(w); vals.append(w * br["xwoba_bvp"])
                except KeyError:
                    pass
            return sum(vals) / sum(weights) if weights else float("nan")

        import numpy as np
        row["home_lineup_xwoba_vs_sp"] = _xwoba_vs(home_ids, away_hand)
        row["away_lineup_xwoba_vs_sp"] = _xwoba_vs(away_ids, home_hand)
        row["home_bvp_xwoba"] = _bvp_score(home_ids, away_sp)
        row["away_bvp_xwoba"] = _bvp_score(away_ids, home_sp)
        rows.append(row)

    return pd.DataFrame(rows)


def join_team_statcast_features(
    games: pd.DataFrame,
    raw_dir: Path,
    n_games: int = 10,
) -> pd.DataFrame:
    """Join rolling team batting Statcast features (xwOBA, barrel rate) to games.

    'home_xwoba_off_l{n}' = home team's rolling offensive xwOBA over last n games.
    'away_xwoba_off_l{n}' = away team's rolling offensive xwOBA over last n games.
    Uses shift(1) per team so the current game is never included in its own features.
    """
    from src.data.statcast import aggregate_team_game_hitting, load_statcast, _TEAM_HIT_COLS

    if games.empty:
        return games

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    gmin = games["game_date"].min()
    gmax = games["game_date"].max()

    logger.info("loading statcast for team batting features (%s -> %s)", gmin.date(), gmax.date())
    pitches = load_statcast(gmin.date(), gmax.date(), raw_dir=raw_dir)

    if pitches.empty:
        logger.warning("no statcast data; team batting features will be NaN")
        for side in ("home", "away"):
            for col in _TEAM_HIT_COLS:
                games[f"{side}_{col}_l{n_games}"] = float("nan")
        return games

    team_hits = aggregate_team_game_hitting(pitches)
    if team_hits.empty:
        for side in ("home", "away"):
            for col in _TEAM_HIT_COLS:
                games[f"{side}_{col}_l{n_games}"] = float("nan")
        return games

    # Join team_id from the games schedule
    id_lookup = (
        games[["game_pk", "home_team_id", "away_team_id"]]
        .drop_duplicates("game_pk")
    )
    team_hits = team_hits.merge(id_lookup, on="game_pk", how="left")
    team_hits["team_id"] = team_hits.apply(
        lambda r: r["home_team_id"] if r["side"] == "home" else r["away_team_id"], axis=1
    )
    team_hits = team_hits.dropna(subset=["team_id"])
    team_hits["team_id"] = team_hits["team_id"].astype(int)
    team_hits["game_date"] = pd.to_datetime(team_hits["game_date"])
    team_hits = team_hits.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # Rolling n-game averages per team, shift(1) to exclude current game
    roll_cols = [f"{col}_l{n_games}" for col in _TEAM_HIT_COLS]
    for col in _TEAM_HIT_COLS:
        team_hits[f"{col}_l{n_games}"] = (
            team_hits.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=n_games, min_periods=1).mean())
        )

    lookup = team_hits[["team_id", "game_date"] + roll_cols].sort_values("game_date")

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        join_left = (
            games[["game_pk", "game_date", team_col]]
            .dropna(subset=[team_col])
            .assign(**{team_col: lambda d: d[team_col].astype(int)})
            .sort_values("game_date")
        )
        merged = pd.merge_asof(
            join_left,
            lookup.rename(columns={"team_id": team_col}),
            on="game_date",
            by=team_col,
            direction="backward",
        )
        merged = merged.rename(columns={c: f"{side}_{c}" for c in roll_cols})
        out_cols = ["game_pk"] + [f"{side}_{c}" for c in roll_cols]
        games = games.merge(merged[out_cols], on="game_pk", how="left")

    logger.info("joined team statcast batting features; columns now %d", len(games.columns))
    return games


def build_team_statcast_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
    n_games: int = 10,
    lookback_days: int = 60,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Rolling team batting Statcast features for today's slate.

    Loads recent Statcast data, joins team_id via the processed game schedule,
    then computes each team's trailing n-game batting xwOBA and barrel rate.
    """
    from src.data.statcast import aggregate_team_game_hitting, load_statcast, _TEAM_HIT_COLS

    if slate.empty:
        return pd.DataFrame()

    game_dates = _slate_dates(slate, target_date)
    max_date = max(game_dates.values())
    start_date = max_date - timedelta(days=lookback_days)

    pitches = load_statcast(start_date, max_date - timedelta(days=1), raw_dir=raw_dir)
    if pitches.empty:
        return pd.DataFrame({"game_pk": slate["game_pk"]})

    team_hits = aggregate_team_game_hitting(pitches)
    if team_hits.empty:
        return pd.DataFrame({"game_pk": slate["game_pk"]})

    # Build game_pk → team_id mapping from processed games
    hist_games = load_processed_games(processed_dir, start_date.year - 1, max_date.year)
    id_lookup = (
        hist_games[["game_pk", "home_team_id", "away_team_id"]]
        .drop_duplicates("game_pk")
    )
    team_hits = team_hits.merge(id_lookup, on="game_pk", how="left")
    team_hits["team_id"] = team_hits.apply(
        lambda r: r["home_team_id"] if r["side"] == "home" else r["away_team_id"], axis=1
    )
    team_hits = team_hits.dropna(subset=["team_id"])
    team_hits["team_id"] = team_hits["team_id"].astype(int)
    team_hits["game_date"] = pd.to_datetime(team_hits["game_date"])
    team_hits = team_hits.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    roll_cols = [f"{col}_l{n_games}" for col in _TEAM_HIT_COLS]

    rows = []
    for game_idx, game in slate.iterrows():
        target_ts = pd.Timestamp(game_dates[game_idx])
        row: dict = {"game_pk": game["game_pk"]}
        for side in ("home", "away"):
            team_id_raw = game.get(f"{side}_team_id")
            if team_id_raw is None or pd.isna(team_id_raw):
                for col in _TEAM_HIT_COLS:
                    row[f"{side}_{col}_l{n_games}"] = float("nan")
                continue
            tid = int(team_id_raw)
            prior = team_hits[
                (team_hits["team_id"] == tid)
                & (team_hits["game_date"] < target_ts)
            ].tail(n_games)
            for col in _TEAM_HIT_COLS:
                row[f"{side}_{col}_l{n_games}"] = (
                    prior[col].mean() if not prior.empty and col in prior.columns else float("nan")
                )
        rows.append(row)

    return pd.DataFrame(rows)


_LINEUP_SLOT_WEIGHTS = [1.20, 1.15, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85]
_POSTED_LINEUP_FEATURES = [
    "lineup_xwoba_vs_hand_L30",
    "lineup_xwoba_weighted",
    "lineup_xwoba_top5",
    "lineup_barrel_rate_vs_hand_L30",
    "lineup_barrel_rate_weighted",
    "lineup_barrel_rate_top5",
]


def _coerce_lineup_ids(value) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [int(part) for part in value.split(",") if part.strip().isdigit()]
    try:
        return [int(pid) for pid in list(value)[:9] if pd.notna(pid)]
    except TypeError:
        return []


def _lineup_player_rates(
    pitches: pd.DataFrame,
    batter_ids: list[int],
    opposing_hand: str | None,
    target_ts: pd.Timestamp,
    lookback_days: int,
    pa_limit: int,
) -> pd.DataFrame:
    """Return one xwOBA/barrel row per lineup slot using prior Statcast PAs."""
    from src.data.statcast import _BARREL_ZONE, _is_pa

    if pitches.empty or not batter_ids:
        return pd.DataFrame(columns=["slot", "batter", "xwoba", "barrel_rate"])

    start_ts = target_ts - pd.Timedelta(days=lookback_days)
    p = pitches.copy()
    p["game_date"] = pd.to_datetime(p["game_date"])
    p = p[
        (p["game_date"] < target_ts)
        & (p["game_date"] >= start_ts)
        & (p["batter"].isin(batter_ids))
    ].copy()
    if opposing_hand:
        p = p[p["p_throws"] == opposing_hand]
    if p.empty:
        return pd.DataFrame(columns=["slot", "batter", "xwoba", "barrel_rate"])

    rows = []
    slot_lookup = {int(pid): idx + 1 for idx, pid in enumerate(batter_ids[:9])}
    for batter_id, grp in p.groupby("batter"):
        grp = grp.sort_values(["game_date", "at_bat_number"]).copy()
        pa = grp.loc[_is_pa(grp)].tail(pa_limit)
        xwoba = (
            pa["estimated_woba_using_speedangle"].dropna().mean()
            if not pa.empty and "estimated_woba_using_speedangle" in pa
            else float("nan")
        )
        bip = grp[grp["type"].eq("X")].tail(pa_limit) if "type" in grp else pd.DataFrame()
        if not bip.empty and "barrel" in bip:
            barrels = bip["barrel"].fillna(0).astype(float).clip(upper=1).sum()
            barrel_rate = barrels / len(bip)
        elif not bip.empty and "launch_speed_angle" in bip:
            barrel_rate = (bip["launch_speed_angle"] == _BARREL_ZONE).sum() / len(bip)
        else:
            barrel_rate = float("nan")
        rows.append({
            "slot": slot_lookup.get(int(batter_id)),
            "batter": int(batter_id),
            "xwoba": float(xwoba) if pd.notna(xwoba) else float("nan"),
            "barrel_rate": float(barrel_rate) if pd.notna(barrel_rate) else float("nan"),
        })
    return pd.DataFrame(rows).dropna(subset=["slot"]).sort_values("slot")


def _prepare_lineup_history(pitches: pd.DataFrame) -> dict[str, dict[tuple[int, str], pd.DataFrame]]:
    """Index Statcast batter history for fast lineup lookups."""
    from src.data.statcast import _BARREL_ZONE, _is_pa

    if pitches.empty:
        return {"pa": {}, "bip": {}}

    p = pitches.copy()
    p["game_date"] = pd.to_datetime(p["game_date"])
    p["batter"] = pd.to_numeric(p["batter"], errors="coerce")
    p = p.dropna(subset=["batter", "p_throws"])
    p["batter"] = p["batter"].astype(int)

    pa = p.loc[_is_pa(p), [
        "batter", "p_throws", "game_date", "at_bat_number",
        "estimated_woba_using_speedangle",
    ]].copy()
    pa = pa.rename(columns={"estimated_woba_using_speedangle": "xwoba"})

    bip = p[p["type"].eq("X")].copy() if "type" in p else pd.DataFrame()
    if not bip.empty:
        if "barrel" in bip:
            bip["_barrel"] = bip["barrel"].fillna(0).astype(float).clip(upper=1)
        elif "launch_speed_angle" in bip:
            bip["_barrel"] = (bip["launch_speed_angle"] == _BARREL_ZONE).astype(float)
        else:
            bip["_barrel"] = float("nan")
        bip = bip[["batter", "p_throws", "game_date", "at_bat_number", "_barrel"]]
    else:
        bip = pd.DataFrame(columns=["batter", "p_throws", "game_date", "at_bat_number", "_barrel"])

    return {
        "pa": {
            key: grp.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)
            for key, grp in pa.groupby(["batter", "p_throws"])
        },
        "bip": {
            key: grp.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)
            for key, grp in bip.groupby(["batter", "p_throws"])
        },
    }


def _lineup_player_rates_from_history(
    history: dict[str, dict[tuple[int, str], pd.DataFrame]],
    batter_ids: list[int],
    opposing_hand: str | None,
    target_ts: pd.Timestamp,
    lookback_days: int,
    pa_limit: int,
) -> pd.DataFrame:
    if not batter_ids or not opposing_hand:
        return pd.DataFrame(columns=["slot", "batter", "xwoba", "barrel_rate"])

    start_ts = target_ts - pd.Timedelta(days=lookback_days)
    rows = []
    for slot, batter_id in enumerate(batter_ids[:9], start=1):
        key = (int(batter_id), opposing_hand)
        pa = history["pa"].get(key, pd.DataFrame())
        if not pa.empty:
            pa_window = pa[(pa["game_date"] < target_ts) & (pa["game_date"] >= start_ts)].tail(pa_limit)
            xwoba = pa_window["xwoba"].dropna().mean()
        else:
            xwoba = float("nan")

        bip = history["bip"].get(key, pd.DataFrame())
        if not bip.empty:
            bip_window = bip[(bip["game_date"] < target_ts) & (bip["game_date"] >= start_ts)].tail(pa_limit)
            barrel_rate = bip_window["_barrel"].dropna().mean()
        else:
            barrel_rate = float("nan")

        rows.append({
            "slot": slot,
            "batter": int(batter_id),
            "xwoba": float(xwoba) if pd.notna(xwoba) else float("nan"),
            "barrel_rate": float(barrel_rate) if pd.notna(barrel_rate) else float("nan"),
        })
    return pd.DataFrame(rows)


def _aggregate_lineup_rates(rates: pd.DataFrame, fallback_xwoba: float, fallback_barrel: float) -> dict[str, float]:
    if rates.empty or rates["xwoba"].notna().sum() == 0:
        return {
            "lineup_xwoba_vs_hand_L30": fallback_xwoba,
            "lineup_xwoba_weighted": fallback_xwoba,
            "lineup_xwoba_top5": fallback_xwoba,
            "lineup_barrel_rate_vs_hand_L30": fallback_barrel,
            "lineup_barrel_rate_weighted": fallback_barrel,
            "lineup_barrel_rate_top5": fallback_barrel,
        }

    weights = pd.Series(_LINEUP_SLOT_WEIGHTS[: len(rates)], index=rates.index)

    def weighted(col: str) -> float:
        valid = rates[col].notna()
        if not valid.any():
            return fallback_barrel if "barrel" in col else fallback_xwoba
        return float((rates.loc[valid, col] * weights.loc[valid]).sum() / weights.loc[valid].sum())

    return {
        "lineup_xwoba_vs_hand_L30": float(rates["xwoba"].mean()),
        "lineup_xwoba_weighted": weighted("xwoba"),
        "lineup_xwoba_top5": float(rates.head(5)["xwoba"].mean()),
        "lineup_barrel_rate_vs_hand_L30": float(rates["barrel_rate"].mean()),
        "lineup_barrel_rate_weighted": weighted("barrel_rate"),
        "lineup_barrel_rate_top5": float(rates.head(5)["barrel_rate"].mean()),
    }


def join_posted_lineup_features(
    games: pd.DataFrame,
    raw_dir: Path,
    lookback_days: int = 30,
    pa_limit: int = 100,
) -> pd.DataFrame:
    """Join posted-lineup weighted batter xwOBA/barrel features."""
    from src.data.lineups import load_lineups_for_games
    from src.data.statcast import load_statcast

    if games.empty:
        return games

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    lineups = load_lineups_for_games(games["game_pk"], raw_dir=raw_dir, historical=True)
    keep = [
        "game_pk", "home_lineup_ids", "away_lineup_ids",
        "home_lineup_status", "away_lineup_status",
        "home_sp_hand", "away_sp_hand", "lineup_status",
    ]
    games = games.merge(lineups[[c for c in keep if c in lineups]], on="game_pk", how="left")

    gmin = games["game_date"].min() - pd.Timedelta(days=lookback_days)
    gmax = games["game_date"].max() - pd.Timedelta(days=1)
    pitches = load_statcast(gmin.date(), gmax.date(), raw_dir=raw_dir)
    history = _prepare_lineup_history(pitches)

    rows = []
    for _, game in games.iterrows():
        row: dict = {"game_pk": game["game_pk"]}
        missing = False
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            status = game.get(f"{side}_lineup_status")
            fallback_xwoba = game.get(f"{side}_xwoba_off_l10", float("nan"))
            fallback_barrel = game.get(f"{side}_barrel_rate_off_l10", float("nan"))
            if status != "confirmed":
                missing = True
                agg = _aggregate_lineup_rates(pd.DataFrame(), fallback_xwoba, fallback_barrel)
            else:
                lineup_ids = _coerce_lineup_ids(game.get(f"{side}_lineup_ids"))
                rates = _lineup_player_rates_from_history(
                    history,
                    lineup_ids,
                    game.get(f"{opp}_sp_hand"),
                    pd.Timestamp(game["game_date"]),
                    lookback_days,
                    pa_limit,
                )
                agg = _aggregate_lineup_rates(rates, fallback_xwoba, fallback_barrel)
            for feature, value in agg.items():
                row[f"{side}_{feature}"] = value
        row["lineup_features_missing"] = int(missing)
        rows.append(row)

    return games.merge(pd.DataFrame(rows), on="game_pk", how="left")


def build_posted_lineup_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    lookback_days: int = 30,
    pa_limit: int = 100,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Posted-lineup xwOBA/barrel features for an upcoming slate."""
    from src.data.lineups import load_lineups_for_games
    from src.data.statcast import load_statcast

    if slate.empty:
        return pd.DataFrame()

    slate = slate.copy()
    game_dates = _slate_dates(slate, target_date)
    max_date = max(game_dates.values())
    pitches = load_statcast(max_date - timedelta(days=lookback_days), max_date - timedelta(days=1), raw_dir=raw_dir)
    history = _prepare_lineup_history(pitches)
    lineups = load_lineups_for_games(slate["game_pk"], raw_dir=raw_dir, historical=False)
    slate = slate.merge(lineups, on="game_pk", how="left", suffixes=("", "_lineup"))

    rows = []
    for game_idx, game in slate.iterrows():
        row: dict = {"game_pk": game["game_pk"]}
        missing = False
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            status = game.get(f"{side}_lineup_status")
            if status != "confirmed":
                missing = True
            lineup_ids = _coerce_lineup_ids(game.get(f"{side}_lineup_ids"))
            rates = _lineup_player_rates_from_history(
                history,
                lineup_ids,
                game.get(f"{opp}_sp_hand"),
                pd.Timestamp(game_dates[game_idx]),
                lookback_days,
                pa_limit,
            )
            agg = _aggregate_lineup_rates(rates, float("nan"), float("nan"))
            for feature, value in agg.items():
                row[f"{side}_{feature}"] = value
        row["lineup_features_missing"] = int(missing)
        rows.append(row)
    return pd.DataFrame(rows)


def join_bullpen_features(
    games: pd.DataFrame,
    raw_dir: Path,
    n_games: int = 14,
    fatigue_days: int = 3,
) -> pd.DataFrame:
    """Join rolling team bullpen quality and recent workload to games."""
    from src.data.statcast import (
        _BULLPEN_STAT_COLS,
        aggregate_team_game_bullpen,
        load_statcast,
    )

    games = games.copy()
    out_cols = [
        f"{side}_{col}_l{n_games}"
        for side in ("home", "away")
        for col in _BULLPEN_STAT_COLS
    ]
    workload_days = [1, 2, fatigue_days]
    out_cols += [
        f"{side}_bullpen_pitches_l{days}d"
        for side in ("home", "away")
        for days in workload_days
    ] + [
        f"{side}_bullpen_games_l{days}d"
        for side in ("home", "away")
        for days in workload_days
    ] + [
        f"{side}_bullpen_back_to_back_l2d"
        for side in ("home", "away")
    ] + [
        f"{side}_bullpen_heavy_work_l2d"
        for side in ("home", "away")
    ]

    if games.empty:
        return games

    games["game_date"] = pd.to_datetime(games["game_date"])
    gmin = games["game_date"].min()
    gmax = games["game_date"].max()
    pitches = load_statcast(gmin.date(), gmax.date(), raw_dir=raw_dir)
    pen = aggregate_team_game_bullpen(pitches)
    if pen.empty:
        for col in out_cols:
            games[col] = float("nan")
        return games

    id_lookup = games[["game_pk", "home_team_id", "away_team_id"]].drop_duplicates("game_pk")
    pen = pen.merge(id_lookup, on="game_pk", how="left")
    pen["team_id"] = pen.apply(
        lambda r: r["home_team_id"] if r["side"] == "home" else r["away_team_id"],
        axis=1,
    )
    pen = pen.dropna(subset=["team_id"])
    pen["team_id"] = pen["team_id"].astype(int)
    pen["game_date"] = pd.to_datetime(pen["game_date"])
    pen = pen.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    for col in _BULLPEN_STAT_COLS:
        pen[f"{col}_l{n_games}"] = (
            pen.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=n_games, min_periods=1).mean())
        )

    for days in workload_days:
        pen[f"bullpen_pitches_l{days}d"] = float("nan")
        pen[f"bullpen_games_l{days}d"] = float("nan")
    for _, grp in pen.groupby("team_id"):
        grp = grp.sort_values("game_date")
        for row_idx, row in grp.iterrows():
            for days in workload_days:
                start = row["game_date"] - pd.Timedelta(days=days)
                prior = grp[(grp["game_date"] < row["game_date"]) & (grp["game_date"] >= start)]
                pen.loc[row_idx, f"bullpen_pitches_l{days}d"] = prior["bullpen_pitches"].sum()
                pen.loc[row_idx, f"bullpen_games_l{days}d"] = len(prior)
    pen["bullpen_back_to_back_l2d"] = (pen["bullpen_games_l2d"] >= 2).astype("int8")
    pen["bullpen_heavy_work_l2d"] = (pen["bullpen_pitches_l2d"] >= 75).astype("int8")

    roll_cols = [f"{col}_l{n_games}" for col in _BULLPEN_STAT_COLS]
    fatigue_cols = [
        f"bullpen_pitches_l{days}d"
        for days in workload_days
    ] + [
        f"bullpen_games_l{days}d"
        for days in workload_days
    ] + ["bullpen_back_to_back_l2d", "bullpen_heavy_work_l2d"]
    lookup = pen[["team_id", "game_date"] + roll_cols + fatigue_cols].sort_values("game_date")

    for side in ("home", "away"):
        team_col = f"{side}_team_id"
        join_left = (
            games[["game_pk", "game_date", team_col]]
            .dropna(subset=[team_col])
            .assign(**{team_col: lambda d: d[team_col].astype(int)})
            .sort_values("game_date")
        )
        merged = pd.merge_asof(
            join_left,
            lookup.rename(columns={"team_id": team_col}),
            on="game_date",
            by=team_col,
            direction="backward",
        )
        rename = {c: f"{side}_{c}" for c in roll_cols + fatigue_cols}
        merged = merged.rename(columns=rename)
        games = games.merge(merged[["game_pk"] + list(rename.values())], on="game_pk", how="left")

    return games


def build_bullpen_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
    n_games: int = 14,
    fatigue_days: int = 3,
    lookback_days: int = 60,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Bullpen rolling quality and workload features for an upcoming slate."""
    from src.data.statcast import (
        _BULLPEN_STAT_COLS,
        aggregate_team_game_bullpen,
        load_statcast,
    )

    if slate.empty:
        return pd.DataFrame()

    slate = slate.copy()
    game_dates = _slate_dates(slate, target_date)
    max_date = max(game_dates.values())
    start_date = max_date - timedelta(days=lookback_days)
    pitches = load_statcast(start_date, max_date - timedelta(days=1), raw_dir=raw_dir)
    pen = aggregate_team_game_bullpen(pitches)

    if pen.empty:
        rows = []
        for _, game in slate.iterrows():
            row = {"game_pk": game["game_pk"]}
            for side in ("home", "away"):
                for col in _BULLPEN_STAT_COLS:
                    row[f"{side}_{col}_l{n_games}"] = float("nan")
                for days in (1, 2, fatigue_days):
                    row[f"{side}_bullpen_pitches_l{days}d"] = float("nan")
                    row[f"{side}_bullpen_games_l{days}d"] = float("nan")
                row[f"{side}_bullpen_back_to_back_l2d"] = float("nan")
                row[f"{side}_bullpen_heavy_work_l2d"] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    hist_games = load_processed_games(processed_dir, start_date.year - 1, max_date.year)
    id_lookup = hist_games[["game_pk", "home_team_id", "away_team_id"]].drop_duplicates("game_pk")
    pen = pen.merge(id_lookup, on="game_pk", how="left")
    pen["team_id"] = pen.apply(
        lambda r: r["home_team_id"] if r["side"] == "home" else r["away_team_id"],
        axis=1,
    )
    pen = pen.dropna(subset=["team_id"])
    pen["team_id"] = pen["team_id"].astype(int)
    pen["game_date"] = pd.to_datetime(pen["game_date"])
    pen = pen.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    rows = []
    for game_idx, game in slate.iterrows():
        target_ts = pd.Timestamp(game_dates[game_idx])
        row: dict = {"game_pk": game["game_pk"]}
        for side in ("home", "away"):
            team_id_raw = game.get(f"{side}_team_id")
            if team_id_raw is None or pd.isna(team_id_raw):
                prior = pd.DataFrame()
            else:
                prior = pen[
                    (pen["team_id"] == int(team_id_raw))
                    & (pen["game_date"] < target_ts)
                ].sort_values("game_date")
            recent = prior.tail(n_games)
            for col in _BULLPEN_STAT_COLS:
                row[f"{side}_{col}_l{n_games}"] = (
                    recent[col].mean() if not recent.empty and col in recent else float("nan")
                )
            fatigue_start = target_ts - pd.Timedelta(days=fatigue_days)
            fatigue = prior[prior["game_date"] >= fatigue_start]
            for days in (1, 2, fatigue_days):
                fatigue_start = target_ts - pd.Timedelta(days=days)
                fatigue = prior[prior["game_date"] >= fatigue_start]
                pitches = fatigue["bullpen_pitches"].sum() if not fatigue.empty else 0
                row[f"{side}_bullpen_pitches_l{days}d"] = pitches
                row[f"{side}_bullpen_games_l{days}d"] = len(fatigue)
            row[f"{side}_bullpen_back_to_back_l2d"] = int(
                row[f"{side}_bullpen_games_l2d"] >= 2
            )
            row[f"{side}_bullpen_heavy_work_l2d"] = int(
                row[f"{side}_bullpen_pitches_l2d"] >= 75
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _empty_pitch_quality_row(game_pk: int) -> dict:
    row: dict = {"game_pk": game_pk}
    for side in ("home", "away"):
        for col in ("rv_per_100", "xwoba_arsenal", "whiff_arsenal"):
            row[f"{side}_sp_{col}"] = float("nan")
            row[f"{side}_bp_{col}_weighted"] = float("nan")
        row[f"{side}_sp_pitch_quality_missing"] = 1
        row[f"{side}_bp_pitch_quality_missing"] = 1
    return row


def join_pitch_quality_features(games: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Join FanGraphs Stuff+/Location+/Pitching+ for starters and bullpens."""
    from src.data.pitch_quality import (
        QUALITY_COLS,
        bullpen_pitch_quality_by_team,
        fetch_pitch_quality,
    )

    if games.empty:
        return games

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    seasons = sorted(games["game_date"].dt.year.unique())
    quality_by_season = {
        int(season): fetch_pitch_quality(int(season), raw_dir=raw_dir)
        for season in seasons
    }
    bullpen_by_season = {
        int(season): bullpen_pitch_quality_by_team(quality, raw_dir=raw_dir)
        for season, quality in quality_by_season.items()
    }

    rows = []
    for _, game in games.iterrows():
        season = int(pd.Timestamp(game["game_date"]).year)
        season_quality = quality_by_season.get(season, pd.DataFrame())
        bp_quality = bullpen_by_season.get(season, pd.DataFrame())
        row = _empty_pitch_quality_row(int(game["game_pk"]))
        for side in ("home", "away"):
            sp_id = game.get(f"{side}_sp_id")
            if pd.notna(sp_id) and not season_quality.empty:
                match = season_quality[season_quality["player_id"] == int(sp_id)]
                if not match.empty:
                    for col in QUALITY_COLS:
                        row[f"{side}_sp_{col}"] = match.iloc[0].get(col, float("nan"))
                    row[f"{side}_sp_pitch_quality_missing"] = int(
                        match[QUALITY_COLS].iloc[0].isna().all()
                    )

            team_id = game.get(f"{side}_team_id")
            if pd.notna(team_id) and not bp_quality.empty:
                match = bp_quality[bp_quality["team_id"] == int(team_id)]
                if not match.empty:
                    for col in QUALITY_COLS:
                        row[f"{side}_bp_{col}_weighted"] = match.iloc[0].get(
                            f"bp_{col}_weighted", float("nan")
                        )
                    row[f"{side}_bp_pitch_quality_missing"] = int(
                        match[[f"bp_{c}_weighted" for c in QUALITY_COLS]].iloc[0].isna().all()
                    )
        rows.append(row)

    return games.merge(pd.DataFrame(rows), on="game_pk", how="left")


def build_pitch_quality_prediction_features(
    slate: pd.DataFrame,
    raw_dir: Path,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Baseball Savant pitch-arsenal quality features for an upcoming slate."""
    if slate.empty:
        return pd.DataFrame()

    game_dates = _slate_dates(slate, target_date)
    # Reuse the training join by passing the slate-shaped frame through.
    frame = slate.copy()
    frame["game_date"] = [game_dates[idx] for idx in slate.index]
    features = join_pitch_quality_features(frame, raw_dir)
    keep = [
        c for c in features.columns
        if c == "game_pk"
        or "_pitch_quality" in c
        or "_rv_per_100" in c
        or "_xwoba_arsenal" in c
        or "_whiff_arsenal" in c
    ]
    return features[keep]


def _join_park_factors(
    games: pd.DataFrame,
    raw_dir: Path | None,
    processed_dir: Path,
) -> pd.DataFrame:
    """Join prior-year FanGraphs park factors (pf_runs, pf_hr) to games by venue_id.

    Uses Y-1 factors for each game in year Y to avoid in-season leakage
    (FanGraphs season factors are computed from that season's games).
    Adds NaN columns gracefully when raw_dir is None or data is missing.
    """
    from src.data.park_factors import fetch_park_factors

    games = games.copy()
    if raw_dir is None or "venue_id" not in games.columns:
        games["pf_runs"] = float("nan")
        games["pf_hr"] = float("nan")
        return games

    games["game_date"] = pd.to_datetime(games["game_date"])
    years = sorted(games["game_date"].dt.year.unique())

    pf_chunks = []
    for year in years:
        try:
            pf = fetch_park_factors(year - 1, raw_dir=raw_dir, processed_dir=processed_dir)
        except Exception as exc:
            logger.warning("park factors unavailable for %d: %s", year - 1, exc)
            pf = pd.DataFrame()
        if not pf.empty and "pf_runs" in pf.columns:
            chunk = pf[["venue_id", "pf_runs", "pf_hr"]].copy()
            chunk["_game_year"] = year
            pf_chunks.append(chunk)

    if not pf_chunks:
        logger.warning("no park factors found; pf_runs/pf_hr will be NaN")
        games["pf_runs"] = float("nan")
        games["pf_hr"] = float("nan")
        return games

    pf_all = pd.concat(pf_chunks, ignore_index=True)
    pf_all = pf_all.dropna(subset=["venue_id"])
    pf_all["venue_id"] = pd.to_numeric(pf_all["venue_id"], errors="coerce").astype("Int64")
    pf_all = pf_all.drop_duplicates(subset=["venue_id", "_game_year"])

    games["venue_id"] = pd.to_numeric(games["venue_id"], errors="coerce").astype("Int64")
    games["_game_year"] = games["game_date"].dt.year

    games = games.merge(pf_all, left_on=["venue_id", "_game_year"], right_on=["venue_id", "_game_year"], how="left")
    games = games.drop(columns=["_game_year"])
    logger.info("joined park factors; %d games have pf_runs", games["pf_runs"].notna().sum())
    return games


def build_training_set(
    processed_dir: Path,
    start_year: int = 2018,
    end_year: int = 2025,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the training matrix from cached processed data.

    Returns a DataFrame with columns:
        game_pk, game_date,
        home_*, away_*,    (team rolling features)
        home_sp_*, away_sp_*  (pitcher Statcast features, if raw_dir provided)
        venue_id,
        target_home_win, target_total_runs

    Sorted by game_date ascending. No NaN handling here; that's a model concern.

    Args:
        raw_dir: if provided, loads Statcast from data/raw/statcast/ and joins
                 rolling pitcher features.  Leave None to reproduce the
                 team-only baseline.
    """
    logger.info("building training set for %d-%d", start_year, end_year)

    # Load processed games
    games = load_processed_games(processed_dir, start_year, end_year)
    if games.empty:
        raise ValueError("No game data found")

    logger.info("loaded %d games", len(games))

    # Compute team rolling features
    team_features = compute_team_rolling_features(games)

    # Season-to-date and home/away split features (require is_home + won columns)
    std_feats = season_to_date(team_features)
    team_features = team_features.merge(
        std_feats[["game_pk", "team_id", "runs_per_game_std", "ra_per_game_std"]],
        on=["game_pk", "team_id"], how="left",
    )
    ha_feats = home_away_split(team_features)
    team_features = team_features.merge(
        ha_feats[["game_pk", "team_id", "win_pct_home_std", "win_pct_away_std"]],
        on=["game_pk", "team_id"], how="left",
    )
    rest_feats = days_rest(team_features)
    team_features = team_features.merge(
        rest_feats[["game_pk", "team_id", "days_rest"]],
        on=["game_pk", "team_id"], how="left",
    )

    # Pivot to game-level with home/away features
    home_features = team_features[team_features["is_home"] == 1].copy()
    home_cols = [c for c in home_features.columns if c not in [
        "game_pk", "game_date", "team_id", "runs_for", "runs_against",
        "venue_id", "is_home", "won", "run_diff"
    ]]
    home_features = home_features.rename(columns={c: f"home_{c}" for c in home_cols})
    home_features = home_features[["game_pk"] + [f"home_{c}" for c in home_cols]]

    away_features = team_features[team_features["is_home"] == 0].copy()
    away_features = away_features.rename(columns={c: f"away_{c}" for c in home_cols})
    away_features = away_features[["game_pk"] + [f"away_{c}" for c in home_cols]]

    # Join to games
    result = games.merge(home_features, on="game_pk", how="left")
    result = result.merge(away_features, on="game_pk", how="left")

    # Join park factors (prior-year, keyed by venue_id)
    result = _join_park_factors(result, raw_dir, processed_dir)

    # Optionally join pitcher features from Stats API gamelogs (traditional: ERA/WHIP/K/BB)
    if raw_dir is not None:
        result = join_gamelog_pitcher_features(result, raw_dir)

    # Optionally join SP Statcast features (xwOBA against, whiff, barrel, platoon splits)
    # Runs after gamelog join so home_sp_id/away_sp_id are already present and reused.
    if raw_dir is not None:
        result = join_pitcher_features(result, raw_dir)

    # Optionally join team batting Statcast features (xwOBA offense, barrel rate)
    if raw_dir is not None:
        result = join_team_statcast_features(result, raw_dir)

    # Optionally join posted-lineup weighted batter features
    if raw_dir is not None:
        result = join_posted_lineup_features(result, raw_dir)

    # Optionally join bullpen rolling quality and recent workload
    if raw_dir is not None:
        result = join_bullpen_features(result, raw_dir)

    # Optionally join Baseball Savant pitch-arsenal metrics for starters and bullpens
    if raw_dir is not None:
        result = join_pitch_quality_features(result, raw_dir)

    # Optionally join lineup matchup features (batter splits vs SP hand + BvP)
    if raw_dir is not None:
        result = join_lineup_matchup_features(result, raw_dir)

    # Sort by date
    result = result.sort_values("game_date").reset_index(drop=True)

    logger.info("built training set with %d features", len(result.columns))

    return result


def _slate_dates(slate: pd.DataFrame, target_date: date | None = None) -> dict[int, date]:
    """Return a per-row as-of date for a slate."""
    if target_date is not None:
        return {idx: target_date for idx in slate.index}

    source_col = "official_date" if "official_date" in slate.columns else "game_date"
    parsed = pd.to_datetime(slate[source_col], errors="coerce")
    fallback = date.today()
    return {
        idx: (value.date() if not pd.isna(value) else fallback)
        for idx, value in parsed.items()
    }


def _add_pitcher_season_to_date(row: dict, side: str, starts: pd.DataFrame) -> None:
    if starts.empty:
        for col in ["era", "whip", "k_per_9", "bb_per_9", "k_minus_bb_pct", "hr_per_9"]:
            row[f"{side}_sp_{col}_std"] = float("nan")
        row[f"{side}_sp_ip_total_std"] = float("nan")
        return

    ip = starts["ip"].sum()
    bf = starts["bf"].sum()
    safe_ip = max(ip, 0.01)
    safe_bf = max(bf, 1)
    row[f"{side}_sp_era_std"] = starts["er"].sum() * 9.0 / safe_ip
    row[f"{side}_sp_whip_std"] = (starts["h"].sum() + starts["bb"].sum()) / safe_ip
    row[f"{side}_sp_k_per_9_std"] = starts["k"].sum() * 9.0 / safe_ip
    row[f"{side}_sp_bb_per_9_std"] = starts["bb"].sum() * 9.0 / safe_ip
    row[f"{side}_sp_k_minus_bb_pct_std"] = (starts["k"].sum() - starts["bb"].sum()) / safe_bf
    row[f"{side}_sp_hr_per_9_std"] = starts["hr"].sum() * 9.0 / safe_ip
    row[f"{side}_sp_ip_total_std"] = ip


def _team_snapshot(team_games: pd.DataFrame, team_id, before_date: date) -> dict[str, float]:
    if pd.isna(team_id):
        return {}

    prior = team_games[
        (team_games["team_id"] == int(team_id))
        & (team_games["game_date"] < pd.Timestamp(before_date))
    ].sort_values("game_date")

    out: dict[str, float] = {}
    for n in [5, 10, 20]:
        recent = prior.tail(n)
        out[f"wins_l{n}"] = recent["won"].sum() if not recent.empty else float("nan")
        out[f"run_diff_l{n}"] = recent["run_diff"].sum() if not recent.empty else float("nan")
        out[f"avg_runs_for_l{n}"] = recent["runs_for"].mean() if not recent.empty else float("nan")
        out[f"avg_runs_against_l{n}"] = (
            recent["runs_against"].mean() if not recent.empty else float("nan")
        )
        out[f"win_pct_l{n}"] = (
            out[f"wins_l{n}"] / n if not pd.isna(out[f"wins_l{n}"]) else float("nan")
        )

    # Days since last game
    out["days_rest"] = (
        (pd.Timestamp(before_date) - prior.iloc[-1]["game_date"]).days
        if not prior.empty else float("nan")
    )

    # Season-to-date runs/game (current season only)
    season = prior[prior["game_date"].dt.year == before_date.year]
    out["runs_per_game_std"] = season["runs_for"].mean() if not season.empty else float("nan")
    out["ra_per_game_std"] = season["runs_against"].mean() if not season.empty else float("nan")

    # Home and road win % this season
    home_g = season[season["is_home"] == 1] if "is_home" in season.columns else pd.DataFrame()
    away_g = season[season["is_home"] == 0] if "is_home" in season.columns else pd.DataFrame()
    out["win_pct_home_std"] = home_g["won"].mean() if not home_g.empty else float("nan")
    out["win_pct_away_std"] = away_g["won"].mean() if not away_g.empty else float("nan")

    return out


def build_prediction_input(
    slate: pd.DataFrame,
    processed_dir: Path,
    target_date: date | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the same feature columns for today's slate (no targets).

    Uses the most recent team stats from processed games to compute rolling features.
    """
    # Load the most recent processed games to get team stats
    # We'll use the last few games of each team to compute rolling features
    slate = slate.copy()
    game_dates = _slate_dates(slate, target_date)
    min_date = min(game_dates.values())
    max_date = max(game_dates.values())
    games = load_processed_games(processed_dir, min_date.year - 1, max_date.year)
    if games.empty:
        raise ValueError("No historical game data available")

    team_games = compute_team_rolling_features(games)

    # Ensure team IDs in slate are integers for matching
    slate["home_team_id"] = pd.to_numeric(slate["home_team_id"], errors="coerce").astype("Int64")
    slate["away_team_id"] = pd.to_numeric(slate["away_team_id"], errors="coerce").astype("Int64")

    # Build feature rows for each game in the slate
    rows = []
    for game_idx, game in slate.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        game_date = game_dates[game_idx]

        row = {"game_pk": game["game_pk"], "game_date": game["game_date"]}
        for col, value in _team_snapshot(team_games, home_id, game_date).items():
            row[f"home_{col}"] = value
        for col, value in _team_snapshot(team_games, away_id, game_date).items():
            row[f"away_{col}"] = value
        row["venue_id"] = game.get("venue_id")
        rows.append(row)

    result = pd.DataFrame(rows)

    # Join park factors for the slate's year
    result = _join_park_factors(result, raw_dir, processed_dir)

    # Convert available feature columns to numeric; model inference adds any
    # columns not produced by this lightweight team/schedule stage.
    feature_cols = [c for c in result.columns if c not in ["game_pk", "game_date"]]
    for col in feature_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2018, help="First season (inclusive)")
    parser.add_argument("--end", type=int, default=2024, help="Last season (inclusive)")
    parser.add_argument(
        "--output", type=str, help="Output parquet path (default: data/processed/training.parquet)"
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    df = build_training_set(PROCESSED_DIR, args.start, args.end)

    output_path = args.output or (PROCESSED_DIR / "training.parquet")
    df.to_parquet(output_path, index=False)
    logger.info("saved training set to %s", output_path)


if __name__ == "__main__":
    main()

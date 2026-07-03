"""Assemble all advanced game-level features for a single date's slate.

Pulls together:
  - Schedule + probable starters + lineups  (MLB Stats API)
  - Statcast pitcher features: xwOBA against, whiff rate, barrel rate,
    platoon splits — rolling 3-start average (Baseball Savant / pybaseball)
  - Lineup-weighted batter features vs opposing starter's handedness
  - BvP history (Statcast 2015+), PA >= 20, weighted by min(PA, 60) / 60
  - Park factors (FanGraphs, with game-log fallback)

Outputs one row per game to:
    data/processed/game_features_{YYYY-MM-DD}.parquet

Usage:
    python scripts/fetch_game_features.py
    python scripts/fetch_game_features.py --date 2026-04-15
    python scripts/fetch_game_features.py --date 2026-04-15 --force
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from src.data.park_factors import fetch_park_factors
from src.data.statcast import (
    aggregate_batter_season,
    aggregate_bvp,
    aggregate_pitcher_starts,
    load_statcast,
)
from src.data.update import fetch_today_slate, today_in_schedule_timezone
from src.features.batting import lineup_vs_starter
from src.features.bvp import compute_lineup_bvp
from src.features.pitcher import days_rest, rolling_pitcher_stats
from src.utils.logging import configure_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# How far back to look for pitcher rolling stats (covers 3 starts + buffer
# for skipped turns and off-days).
_PITCHER_LOOKBACK_DAYS = 60

# How far back to look for batter season stats.
_BATTER_SEASON_LOOKBACK_DAYS = 180

# BvP history: pull Statcast from this many years back.
_BVP_LOOKBACK_YEARS = 5


def _build_pitcher_features(
    pitcher_id: int | None,
    pitcher_starts: pd.DataFrame,
    target_date: pd.Timestamp,
    prefix: str,
) -> dict:
    """Return rolling-3-start features for one probable starter."""
    if pitcher_id is None:
        return {}
    stats = rolling_pitcher_stats(pitcher_starts, pitcher_id, target_date, n_starts=3)
    rest = days_rest(pitcher_starts, pitcher_id, target_date)
    out = {f"{prefix}_{k}": v for k, v in stats.items() if k != "sp_starts_available"}
    out[f"{prefix}_starts_available"] = stats["sp_starts_available"]
    out[f"{prefix}_days_rest"] = rest
    return out


def _build_lineup_features(
    lineup_ids: list[int] | None,
    batter_season: pd.DataFrame,
    bvp_table: pd.DataFrame,
    opposing_sp_id: int | None,
    opposing_sp_hand: str | None,
    prefix: str,
) -> dict:
    """Return lineup-weighted and BvP features for one team's batting order."""
    out: dict = {}

    hand = opposing_sp_hand or "R"
    ids = lineup_ids or []

    batting = lineup_vs_starter(ids, batter_season, hand)
    out.update({f"{prefix}_{k}": v for k, v in batting.items()})

    bvp = compute_lineup_bvp(ids, opposing_sp_id, bvp_table)
    out.update({f"{prefix}_{k}": v for k, v in bvp.items()})

    return out


def fetch_game_features(target_date: date, force: bool = False) -> pd.DataFrame:
    """Build the full advanced feature row for every game on target_date.

    Returns a DataFrame with one row per game_pk.
    """
    logger.info("building game features for %s", target_date)

    # ------------------------------------------------------------------
    # 1. Slate: schedule + probables + lineups
    # ------------------------------------------------------------------
    slate = fetch_today_slate(target_date)
    if slate.empty:
        logger.warning("no games on %s", target_date)
        return pd.DataFrame()
    logger.info("slate: %d games", len(slate))

    # ------------------------------------------------------------------
    # 2. Statcast: load recent pitches for pitcher stats and batter stats
    # ------------------------------------------------------------------
    target_ts = pd.Timestamp(target_date)
    pitcher_start_date = target_date - timedelta(days=_PITCHER_LOOKBACK_DAYS)
    batter_start_date = target_date - timedelta(days=_BATTER_SEASON_LOOKBACK_DAYS)

    logger.info("loading statcast pitches (%s → %s)", pitcher_start_date, target_date - timedelta(days=1))
    recent_pitches = load_statcast(
        pitcher_start_date,
        target_date - timedelta(days=1),  # strictly before today
        raw_dir=RAW_DIR,
        force=force,
    )

    if not recent_pitches.empty:
        pitcher_starts = aggregate_pitcher_starts(recent_pitches)
        logger.info("pitcher starts aggregated: %d rows", len(pitcher_starts))
    else:
        pitcher_starts = pd.DataFrame()
        logger.warning("no statcast data found; pitcher features will be NaN")

    # For batter season stats, pull a longer window if we have more data cached.
    logger.info("loading statcast for batter season stats (%s → %s)", batter_start_date, target_date - timedelta(days=1))
    season_pitches = load_statcast(
        batter_start_date,
        target_date - timedelta(days=1),
        raw_dir=RAW_DIR,
        force=force,
    )
    batter_season = aggregate_batter_season(season_pitches, target_date.year) if not season_pitches.empty else pd.DataFrame()
    logger.info("batter season stats: %d batters", len(batter_season))

    # ------------------------------------------------------------------
    # 3. BvP: aggregate multi-year history
    # ------------------------------------------------------------------
    bvp_start = date(target_date.year - _BVP_LOOKBACK_YEARS, 1, 1)
    logger.info("loading statcast for BvP (%s → %s)", bvp_start, target_date - timedelta(days=1))
    bvp_pitches = load_statcast(
        bvp_start,
        target_date - timedelta(days=1),
        raw_dir=RAW_DIR,
        force=force,
    )
    bvp_table = aggregate_bvp(bvp_pitches) if not bvp_pitches.empty else pd.DataFrame()
    logger.info("BvP table: %d (batter, pitcher) pairs", len(bvp_table))

    # ------------------------------------------------------------------
    # 4. Park factors
    # ------------------------------------------------------------------
    park_factors = fetch_park_factors(
        target_date.year,
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        force=force,
    )

    # ------------------------------------------------------------------
    # 5. Pitcher handedness lookup (for lineup features)
    # ------------------------------------------------------------------
    sp_hand: dict[int, str] = {}
    if not pitcher_starts.empty and "pitcher" in pitcher_starts.columns and "p_throws" in pitcher_starts.columns:
        sp_hand = (
            pitcher_starts.dropna(subset=["p_throws"])
            .groupby("pitcher")["p_throws"]
            .agg(lambda x: x.mode()[0])
            .to_dict()
        )

    # ------------------------------------------------------------------
    # 6. Assemble one row per game
    # ------------------------------------------------------------------
    rows = []
    for _, game in slate.iterrows():
        home_sp_id = game.get("home_sp_id")
        away_sp_id = game.get("away_sp_id")
        home_lineup = game.get("home_lineup_ids")
        away_lineup = game.get("away_lineup_ids")
        venue_id = game.get("venue_id")

        home_sp_hand = sp_hand.get(home_sp_id) if home_sp_id else None
        away_sp_hand = sp_hand.get(away_sp_id) if away_sp_id else None

        row: dict = {
            "game_pk": game["game_pk"],
            "game_date": game["game_date"],
            "home_team_id": game["home_team_id"],
            "away_team_id": game["away_team_id"],
            "home_sp_id": home_sp_id,
            "away_sp_id": away_sp_id,
            "venue_id": venue_id,
        }

        # Home SP rolling stats
        row.update(
            _build_pitcher_features(home_sp_id, pitcher_starts, target_ts, "home_sp")
        )
        # Away SP rolling stats
        row.update(
            _build_pitcher_features(away_sp_id, pitcher_starts, target_ts, "away_sp")
        )

        # Home lineup vs away SP
        row.update(
            _build_lineup_features(
                home_lineup, batter_season, bvp_table,
                opposing_sp_id=away_sp_id,
                opposing_sp_hand=away_sp_hand,
                prefix="home_lineup",
            )
        )
        # Away lineup vs home SP
        row.update(
            _build_lineup_features(
                away_lineup, batter_season, bvp_table,
                opposing_sp_id=home_sp_id,
                opposing_sp_hand=home_sp_hand,
                prefix="away_lineup",
            )
        )

        # Park factors
        if not park_factors.empty and venue_id is not None:
            pf_row = park_factors[park_factors["venue_id"] == venue_id]
            if not pf_row.empty:
                row["park_factor_runs"] = float(pf_row["pf_runs"].iloc[0])
                row["park_factor_hr"] = float(pf_row["pf_hr"].iloc[0])
            else:
                row["park_factor_runs"] = float("nan")
                row["park_factor_hr"] = float("nan")
        else:
            row["park_factor_runs"] = float("nan")
            row["park_factor_hr"] = float("nan")

        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info("assembled %d feature rows", len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch data even when cached",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()

    df = fetch_game_features(target, force=args.force)
    if df.empty:
        return

    out_path = PROCESSED_DIR / f"game_features_{target.isoformat()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows → %s", len(df), out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

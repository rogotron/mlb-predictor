"""Lineup and offense features.

All functions consume pre-aggregated tables produced by
src.data.statcast — no raw data is fetched here.

Primary entry point for the daily pipeline is lineup_vs_starter(), which
returns a single dict of weighted features for one team's lineup against
a specific opposing starter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def lineup_vs_starter(
    lineup_ids: list[int],
    batter_season: pd.DataFrame,
    pitcher_hand: str,
) -> dict[str, float]:
    """Lineup-weighted Statcast features vs the opposing starter's handedness.

    For each batter in lineup_ids, looks up their season-level xwOBA and
    barrel rate against pitchers of pitcher_hand (R or L).  Returns a
    simple mean across the batters for whom stats are available.

    Args:
        lineup_ids:    ordered list of MLBAM batter IDs (batting-order position).
        batter_season: output of aggregate_batter_season(), all batters.
        pitcher_hand:  'R' or 'L'.

    Returns dict with keys:
        lineup_xwoba_vs_sp      — mean xwOBA vs starter hand
        lineup_barrel_rate_vs_sp — mean barrel rate vs starter hand
        lineup_batters_matched  — how many batters had season data
    """
    hand = pitcher_hand if pitcher_hand in ("R", "L") else "R"
    xwoba_col = f"xwoba_vs_{hand}"
    barrel_col = f"barrel_rate_vs_{hand}"

    nan_row = {
        "lineup_xwoba_vs_sp": float("nan"),
        "lineup_barrel_rate_vs_sp": float("nan"),
        "lineup_batters_matched": 0,
    }

    if not lineup_ids or batter_season.empty:
        return nan_row

    subset = batter_season[batter_season["batter"].isin(lineup_ids)]
    if subset.empty:
        return nan_row

    xwoba_vals = subset[xwoba_col].dropna() if xwoba_col in subset.columns else pd.Series(dtype=float)
    barrel_vals = subset[barrel_col].dropna() if barrel_col in subset.columns else pd.Series(dtype=float)

    return {
        "lineup_xwoba_vs_sp": float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else float("nan"),
        "lineup_barrel_rate_vs_sp": float(barrel_vals.mean()) if len(barrel_vals) > 0 else float("nan"),
        "lineup_batters_matched": len(subset),
    }


def vs_handedness_split(
    team_games: pd.DataFrame,
    batter_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Team OPS / wOBA vs LHP and vs RHP, season to date.

    Placeholder — not yet consumed by the pipeline.
    """
    raise NotImplementedError


def bullpen_usage_l3(bullpen_log: pd.DataFrame) -> pd.DataFrame:
    """Total bullpen innings and high-leverage usage in last 3 days.

    Placeholder — not yet consumed by the pipeline.
    """
    raise NotImplementedError

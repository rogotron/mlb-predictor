"""Starting-pitcher Statcast features.

All functions operate on the pitcher_starts table produced by
src.data.statcast.aggregate_pitcher_starts — one row per (pitcher, game_pk).
No raw data is fetched here.

Strict no-leakage rule: every rolling window is built from starts whose
game_date is strictly before the target game's date.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_pitcher_stats(
    pitcher_starts: pd.DataFrame,
    pitcher_id: int,
    before_date: pd.Timestamp,
    n_starts: int = 3,
) -> dict[str, float]:
    """Rolling n-start Statcast averages for a single pitcher.

    Args:
        pitcher_starts: output of aggregate_pitcher_starts(), all pitchers.
        pitcher_id:     MLBAM pitcher ID.
        before_date:    target game date; only starts strictly before this
                        date are included (no same-day leakage).
        n_starts:       rolling window size (default 3).

    Returns dict with keys:
        xwoba_against_l{n}, whiff_rate_l{n}, barrel_rate_l{n},
        xwoba_against_vs_L_l{n}, xwoba_against_vs_R_l{n},
        whiff_rate_vs_L_l{n}, whiff_rate_vs_R_l{n},
        sp_starts_available   (int — how many starts were in the window)
    """
    suffix = f"_l{n_starts}"
    nan_row = {
        f"xwoba_against{suffix}": float("nan"),
        f"whiff_rate{suffix}": float("nan"),
        f"barrel_rate{suffix}": float("nan"),
        f"xwoba_against_vs_L{suffix}": float("nan"),
        f"xwoba_against_vs_R{suffix}": float("nan"),
        f"whiff_rate_vs_L{suffix}": float("nan"),
        f"whiff_rate_vs_R{suffix}": float("nan"),
        "sp_starts_available": 0,
    }

    if pitcher_starts.empty:
        return nan_row

    mask = (pitcher_starts["pitcher"] == pitcher_id) & (
        pitcher_starts["game_date"] < before_date
    )
    recent = (
        pitcher_starts.loc[mask]
        .sort_values("game_date", ascending=False)
        .head(n_starts)
    )

    if recent.empty:
        return nan_row

    def _mean(col: str) -> float:
        if col not in recent.columns:
            return float("nan")
        return float(recent[col].dropna().mean())

    return {
        f"xwoba_against{suffix}": _mean("xwoba_against"),
        f"whiff_rate{suffix}": _mean("whiff_rate"),
        f"barrel_rate{suffix}": _mean("barrel_rate"),
        f"xwoba_against_vs_L{suffix}": _mean("xwoba_against_vs_L"),
        f"xwoba_against_vs_R{suffix}": _mean("xwoba_against_vs_R"),
        f"whiff_rate_vs_L{suffix}": _mean("whiff_rate_vs_L"),
        f"whiff_rate_vs_R{suffix}": _mean("whiff_rate_vs_R"),
        "sp_starts_available": len(recent),
    }


def days_rest(
    pitcher_starts: pd.DataFrame,
    pitcher_id: int,
    before_date: pd.Timestamp,
) -> int | None:
    """Days since the pitcher's most recent start before before_date.

    Returns None if no prior start is found (e.g. season opener).
    """
    mask = (pitcher_starts["pitcher"] == pitcher_id) & (
        pitcher_starts["game_date"] < before_date
    )
    prior = pitcher_starts.loc[mask].sort_values("game_date", ascending=False)
    if prior.empty:
        return None
    last_date = pd.Timestamp(prior["game_date"].iloc[0])
    return (pd.Timestamp(before_date) - last_date).days

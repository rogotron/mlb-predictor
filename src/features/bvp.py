"""Batter-vs-pitcher (BvP) features.

Source: Statcast pitch data (2015–present) aggregated by
src.data.statcast.aggregate_bvp() into a (batter, pitcher, pa_count,
xwoba_bvp) table.  This covers the active-player history that Retrosheet
would otherwise supply for modern matchups.

Weighting scheme (per user spec):
  weight = min(PA, 60) / 60
  Only pairs with PA >= min_pa (default 20) contribute.

A lineup's BvP score is the unweighted mean of the weight-adjusted xwOBA
values across all qualifying batter–pitcher pairs in the lineup.  Batters
with no qualifying history are excluded rather than imputed, so the score
reflects only genuine matchup evidence.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_lineup_bvp(
    lineup_ids: list[int],
    pitcher_id: int,
    bvp_table: pd.DataFrame,
    min_pa: int = 20,
) -> dict[str, float]:
    """Lineup-weighted BvP xwOBA vs a specific pitcher.

    Args:
        lineup_ids: ordered list of MLBAM batter IDs.
        pitcher_id: MLBAM pitcher ID.
        bvp_table:  output of aggregate_bvp(), all (batter, pitcher) pairs.
        min_pa:     minimum PAs for a pair to qualify (default 20).

    Returns dict with keys:
        bvp_xwoba          — lineup mean of weight * xwoba_bvp
        bvp_batters_matched — number of batters with qualifying BvP history
    """
    nan_row = {"bvp_xwoba": float("nan"), "bvp_batters_matched": 0}

    if not lineup_ids or bvp_table.empty or pitcher_id is None:
        return nan_row

    pairs = bvp_table[
        (bvp_table["pitcher"] == pitcher_id)
        & (bvp_table["batter"].isin(lineup_ids))
        & (bvp_table["pa_count"] >= min_pa)
    ].copy()

    if pairs.empty:
        return nan_row

    pairs["weight"] = (pairs["pa_count"].clip(upper=60) / 60).astype(float)
    valid = pairs.dropna(subset=["xwoba_bvp"])
    if valid.empty:
        return nan_row

    # weighted mean: sum(weight * xwoba) / sum(weight)
    weighted_xwoba = float(
        (valid["weight"] * valid["xwoba_bvp"]).sum() / valid["weight"].sum()
    )

    return {
        "bvp_xwoba": weighted_xwoba,
        "bvp_batters_matched": len(valid),
    }

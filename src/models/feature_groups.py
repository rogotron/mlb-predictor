"""Shared factor-group mapping for feature-importance transparency.

One source of truth for how the model's features roll up into human-readable
factor groups by data source. Used by training (to emit the importance
artifact) and by the dashboard assembler (to show real importances). Keeping
it here — not in the backend — lets both sides import it without a layering
dependency.

Each feature is assigned to the FIRST group whose any substring it contains,
so group importances are a non-overlapping partition of total model gain.
Order matters: availability, pitch-arsenal and lineup rules come before the
broad "sp_"/lineup rules so specific groups claim their features first. Groups
whose features are absent from a model contribute zero gain and drop out — this
is how excluded lineup/BvP and pitch-arsenal groups stay off pregame_safe.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# (label, data source, feature-name substrings)
FACTOR_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Rest & Availability", "Schedule / rest", [
        "days_rest", "sp_season_starts", "sp_recent_starts",
        "sp_short_history", "sp_unknown",
    ]),
    ("Pitch Quality", "Pitch-arsenal (excluded from pregame_safe)", [
        "_arsenal", "_rv_per_100", "pitch_quality",
    ]),
    ("Lineup Matchup", "Lineup Statcast (excluded from pregame_safe)", [
        "lineup_xwoba_vs_sp", "bvp_",
    ]),
    ("Posted Lineup Quality", "Posted lineup Statcast (excluded from pregame_safe)", [
        "lineup_xwoba_vs_hand", "lineup_xwoba_weighted", "lineup_xwoba_top5",
        "lineup_barrel", "lineup_features_missing",
    ]),
    ("Starting Pitcher Quality", "SP rolling + Statcast", ["sp_"]),
    ("Bullpen Quality + Load", "Bullpen Statcast + workload", ["bullpen_"]),
    ("Recent Form (L10/L20)", "Team rolling form", [
        "wins_l", "run_diff_l", "xwoba_off_l", "barrel_rate_off_l", "win_pct_l",
    ]),
    ("Run Production", "Season-to-date team rates", [
        "avg_runs_for", "avg_runs_against", "runs_per_game", "ra_per_game",
    ]),
    ("Home/Away Record", "Season-to-date team rates", [
        "win_pct_home_std", "win_pct_away_std",
    ]),
    ("Park Factors", "FanGraphs / game-log fallback", ["pf_runs", "pf_hr"]),
]

OTHER_GROUP = ("Other model features", "Mixed")


def assign_group(feature_name: str) -> int | None:
    """Return the index of the first factor group matching a feature, or None."""
    for gi, (_name, _src, patterns) in enumerate(FACTOR_GROUPS):
        if any(p in feature_name for p in patterns):
            return gi
    return None


def group_name_and_source(feature_name: str) -> tuple[str, str]:
    """Return the (group label, source) a feature belongs to."""
    gi = assign_group(feature_name)
    if gi is None:
        return OTHER_GROUP
    name, source, _ = FACTOR_GROUPS[gi]
    return name, source


def group_importances(
    feature_names: Sequence[str],
    importances: Iterable[float],
) -> tuple[list[dict], float]:
    """Partition feature gains across factor groups.

    Returns (groups, total_gain) with groups sorted by gain descending, each
    entry a dict with name, source, gain, and pct (share of total gain). Only
    groups with non-zero gain are returned; unmatched features fall into
    "Other model features".
    """
    gains: dict[int, float] = {}
    other = 0.0
    for name, gain in zip(feature_names, importances, strict=False):
        gain = float(gain)
        gi = assign_group(name)
        if gi is None:
            other += gain
        else:
            gains[gi] = gains.get(gi, 0.0) + gain

    total = sum(gains.values()) + other
    if total <= 0:
        total = 1.0

    groups: list[dict] = []
    for gi, gain in gains.items():
        if gain <= 0:
            continue
        gname, gsrc, _ = FACTOR_GROUPS[gi]
        groups.append({
            "name": gname,
            "source": gsrc,
            "gain": gain,
            "pct": round(gain / total * 100, 1),
        })
    if other > 0:
        groups.append({
            "name": OTHER_GROUP[0],
            "source": OTHER_GROUP[1],
            "gain": other,
            "pct": round(other / total * 100, 1),
        })

    groups.sort(key=lambda g: g["gain"], reverse=True)
    return groups, total

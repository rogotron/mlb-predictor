"""Model metadata and feature importance endpoint."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter

from src.models.feature_config import DEFAULT_MODEL_MODE, model_artifact_name
from src.models.predict import load_latest_model
from src.utils.paths import MODEL_DIR

router = APIRouter(prefix="/model", tags=["model"])

# Ordered from most-specific to least-specific so the first match wins.
_GROUP_RULES: list[tuple[str, str]] = [
    # Lineup / BvP
    ("lineup_xwoba_vs_hand", "Posted Lineup Quality"),
    ("lineup_xwoba_weighted", "Posted Lineup Quality"),
    ("lineup_xwoba_top5", "Posted Lineup Quality"),
    ("lineup_barrel_rate", "Posted Lineup Quality"),
    ("lineup_features_missing", "Posted Lineup Quality"),
    ("lineup_xwoba_vs_sp",  "Lineup Matchup"),
    ("bvp_xwoba",           "Lineup Matchup"),
    # Pitch-quality metrics
    ("rv_per_100",          "Pitch Quality"),
    ("xwoba_arsenal",       "Pitch Quality"),
    ("whiff_arsenal",       "Pitch Quality"),
    ("pitch_quality_missing", "Pitch Quality"),
    # Team batting Statcast
    ("xwoba_off_l",         "Team Batting Statcast"),
    ("barrel_rate_off_l",   "Team Batting Statcast"),
    # SP Statcast quality (check before generic _sp_ rules)
    ("sp_xwoba_against",    "SP Statcast Quality"),
    ("sp_whiff_rate",       "SP Statcast Quality"),
    ("sp_barrel_rate",      "SP Statcast Quality"),
    # SP traditional
    ("_sp_", "_std",        "SP Season-to-Date"),
    ("_sp_", "_l3",         "SP Rolling (L3)"),
    # Team rolling windows
    ("_l5",                 "Team Rolling (L5)"),
    ("_l10",                "Team Rolling (L10)"),
    ("_l20",                "Team Rolling (L20)"),
    # Season-to-date team rates
    ("runs_per_game_std",   "Season-to-Date Rates"),
    ("ra_per_game_std",     "Season-to-Date Rates"),
    ("win_pct_home_std",    "Season-to-Date Rates"),
    ("win_pct_away_std",    "Season-to-Date Rates"),
    # Schedule / rest
    ("days_rest",           "Schedule / Rest"),
    # Park factors
    ("pf_",                 "Park Factors"),
]

_GROUP_COLORS = {
    "Lineup Matchup":        "#5a1a5a",
    "Posted Lineup Quality": "#6a3d1a",
    "Pitch Quality":         "#245a6a",
    "Team Batting Statcast": "#1e4d3a",
    "SP Statcast Quality":   "#2a5a1a",
    "SP Season-to-Date":     "#7a5a1a",
    "SP Rolling (L3)":       "#1e4d1e",
    "Team Rolling (L5)":     "#9a1515",
    "Team Rolling (L10)":    "#7a1515",
    "Team Rolling (L20)":    "#5a0f0f",
    "Season-to-Date Rates":  "#1a3264",
    "Schedule / Rest":       "#5a4a38",
    "Park Factors":          "#4a3a1a",
    "Other":                 "#8a7a65",
}

# Maps raw stat token → human label
_STAT_LABELS = {
    "wins":                 "Wins",
    "run_diff":             "Run Diff",
    "avg_runs_for":         "R/G Scored",
    "avg_runs_against":     "R/G Allowed",
    "win_pct":              "Win%",
    "era":                  "ERA",
    "whip":                 "WHIP",
    "k_per_9":              "K/9",
    "bb_per_9":             "BB/9",
    "k_minus_bb_pct":       "K-BB%",
    "hr_per_9":             "HR/9",
    "ip_per_start":         "IP/Start",
    "ip_total":             "IP Total",
    "xwoba_against":        "xwOBA Against",
    "xwoba_against_vs_l":   "xwOBA vs LHB",
    "xwoba_against_vs_r":   "xwOBA vs RHB",
    "whiff_rate":           "Whiff Rate",
    "whiff_rate_vs_l":      "Whiff% vs LHB",
    "whiff_rate_vs_r":      "Whiff% vs RHB",
    "barrel_rate":          "Barrel Rate",
    "xwoba_off":            "xwOBA Offense",
    "barrel_rate_off":      "Barrel Rate Off",
    "lineup_xwoba_vs_sp":   "Lineup xwOBA vs SP",
    "lineup_xwoba_vs_hand_l30": "Lineup xwOBA vs Hand",
    "lineup_xwoba_weighted": "Weighted Lineup xwOBA",
    "lineup_xwoba_top5":    "Top-5 Lineup xwOBA",
    "lineup_barrel_rate_vs_hand_l30": "Lineup Barrel vs Hand",
    "lineup_barrel_rate_weighted": "Weighted Lineup Barrel",
    "lineup_barrel_rate_top5": "Top-5 Lineup Barrel",
    "bvp_xwoba":            "BvP xwOBA",
    "rv_per_100":           "RV/100",
    "xwoba_arsenal":        "Arsenal xwOBA",
    "whiff_arsenal":        "Arsenal Whiff%",
    "days_rest":            "Days Rest",
    "runs_per_game_std":    "R/G (Season)",
    "ra_per_game_std":      "RA/G (Season)",
    "win_pct_home_std":     "Home Win% (Season)",
    "win_pct_away_std":     "Road Win% (Season)",
    "pf_runs":              "Park Factor (Runs)",
    "pf_hr":                "Park Factor (HR)",
}


def _categorise(name: str) -> str:
    for rule in _GROUP_RULES:
        if len(rule) == 2:
            pattern, group = rule
            if pattern in name:
                return group
        else:
            prefix, suffix, group = rule
            if prefix in name and suffix in name:
                return group
    return "Other"


def _display_name(name: str) -> str:
    """Turn a raw feature column name into a human-readable label."""
    # Exact matches for special cases
    if name in _STAT_LABELS:
        return _STAT_LABELS[name]

    n = name
    side = "Home" if n.startswith("home_") else ("Away" if n.startswith("away_") else "")
    n = n.removeprefix("home_").removeprefix("away_")
    is_sp = n.startswith("sp_")
    if is_sp:
        n = n.removeprefix("sp_")

    # Strip trailing window suffix
    window = ""
    for suffix in ("_L30", "_l3", "_l5", "_l10", "_l20", "_std"):
        if n.endswith(suffix):
            window = suffix.lstrip("_").upper()
            n = n[: -len(suffix)]
            break

    stat = _STAT_LABELS.get(n, n.replace("_", " ").title())
    parts = []
    if side:
        parts.append(side)
    if is_sp:
        parts.append("SP")
    parts.append(stat)
    if window:
        parts.append(f"({window})")
    return " ".join(parts)


@router.get("/insights")
def get_insights():
    """Feature importances, group breakdown, and model metadata."""
    model = load_latest_model(MODEL_DIR, "home_win")

    names       = model.feature_name_
    importances = list(model.feature_importances_)
    total       = max(sum(importances), 1)

    features = sorted(
        [
            {
                "feature":      name,
                "label":        _display_name(name),
                "importance":   int(imp),
                "pct":          round(imp / total * 100, 2),
                "group":        _categorise(name),
                "color":        _GROUP_COLORS.get(_categorise(name), _GROUP_COLORS["Other"]),
            }
            for name, imp in zip(names, importances, strict=False)
        ],
        key=lambda x: -x["importance"],
    )

    # Group totals
    group_totals: dict[str, int] = defaultdict(int)
    for f in features:
        group_totals[f["group"]] += f["importance"]

    groups = sorted(
        [
            {
                "name":       k,
                "importance": v,
                "pct":        round(v / total * 100, 1),
                "color":      _GROUP_COLORS.get(k, _GROUP_COLORS["Other"]),
            }
            for k, v in group_totals.items()
        ],
        key=lambda x: -x["importance"],
    )

    # Model file metadata
    artifact = model_artifact_name("home_win", DEFAULT_MODEL_MODE)
    trained_date = "unknown"
    for p in sorted(MODEL_DIR.glob(f"{artifact}_2*.pkl"), reverse=True):
        stem = p.stem.replace(f"{artifact}_", "")
        if stem.isdigit() and len(stem) == 8:
            trained_date = f"{stem[:4]}-{stem[4:6]}-{stem[6:]}"
            break

    return {
        "meta": {
            "name":       "home_win",
            "algorithm":  "LightGBM Gradient Boosted Classifier",
            "n_features": len(names),
            "trained":    trained_date,
            "model_mode":  DEFAULT_MODEL_MODE,
            "calibration": getattr(model, "calibration_method", "none"),
            "target":     "P(home team wins)",
        },
        "features": features,
        "groups":   groups,
    }

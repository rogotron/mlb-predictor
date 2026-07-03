"""Ranked feature importance report for the home-win model.

Reads feature names and importances directly from the saved LightGBM model
so it always stays in sync with whatever features the model was trained on.

Example:
    python scripts/feature_importance.py
    python scripts/feature_importance.py --model total_runs
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils.paths import MODEL_DIR

# Human-readable category labels, matched by substring against feature names.
# Order determines the display order in the report.
CATEGORIES = [
    ("Team Rolling Form (L5/L10/L20)",      ["wins_l", "run_diff_l", "avg_runs_for_l", "avg_runs_against_l", "win_pct_l"]),
    ("SP Traditional — Rolling L3",          ["sp_era_l3", "sp_whip_l3", "sp_k_per_9_l3", "sp_bb_per_9_l3",
                                               "sp_k_minus_bb_pct_l3", "sp_hr_per_9_l3", "sp_ip_per_start_l3"]),
    ("SP Traditional — Season-to-Date",      ["sp_era_std", "sp_whip_std", "sp_k_per_9_std", "sp_bb_per_9_std",
                                               "sp_k_minus_bb_pct_std", "sp_hr_per_9_std", "sp_ip_total_std"]),
    ("SP Statcast Quality (xwOBA / Whiff)", ["sp_xwoba_against", "sp_whiff_rate", "sp_barrel_rate"]),
    ("Season-to-Date Team Rates",            ["runs_per_game_std", "ra_per_game_std", "win_pct_home_std", "win_pct_away_std"]),
    ("Team Batting Statcast",                ["xwoba_off_l", "barrel_rate_off_l"]),
    ("Lineup Matchup (xwOBA vs Hand / BvP)", ["lineup_xwoba_vs_sp", "bvp_xwoba"]),
    ("Schedule / Rest",                      ["days_rest"]),
    ("Park Factors",                         ["pf_runs", "pf_hr"]),
]


def _category(name: str) -> str:
    for label, patterns in CATEGORIES:
        if any(p in name for p in patterns):
            return label
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="home_win",
                        help="Model name: home_win or total_runs (default: home_win)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top features to show in detail (default: 20)")
    args = parser.parse_args()

    model_path = MODEL_DIR / f"{args.model}_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path} — run train_model.py first")

    model = pd.read_pickle(model_path)

    if not hasattr(model, "feature_name_"):
        raise AttributeError("Model does not expose feature_name_ — must be a LightGBM estimator")

    features = list(model.feature_name_)
    importances = list(model.feature_importances_)
    total = max(sum(importances), 1)

    df = pd.DataFrame({"feature": features, "importance": importances})
    df["pct"] = df["importance"] / total * 100
    df["category"] = df["feature"].map(_category)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\nFEATURE IMPORTANCE — {args.model.upper()} MODEL")
    print(f"{'=' * 65}")
    print(f"  {len(features)} features  |  best_iteration={getattr(model, 'best_iteration_', '?')}")
    print()

    # Category summary
    print("BY CATEGORY")
    print("-" * 65)
    cat_summary = (
        df.groupby("category")["pct"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    for _, row in cat_summary.iterrows():
        bar = "#" * int(row["pct"] / 2)
        print(f"  {row['category']:<40} {row['pct']:>5.1f}%  {bar}")
    print()

    # Top N individual features
    print(f"TOP {args.top} FEATURES")
    print("-" * 65)
    for rank, (_, row) in enumerate(df.head(args.top).iterrows(), 1):
        bar = "#" * int(row["pct"] * 3)
        print(f"  {rank:>2}. {row['feature']:<42} {row['pct']:>4.1f}%  {bar}")
    print()

    print("=" * 65)
    print("INTERPRETATION")
    print("  Importance = split gain: how much each feature reduces loss")
    print("  across all trees. Higher = more influence on each prediction.")
    print("  Features with <0.1% can usually be pruned with no loss.")
    low_signal = df[df["pct"] < 0.1]
    if not low_signal.empty:
        print(f"\n  {len(low_signal)} features below 0.1% threshold:")
        for _, row in low_signal.iterrows():
            print(f"    {row['feature']}")


if __name__ == "__main__":
    main()

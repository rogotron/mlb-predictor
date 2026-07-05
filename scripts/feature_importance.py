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

from src.models.feature_config import DEFAULT_MODEL_MODE, model_artifact_name
from src.models.feature_groups import group_name_and_source
from src.utils.paths import MODEL_DIR


def _category(name: str) -> str:
    # Shared with the dashboard's factor grouping so the CLI and UI agree.
    return group_name_and_source(name)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="home_win",
                        help="Model name: home_win or total_runs (default: home_win)")
    parser.add_argument("--model-mode", choices=["legacy_full", "pregame_safe"],
                        default=DEFAULT_MODEL_MODE,
                        help="Which trained artifact to inspect (default: the deployed mode)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top features to show in detail (default: 20)")
    args = parser.parse_args()

    artifact = model_artifact_name(args.model, args.model_mode)
    model_path = MODEL_DIR / f"{artifact}_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path} — run train_model.py first")
    print(f"model artifact: {model_path.name}  (mode={args.model_mode})")

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

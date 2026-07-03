"""Detailed analysis for Yankees game."""

from datetime import datetime

import requests

from src.data.update import SCHEDULE_TIMEZONE, fetch_today_slate, today_in_schedule_timezone
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.predict import predict_slate
from src.models.pregame import build_pregame_prediction_features
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Get team map
r = requests.get(f"{BASE_URL}/teams", params={"sportId": 1})
team_map = {t["id"]: t["name"] for t in r.json()["teams"]}

# Get slate and predictions
target = today_in_schedule_timezone()
slate = fetch_today_slate(target)
features = build_pregame_prediction_features(
    slate,
    processed_dir=PROCESSED_DIR,
    raw_dir=RAW_DIR,
    target_date=target,
    model_mode=DEFAULT_MODEL_MODE,
)
preds = predict_slate(features, MODEL_DIR, prediction_timestamp=datetime.now(SCHEDULE_TIMEZONE))

# Find Yankees game
yankees_game = None
for _, row in slate.iterrows():
    home_name = team_map.get(row["home_team_id"], "")
    away_name = team_map.get(row["away_team_id"], "")
    if "Yankees" in home_name or "Yankees" in away_name or "Astros" in home_name or "Astros" in away_name:
        yankees_game = row
        break

home_id = yankees_game["home_team_id"]
away_id = yankees_game["away_team_id"]
home_name = team_map[home_id]
away_name = team_map[away_id]

print("=" * 70)
print(f"GAME: {away_name} @ {home_name}")
print("=" * 70)

# Get features for this game
game_features = features[features["game_pk"] == yankees_game["game_pk"]].iloc[0]

# Get prediction
pred = preds[preds["game_pk"] == yankees_game["game_pk"]].iloc[0]

print("\nMODEL PREDICTION:")
print(f"  Winner: {home_name if pred['p_home_win'] > 0.5 else away_name}")
print(f"  {home_name} win probability: {pred['p_home_win']:.1%}")
print(f"  Expected total runs: {pred['expected_total_runs']:.1f}")

print(f"\n{'='*70}")
print(f"{home_name} (Home Team) - Recent Performance:")
print("=" * 70)
print(f"  Last 5 games:  {game_features.get('home_wins_l5', 0):.0f} wins")
print(f"  Last 10 games: {game_features.get('home_wins_l10', 0):.0f} wins")
print(f"  Last 20 games: {game_features.get('home_wins_l20', 0):.0f} wins")
print(f"  Run differential (last 10):  {game_features.get('home_run_diff_l10', 0):+.0f} runs")
print(f"  Run differential (last 20):  {game_features.get('home_run_diff_l20', 0):+.0f} runs")
print(f"  Avg runs scored (last 10):   {game_features.get('home_avg_runs_for_l10', 0):.1f} per game")
print(f"  Avg runs allowed (last 10):  {game_features.get('home_avg_runs_against_l10', 0):.1f} per game")
print(f"  Win percentage (last 20):    {game_features.get('home_win_pct_l20', 0.5)*100:.0f}%")

print(f"\n{'='*70}")
print(f"{away_name} (Away Team) - Recent Performance:")
print("=" * 70)
print(f"  Last 5 games:  {game_features.get('away_wins_l5', 0):.0f} wins")
print(f"  Last 10 games: {game_features.get('away_wins_l10', 0):.0f} wins")
print(f"  Last 20 games: {game_features.get('away_wins_l20', 0):.0f} wins")
print(f"  Run differential (last 10):  {game_features.get('away_run_diff_l10', 0):+.0f} runs")
print(f"  Run differential (last 20):  {game_features.get('away_run_diff_l20', 0):+.0f} runs")
print(f"  Avg runs scored (last 10):   {game_features.get('away_avg_runs_for_l10', 0):.1f} per game")
print(f"  Avg runs allowed (last 10):  {game_features.get('away_avg_runs_against_l10', 0):.1f} per game")
print(f"  Win percentage (last 20):    {game_features.get('away_win_pct_l20', 0.5)*100:.0f}%")

print(f"\n{'='*70}")
print("WHY THE MODEL PICKS NEW YORK YANKEES:")
print("=" * 70)

# Calculate advantages
home_rd_10 = game_features.get("home_run_diff_l10", 0) or 0
away_rd_10 = game_features.get("away_run_diff_l10", 0) or 0
home_wins_10 = game_features.get("home_wins_l10", 0) or 0
away_wins_10 = game_features.get("away_wins_l10", 0) or 0
home_rf_10 = game_features.get("home_avg_runs_for_l10", 0) or 0
away_rf_10 = game_features.get("away_avg_runs_for_l10", 0) or 0
home_ra_10 = game_features.get("home_avg_runs_against_l10", 0) or 0
away_ra_10 = game_features.get("away_avg_runs_against_l10", 0) or 0

if home_rd_10 > away_rd_10:
    print(f"  ✓ Run differential: Yankees +{home_rd_10} vs Astros {away_rd_10:+d}")
    print(f"    → Yankees have outscored opponents by {home_rd_10 - away_rd_10} more runs in last 10 games")

if home_wins_10 > away_wins_10:
    print(f"  ✓ Recent wins: Yankees {home_wins_10:.0f} vs Astros {away_wins_10:.0f} in last 10 games")

if home_rf_10 > away_rf_10:
    print(f"  ✓ Scoring: Yankees {home_rf_10:.1f} vs Astros {away_rf_10:.1f} runs per game")
    print(f"    → Yankees score {home_rf_10 - away_rf_10:.1f} more runs per game")

if home_ra_10 < away_ra_10:
    print(f"  ✓ Defense: Yankees allow {home_ra_10:.1f} vs Astros allow {away_ra_10:.1f} runs per game")
    print(f"    → Yankees pitching has been {away_ra_10 - home_ra_10:.1f} runs better")

print(f"\n  → Final prediction: Yankees win {pred['p_home_win']:.1%} of the time")
print(f"  → Expected score: ~{pred['expected_total_runs']:.0f} total runs")

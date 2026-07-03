"""Show Yankees predictions vs actual results for last 20 games."""

import pandas as pd
import requests
from src.utils.paths import PROCESSED_DIR, MODEL_DIR
from src.features.build import compute_team_rolling_features
from src.models.predict import FEATURE_COLS
import lightgbm as lgb

# Get team map
BASE_URL = "https://statsapi.mlb.com/api/v1"
r = requests.get(f"{BASE_URL}/teams", params={"sportId": 1})
team_map = {t["id"]: t["name"] for t in r.json()["teams"]}

# Yankees team ID
YANKEES_ID = 147

# Load processed games
games = pd.read_parquet(PROCESSED_DIR / "games" / "games_2024.parquet")
games["game_date"] = pd.to_datetime(games["game_date"])

# Find Yankees games (either home or away)
yankees_home = games[games["home_team_id"] == YANKEES_ID].copy()
yankees_away = games[games["away_team_id"] == YANKEES_ID].copy()

# Mark home/away
yankees_home["is_yankees_home"] = True
yankees_away["is_yankees_home"] = False

# Combine and sort by date
yankees_games = pd.concat([yankees_home, yankees_away]).sort_values("game_date")

# Get last 20 games
yankees_games = yankees_games.tail(20).copy()

print(f"YANKEEES LAST 20 GAMES IN 2024")
print("=" * 80)

# Load model
model = pd.read_pickle(MODEL_DIR / "home_win_latest.pkl")

# For each game, we need to compute features as of that date
# This requires computing rolling features up to each game date
results = []

for idx, game in yankees_games.iterrows():
    game_date = game["game_date"]
    game_pk = game["game_pk"]

    # Get all games before this date
    prior_games = games[games["game_date"] < game_date].copy()

    if len(prior_games) < 10:
        continue

    # Compute rolling features from prior games
    team_features = compute_team_rolling_features(prior_games)

    # Get latest stats for each team
    latest = team_features.sort_values("game_date").groupby("team_id").last().reset_index()

    # Get Yankees and opponent stats
    if game["is_yankees_home"]:
        home_id = YANKEES_ID
        away_id = game["away_team_id"]
        home_name = "Yankees"
        away_name = team_map.get(away_id, "Unknown")
    else:
        home_id = game["home_team_id"]
        away_id = YANKEES_ID
        home_name = team_map.get(home_id, "Unknown")
        away_name = "Yankees"

    home_stats = latest[latest["team_id"] == home_id]
    away_stats = latest[latest["team_id"] == away_id]

    if home_stats.empty or away_stats.empty:
        continue

    home_stats = home_stats.iloc[0]
    away_stats = away_stats.iloc[0]

    # Build feature row
    features = {}
    for col in FEATURE_COLS:
        if "home_" in col:
            stat_name = col.replace("home_", "")
            features[col] = home_stats.get(stat_name, 0)
        else:
            stat_name = col.replace("away_", "")
            features[col] = away_stats.get(stat_name, 0)

    # Make prediction
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    pred_proba = model.predict_proba(X)[0][1]

    # Actual result
    if game["is_yankees_home"]:
        actual_win = game["target_home_win"] == 1
        yankees_score = game["home_score"]
        opp_score = game["away_score"]
    else:
        actual_win = game["target_home_win"] == 0
        yankees_score = game["away_score"]
        opp_score = game["home_score"]

    # Predicted winner
    predicted_yankees = pred_proba > 0.5 if game["is_yankees_home"] else pred_proba < 0.5

    results.append({
        "date": game_date.strftime("%m/%d"),
        "home": home_name,
        "away": away_name,
        "score": f"{yankees_score}-{opp_score}",
        "pred_prob": pred_proba,
        "predicted": "Yankees" if predicted_yankees else home_name,
        "actual": "Yankees" if actual_win else home_name,
        "correct": predicted_yankees == actual_win
    })

# Create results dataframe
results_df = pd.DataFrame(results)

print(f"\n{'Date':<8} {'Matchup':<30} {'Score':<6} {'Pred%':<6} {'Predicted':<15} {'Actual':<15} {'Result'}")
print("-" * 95)

correct = 0
for _, row in results_df.iterrows():
    status = "OK" if row["correct"] else "X"
    print(f"{row['date']:<8} {row['home']:>12} vs {row['away']:<12} {row['score']:<6} {row['pred_prob']:.1%}   {row['predicted']:<15} {row['actual']:<15} {status}")
    if row["correct"]:
        correct += 1

print("-" * 95)
accuracy = correct / len(results_df) * 100
print(f"\nModel Accuracy: {correct}/{len(results_df)} = {accuracy:.1f}%")

# Calculate if model beat baseline
baseline = 0.5  # Random guessing
print(f"Baseline (random): {baseline:.1%}")
print(f"Model performance: {'BETTER' if accuracy > 50 else 'WORSE'} than baseline")
"""Show today's predictions with team names, probable starters, and key reasons."""

from __future__ import annotations

import argparse
from datetime import date, datetime

import requests

from src.data.update import SCHEDULE_TIMEZONE, fetch_today_slate, today_in_schedule_timezone
from src.models.audit import append_prediction_audit
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.predict import predict_slate
from src.models.pregame import build_pregame_prediction_features
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

BASE_URL = "https://statsapi.mlb.com/api/v1"

# MLBAM team ID for the New York Yankees
YANKEES_ID = 147


def get_team_map() -> dict[int, str]:
    r = requests.get(f"{BASE_URL}/teams", params={"sportId": 1})
    r.raise_for_status()
    return {t["id"]: t["name"] for t in r.json()["teams"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument("--team", type=int, default=None, help="Filter to one MLBAM team ID")
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()
    team_map = get_team_map()

    slate = fetch_today_slate(target)
    if slate.empty:
        print(f"No games scheduled for {target}.")
        return

    features = build_pregame_prediction_features(
        slate,
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        target_date=target,
        model_mode=DEFAULT_MODEL_MODE,
    )
    prediction_timestamp = datetime.now(SCHEDULE_TIMEZONE)
    preds = predict_slate(features, MODEL_DIR, prediction_timestamp=prediction_timestamp)
    append_prediction_audit(
        slate=slate,
        predictions=preds,
        features=features,
        model_dir=MODEL_DIR,
        now=prediction_timestamp,
    )

    # Attach team names and starter names
    slate["home_team_name"] = slate["home_team_id"].map(team_map)
    slate["away_team_name"] = slate["away_team_id"].map(team_map)

    preds = preds.merge(
        slate[["game_pk", "home_team_id", "away_team_id",
               "home_team_name", "away_team_name",
               "home_sp_name", "away_sp_name"]],
        on="game_pk",
    )
    preds = preds.sort_values("p_home_win", ascending=False).reset_index(drop=True)

    # Optionally filter to a single team
    filter_id = args.team
    if filter_id:
        mask = (preds["home_team_id"] == filter_id) | (preds["away_team_id"] == filter_id)
        preds = preds[mask]

    print("=" * 80)
    print(f"MLB PREDICTIONS FOR {target.strftime('%B %d, %Y')}")
    print("=" * 80)

    for _, row in preds.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        p_home = row["p_home_win"]
        total = row["expected_total_runs"]

        home_sp = row.get("home_sp_name") or "TBD"
        away_sp = row.get("away_sp_name") or "TBD"

        is_yankees_game = (
            row["home_team_id"] == YANKEES_ID or row["away_team_id"] == YANKEES_ID
        )
        sep = "** " if is_yankees_game else "   "

        print(f"\n{sep}{away} @ {home}")
        print(f"   Starters:  {away} - {away_sp}  |  {home} - {home_sp}")
        print(f"  {'TBD' if home_sp == 'TBD' or away_sp == 'TBD' else 'Announced'} probable starters")
        print("-" * 50)

        if p_home > 0.5:
            print(f"  Pick: {home}  ({p_home:.1%} win probability)")
        else:
            print(f"  Pick: {away}  ({1 - p_home:.1%} win probability)")
        print(f"  Expected total runs: {total:.1f}")

        # Fetch latest feature row for reasons
        feat_row = features[features["game_pk"] == row["game_pk"]]
        if feat_row.empty:
            continue
        feat_row = feat_row.iloc[0]

        reasons = []
        home_w5 = feat_row.get("home_wins_l5") or 0
        away_w5 = feat_row.get("away_wins_l5") or 0
        if home_w5 > away_w5 + 1:
            reasons.append(f"{home} won {home_w5:.0f}/5 recent vs {away}'s {away_w5:.0f}/5")
        elif away_w5 > home_w5 + 1:
            reasons.append(f"{away} won {away_w5:.0f}/5 recent vs {home}'s {home_w5:.0f}/5")

        home_rd = feat_row.get("home_run_diff_l10") or 0
        away_rd = feat_row.get("away_run_diff_l10") or 0
        if abs(home_rd - away_rd) >= 5:
            leader = home if home_rd > away_rd else away
            reasons.append(f"{leader} has +{abs(home_rd - away_rd):.0f} run diff advantage (last 10)")

        home_wp = feat_row.get("home_win_pct_l20") or 0.5
        away_wp = feat_row.get("away_win_pct_l20") or 0.5
        if abs(home_wp - away_wp) >= 0.1:
            leader = home if home_wp > away_wp else away
            reasons.append(
                f"{leader} has better L20 win pct ({max(home_wp, away_wp):.0%} vs {min(home_wp, away_wp):.0%})"
            )

        if reasons:
            print("  Key factors:")
            for r in reasons:
                print(f"    -{r}")

        # Yankees-specific callout
        if is_yankees_game:
            yankees_side = "home" if row["home_team_id"] == YANKEES_ID else "away"
            p_yankees = p_home if yankees_side == "home" else 1 - p_home
            opp = away if yankees_side == "home" else home
            print(f"\n  ** YANKEES WIN PROBABILITY vs {opp}: {p_yankees:.1%}")


if __name__ == "__main__":
    main()

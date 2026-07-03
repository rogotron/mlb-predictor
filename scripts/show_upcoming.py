"""Show model predictions for today and the next N days with probable starters.

Example:
    python scripts/show_upcoming.py            # today + 3 days
    python scripts/show_upcoming.py --days 7   # week ahead
    python scripts/show_upcoming.py --days 1   # today only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import requests

from src.data.update import SCHEDULE_TIMEZONE, fetch_slate_range, today_in_schedule_timezone
from src.models.audit import append_prediction_audit
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.predict import predict_slate
from src.models.pregame import build_pregame_prediction_features
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_team_map() -> dict[int, str]:
    r = requests.get(f"{BASE_URL}/teams", params={"sportId": 1})
    r.raise_for_status()
    return {t["id"]: t["abbreviation"] for t in r.json()["teams"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3, help="Number of days ahead (default: 3)")
    parser.add_argument("--team", type=int, default=None, help="Filter to one MLBAM team ID")
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    today = today_in_schedule_timezone()
    end = today + timedelta(days=args.days)
    team_map = get_team_map()

    slate = fetch_slate_range(today, end)
    if slate.empty:
        print(f"No games found {today} – {end}.")
        return

    if args.team:
        mask = (slate["home_team_id"] == args.team) | (slate["away_team_id"] == args.team)
        slate = slate[mask].reset_index(drop=True)

    # Build features + run model for the full slate at once
    features = build_pregame_prediction_features(
        slate,
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        target_date=today,
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
    preds = preds.merge(
        slate[["game_pk", "official_date", "home_team_id", "away_team_id",
               "home_sp_name", "away_sp_name", "venue_name"]],
        on="game_pk",
    )

    current_day = None
    for _, row in preds.sort_values(["official_date", "p_home_win"], ascending=[True, False]).iterrows():
        day = row["official_date"]
        if day != current_day:
            current_day = day
            label = "TODAY" if day == today else day.strftime("%a %b %d").upper()
            print(f"\n{'='*62}")
            print(f"  {label}")
            print(f"{'='*62}")

        away_abbr = team_map.get(int(row["away_team_id"]), "???")
        home_abbr = team_map.get(int(row["home_team_id"]), "???")
        away_sp   = row.get("away_sp_name") or "TBD"
        home_sp   = row.get("home_sp_name") or "TBD"
        p_home    = float(row["p_home_win"])
        p_away    = 1 - p_home
        total     = float(row["expected_total_runs"])

        if p_home >= p_away:
            pick = home_abbr
            pct  = p_home
        else:
            pick = away_abbr
            pct  = p_away

        announced = away_sp != "TBD" and home_sp != "TBD"
        starter_flag = "" if announced else " [probables TBD]"

        print(f"\n  {away_abbr} @ {home_abbr}{starter_flag}")
        print(f"    {away_abbr}: {away_sp}")
        print(f"    {home_abbr}: {home_sp}")
        print(f"    Pick: {pick}  ({pct:.1%})   |  Away {p_away:.1%} / Home {p_home:.1%}   |  O/U {total:.1f}")


if __name__ == "__main__":
    main()

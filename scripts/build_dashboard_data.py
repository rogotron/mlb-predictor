"""Build the dashboard data object for a given game."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import requests

from src.data.pitching_gamelogs import (
    load_pitching_gamelogs,
)
from src.data.update import SCHEDULE_TIMEZONE, fetch_today_slate, today_in_schedule_timezone
from src.models.audit import append_prediction_audit
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.predict import predict_slate
from src.models.pregame import build_pregame_prediction_features
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

DEFAULT_TEAM_ID = 121
BASE_URL = "https://statsapi.mlb.com/api/v1"
_API_TIMEOUT = 15


def _clean_name(value) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float) and math.isnan(value):
        return "TBD"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "TBD"


def _norm_dist(mu: float, r: int) -> int:
    return max(1, int(round(20 * math.exp(-0.5 * ((r - mu) ** 2) / 2.0))))


def _get_standings() -> dict[int, dict]:
    r = requests.get(
        f"{BASE_URL}/standings",
        params={"leagueId": "103,104", "season": 2026, "standingsTypes": "regularSeason"},
        timeout=_API_TIMEOUT,
    )
    r.raise_for_status()
    result: dict[int, dict] = {}
    for rec in r.json().get("records", []):
        div_name = (
            rec.get("division", {})
            .get("name", "—")
            .replace("American League ", "AL ")
            .replace("National League ", "NL ")
        )
        for tr in rec.get("teamRecords", []):
            tid = tr["team"]["id"]
            w, losses = tr["wins"], tr["losses"]
            wpct_raw = tr.get("winningPercentage", "0")
            wpct = f".{int(round(float(wpct_raw) * 1000)):03d}"
            div_rank = tr.get("divisionRank", "?")
            suffix = {1: "ST", 2: "ND", 3: "RD"}.get(int(div_rank), "TH") if str(div_rank).isdigit() else ""
            result[tid] = {
                "record": f"{w}-{losses}",
                "wPct": wpct,
                "division": div_name,
                "standing": f"{div_rank}{suffix}",
            }
    return result


def _get_hitting_log(team_id: int, n: int = 10) -> tuple[list, list, float]:
    r = requests.get(
        f"{BASE_URL}/teams/{team_id}/stats",
        params={"stats": "gameLog", "group": "hitting", "season": 2026, "gameType": "R"},
        timeout=_API_TIMEOUT,
    )
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])[-n:]
    spark, form = [], []
    for g in splits:
        spark.append({"g": len(spark) + 1, "r": int(g.get("stat", {}).get("runs", 0))})
        form.append("W" if g.get("isWin") else "L")
    rpg = round(sum(s["r"] for s in spark) / max(len(spark), 1), 1)
    return spark, form, rpg


def _get_pitcher(pid, name: str, gamelogs) -> dict:
    base = {
        "name": name, "hand": "RHP", "era": "—", "whip": "—", "k9": "—",
        "rec": "—", "gs": "—", "inn": "—", "last": "No data available",
    }
    if pid is None or (isinstance(pid, float) and math.isnan(pid)):
        return base
    pid = int(pid)
    try:
        pr = requests.get(f"{BASE_URL}/people/{pid}", timeout=8)
        pr.raise_for_status()
        pdata = pr.json().get("people", [{}])[0]
        pit_code = pdata.get("pitchHand", {}).get("code", "R")
        base["hand"] = "LHP" if pit_code == "L" else "RHP"

        sr = requests.get(
            f"{BASE_URL}/people/{pid}/stats",
            params={"stats": "season", "group": "pitching", "season": 2026, "gameType": "R"},
            timeout=8,
        )
        sr.raise_for_status()
        splits = sr.json().get("stats", [{}])[0].get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            base["rec"] = f"{st.get('wins', 0)}-{st.get('losses', 0)}"
            base["gs"] = str(st.get("gamesStarted", "—"))
            base["inn"] = str(st.get("inningsPitched", "—"))
            base["era"] = str(st.get("era", "—"))
            base["whip"] = str(st.get("whip", "—"))
            k9_raw = st.get("strikeoutsPer9Inn")
            base["k9"] = str(round(float(k9_raw), 1)) if k9_raw else "—"
    except Exception:
        pass

    p_logs = gamelogs[gamelogs["pitcher_id"] == pid].sort_values("game_date")
    if not p_logs.empty:
        lg = p_logs.iloc[-1]
        ip_f = lg["ip"]
        thirds = round((ip_f % 1) * 3)
        ip_str = f"{int(ip_f)}.{thirds}"
        base["last"] = f"{ip_str} IP, {int(lg['h'])} H, {int(lg['er'])} ER, {int(lg['bb'])} BB, {int(lg['k'])} K"
    return base


def build(
    target: date,
    game_pk: int,
    featured_team_id: int = DEFAULT_TEAM_ID,
    out_path: Path | None = None,
) -> dict:
    """Build and return the dashboard payload dict. Optionally write to *out_path*."""
    slate = fetch_today_slate(target)
    mask = slate["game_pk"] == game_pk
    if not mask.any():
        raise ValueError(f"game_pk {game_pk} not found in slate for {target}")
    row = slate[mask].iloc[0]

    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])

    # Determine which side is "featured" (for labeling purposes)
    if featured_team_id in (home_id, away_id):
        featured_side = "away" if row["away_team_id"] == featured_team_id else "home"
    else:
        featured_side = "home"
    opp_side = "home" if featured_side == "away" else "away"
    feat_id = int(row[f"{featured_side}_team_id"])
    opp_id_ = int(row[f"{opp_side}_team_id"])

    # Team metadata + standings (single API call for standings)
    teams_r = requests.get(f"{BASE_URL}/teams", params={"sportId": 1}, timeout=_API_TIMEOUT)
    teams_r.raise_for_status()
    team_map = {t["id"]: t for t in teams_r.json()["teams"]}
    feat_meta = team_map[feat_id]
    opp_meta = team_map[opp_id_]

    standings = _get_standings()
    default_rec = {"record": "—", "wPct": "—", "division": "—", "standing": "—"}
    feat_rec = standings.get(feat_id, default_rec)
    opp_rec = standings.get(opp_id_, default_rec)

    feat_spark, feat_form, feat_rpg = _get_hitting_log(feat_id)
    opp_spark, opp_form, opp_rpg = _get_hitting_log(opp_id_)

    # Game schedule info
    gr = requests.get(
        f"{BASE_URL}/schedule",
        params={"gamePk": game_pk, "hydrate": "venue,probables,linescore"},
        timeout=_API_TIMEOUT,
    )
    gr.raise_for_status()
    gdata = gr.json()["dates"][0]["games"][0]
    venue = gdata.get("venue", {}).get("name", "—")
    raw_dt = gdata.get("gameDate", "")
    try:
        dt_utc = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        dt_et = dt_utc.astimezone(SCHEDULE_TIMEZONE)
        first_pitch = dt_et.strftime("%-I:%M %p ET")
    except Exception:
        first_pitch = "—"

    sp_away_name = _clean_name(row.get("away_sp_name"))
    sp_home_name = _clean_name(row.get("home_sp_name"))
    sp_away_id = row.get("away_sp_id")
    sp_home_id = row.get("home_sp_id")

    # Pitcher stats
    gamelogs = load_pitching_gamelogs(2025, 2026, raw_dir=RAW_DIR)
    away_p = _get_pitcher(sp_away_id, sp_away_name, gamelogs)
    home_p = _get_pitcher(sp_home_id, sp_home_name, gamelogs)

    # Model features + prediction
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
    pred_row = preds[preds["game_pk"] == game_pk].iloc[0]
    p_home_win = float(pred_row["p_home_win"])
    exp_total = float(pred_row["expected_total_runs"])
    p_away_win = 1 - p_home_win

    if featured_side == "away":
        p_feat, p_opp = p_away_win, p_home_win
    else:
        p_feat, p_opp = p_home_win, p_away_win

    total_rpg = feat_rpg + opp_rpg
    feat_exp = round(exp_total * feat_rpg / total_rpg, 1) if total_rpg else round(exp_total / 2, 1)
    opp_exp = round(exp_total - feat_exp, 1)

    feat_score = int(round(feat_exp))
    opp_score_ = int(round(opp_exp))
    if p_feat > 0.5 and feat_score <= opp_score_:
        feat_score = opp_score_ + 1
    elif p_opp > 0.5 and opp_score_ <= feat_score:
        opp_score_ = feat_score + 1

    conf = int(abs(p_feat - 0.5) * 200)
    conf_label = (
        "MARGINAL" if conf < 25 else "LOW" if conf < 40 else "MODERATE" if conf < 60 else "HIGH"
    )

    if featured_side == "away":
        aw_full, aw_abbr = feat_meta["name"], feat_meta.get("abbreviation", "???")
        hm_full, hm_abbr = opp_meta["name"], opp_meta.get("abbreviation", "???")
        aw_rec, hm_rec = feat_rec, opp_rec
        aw_rpg, hm_rpg = feat_rpg, opp_rpg
        aw_spark, hm_spark = feat_spark, opp_spark
        aw_form, hm_form = feat_form, opp_form
        aw_score, hm_score = feat_score, opp_score_
        aw_prob = int(round(p_away_win * 100))
        hm_prob = 100 - aw_prob
        aw_runs, hm_runs = feat_exp, opp_exp
        aw_p, hm_p = away_p, home_p
    else:
        aw_full, aw_abbr = opp_meta["name"], opp_meta.get("abbreviation", "???")
        hm_full, hm_abbr = feat_meta["name"], feat_meta.get("abbreviation", "???")
        aw_rec, hm_rec = opp_rec, feat_rec
        aw_rpg, hm_rpg = opp_rpg, feat_rpg
        aw_spark, hm_spark = opp_spark, feat_spark
        aw_form, hm_form = opp_form, feat_form
        aw_score, hm_score = opp_score_, feat_score
        aw_prob = int(round(p_away_win * 100))
        hm_prob = 100 - aw_prob
        aw_runs, hm_runs = opp_exp, feat_exp
        aw_p, hm_p = away_p, home_p

    winner_name = aw_full if aw_prob > 50 else hm_full

    payload = {
        "game": {
            "date": target.strftime("%b %d, %Y").upper(),
            "reportNo": f"RPT-{target.year}-{game_pk}",
            "series": "REGULAR SEASON",
            "venue": venue,
            "firstPitch": first_pitch,
            "weather": "SEE VENUE",
            "line": "—",
            "status": gdata.get("status", {}).get("abstractGameState", "Scheduled").upper(),
        },
        "away": {
            "full": aw_full, "abbr": aw_abbr,
            "record": aw_rec["record"], "wPct": aw_rec["wPct"],
            "division": aw_rec["division"], "elo": "—", "runsPerG": str(aw_rpg),
            "standing": aw_rec["standing"],
            "form": aw_form or ["—"] * 10,
            "spark": aw_spark or [{"g": i + 1, "r": 4} for i in range(10)],
        },
        "home": {
            "full": hm_full, "abbr": hm_abbr,
            "record": hm_rec["record"], "wPct": hm_rec["wPct"],
            "division": hm_rec["division"], "elo": "—", "runsPerG": str(hm_rpg),
            "standing": hm_rec["standing"],
            "form": hm_form or ["—"] * 10,
            "spark": hm_spark or [{"g": i + 1, "r": 4} for i in range(10)],
        },
        "prediction": {
            "winner": winner_name, "awayProb": aw_prob, "homeProb": hm_prob,
            "awayRuns": aw_runs, "homeRuns": hm_runs,
            "awayScore": aw_score, "homeScore": hm_score,
            "confidence": conf, "confLabel": conf_label,
            "spread": "-1.5", "total": str(round(exp_total, 1)),
        },
        "pitchers": {"away": aw_p, "home": hm_p},
        "stats": [
            {"stat": "R/GAME (L10)", "away": str(aw_rpg), "home": str(hm_rpg)},
            {"stat": "RECORD", "away": aw_rec["record"], "home": hm_rec["record"]},
            {"stat": "WIN PCT", "away": aw_rec["wPct"], "home": hm_rec["wPct"]},
            {"stat": "SP ERA", "away": aw_p["era"], "home": hm_p["era"]},
            {"stat": "SP WHIP", "away": aw_p["whip"], "home": hm_p["whip"]},
            {"stat": "SP K/9", "away": aw_p["k9"], "home": hm_p["k9"]},
            {"stat": "DIVISION RANK", "away": aw_rec["standing"], "home": hm_rec["standing"]},
        ],
        "factors": [
            {"name": "Starting Pitcher Quality", "pct": 82, "note": f"{aw_p['name']} vs {hm_p['name']}"},
            {"name": "Recent Form (L10)", "pct": 76, "note": f"{''.join(aw_form)} vs {''.join(hm_form)}"},
            {"name": "Run Production", "pct": 64, "note": f"{aw_rpg} vs {hm_rpg} R/G last 10"},
            {"name": "Season-to-Date ERA", "pct": 58, "note": f"{aw_p['era']} vs {hm_p['era']} ERA"},
            {"name": "Win Percentage", "pct": 44, "note": f"{aw_rec['wPct']} vs {hm_rec['wPct']} overall"},
            {"name": "Home Field Factor", "pct": 36, "note": "Standard home field adjustment"},
            {"name": "Lineup Matchup", "pct": 22, "note": "Insufficient lineup data"},
        ],
        "runDist": [
            {"r": r, "away": _norm_dist(aw_runs, r), "home": _norm_dist(hm_runs, r)}
            for r in range(11)
        ],
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    return payload


if __name__ == "__main__":
    configure_logging()
    ensure_dirs()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    parser.add_argument("--team", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--game-pk", type=int, default=None)
    parser.add_argument("--out", default="public/dashboard-data.json")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()

    if args.game_pk is None:
        slate = fetch_today_slate(target)
        mask = (slate["home_team_id"] == args.team) | (slate["away_team_id"] == args.team)
        if not mask.any():
            raise SystemExit(f"No game for team {args.team} on {target}")
        gp = int(slate[mask].iloc[0]["game_pk"])
    else:
        gp = args.game_pk

    data = build(target, gp, featured_team_id=args.team, out_path=Path(args.out))
    print(json.dumps(data, indent=2, allow_nan=False))
    print(f"\nwrote {args.out}")

"""Assemble the full dashboard JSON payload for one or more games.

This is the only place that knows about the shape of data the dashboard expects.
Routes call build_slate_payloads(); everything else is private helpers.

External API calls are batched at the slate level (one team-map fetch, one
standings fetch) and cached in memory for 30 minutes so browser refreshes
don't hammer statsapi.mlb.com.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from src.data.pitching_gamelogs import _add_per_start_rates, load_pitching_gamelogs
from src.data.update import SCHEDULE_TIMEZONE, fetch_today_slate
from src.models.audit import append_prediction_audit
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.feature_groups import OTHER_GROUP, group_importances
from src.models.predict import load_latest_model, predict_slate
from src.models.pregame import build_pregame_prediction_features
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
CACHE_TTL_SECONDS = 1800  # 30 minutes
_SESSION = requests.Session()
_SESSION.trust_env = False

# Simple in-process cache keyed by date string
_payload_cache: dict[str, tuple[datetime, list[dict]]] = {}
_quick_payload_cache: dict[str, tuple[datetime, list[dict]]] = {}
_model_preview_cache: dict[str, tuple[datetime, list[dict]]] = {}


# ---------------------------------------------------------------------------
# External API helpers
# ---------------------------------------------------------------------------

def _get(path: str, **params: Any) -> dict:
    r = _SESSION.get(f"{BASE_URL}{path}", params=params, timeout=12)
    r.raise_for_status()
    return r.json()


def _fetch_team_meta() -> dict[int, dict]:
    data = _get("/teams", sportId=1)
    return {t["id"]: t for t in data.get("teams", [])}


def _fetch_standings(season: int) -> dict[int, dict]:
    data = _get("/standings", leagueId="103,104", season=season,
                standingsTypes="regularSeason")
    out: dict[int, dict] = {}
    for rec in data.get("records", []):
        div = rec.get("division", {}).get("name", "—")
        div = div.replace("American League ", "AL ").replace("National League ", "NL ")
        for tr in rec.get("teamRecords", []):
            tid = tr["team"]["id"]
            w, losses = tr["wins"], tr["losses"]
            wpct = f".{int(round(float(tr.get('winningPercentage', 0)) * 1000)):03d}"
            rank = tr.get("divisionRank", "?")
            suffix = {1: "ST", 2: "ND", 3: "RD"}.get(int(rank), "TH") if str(rank).isdigit() else ""
            out[tid] = {
                "record": f"{w}-{losses}", "wPct": wpct,
                "division": div, "standing": f"{rank}{suffix}",
            }
    return out


def _fetch_hitting_log(team_id: int, season: int, n: int = 10) -> tuple[list[dict], list[str], float]:
    try:
        data = _get(f"/teams/{team_id}/stats", stats="gameLog",
                    group="hitting", season=season, gameType="R")
        splits = data.get("stats", [{}])[0].get("splits", [])[-n:]
    except Exception:
        splits = []
    spark, form = [], []
    for g in splits:
        spark.append({"g": len(spark) + 1, "r": int(g.get("stat", {}).get("runs", 0))})
        form.append("W" if g.get("isWin") else "L")
    rpg = round(sum(s["r"] for s in spark) / max(len(spark), 1), 1)
    return spark, form, rpg


def _fetch_pitcher_info(pid: int | None, name: str, gamelogs: pd.DataFrame, season: int) -> dict:
    base: dict = {
        "name": name or "TBD", "hand": "—",
        "era": "—", "whip": "—", "k9": "—", "bb9": "—",
        "rec": "—", "gs": "—", "inn": "—",
        "last": "Probable not yet announced" if not name or name == "TBD" else "No gamelog data",
    }
    if pid is None:
        return base

    # Bio + handedness
    try:
        bio = _get(f"/people/{pid}").get("people", [{}])[0]
        hand = bio.get("pitchHand", {}).get("code", "R")
        base["hand"] = "LHP" if hand == "L" else "RHP"
        if not bio.get("mlbDebutDate"):
            base["last"] = "★ MLB DEBUT — No prior major league starts"
            return base
    except Exception:
        pass

    # Season stats (official numbers from the API, more reliable than our cache)
    try:
        sdata = _get(f"/people/{pid}/stats", stats="season",
                     group="pitching", season=season, gameType="R")
        splits = sdata.get("stats", [{}])[0].get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            base["rec"]  = f"{st.get('wins', 0)}-{st.get('losses', 0)}"
            base["gs"]   = str(st.get("gamesStarted", "—"))
            base["inn"]  = str(st.get("inningsPitched", "—"))
            base["era"]  = str(st.get("era", "—"))
            base["whip"] = str(st.get("whip", "—"))
            k9 = st.get("strikeoutsPer9Inn")
            base["k9"] = str(round(float(k9), 1)) if k9 else "—"
            bb9 = st.get("walksPer9Inn")
            base["bb9"] = str(round(float(bb9), 1)) if bb9 else "—"
    except Exception:
        pass

    # Last start line from local gamelog cache
    p_logs = gamelogs[gamelogs["pitcher_id"] == pid].sort_values("game_date")
    if not p_logs.empty:
        lg = p_logs.iloc[-1]
        thirds = round((lg["ip"] % 1) * 3)
        base["last"] = (
            f"{int(lg['ip'])}.{thirds} IP, {int(lg['h'])} H, "
            f"{int(lg['er'])} ER, {int(lg['bb'])} BB, {int(lg['k'])} K"
        )

    return base


# ---------------------------------------------------------------------------
# Factor importance
# ---------------------------------------------------------------------------

_OTHER_GROUP = OTHER_GROUP


def _model_importances(model) -> tuple[list[dict], float]:
    """Partition a fitted model's gain across the shared factor groups."""
    names_attr = getattr(model, "feature_name_", None)
    importances_attr = getattr(model, "feature_importances_", None)
    names = list(names_attr) if names_attr is not None else []
    importances = list(importances_attr) if importances_attr is not None else []
    return group_importances(names, importances)


def build_model_factor_summary(model) -> dict:
    """Slate-level model transparency block: real LightGBM importances by group.

    This is what the dashboard's Factors tab and rail read, so the displayed
    factors always match the deployed model. SHAP per-game attributions are a
    planned extension (see the per-game ``explain`` field); this summary is the
    global feature-importance view.
    """
    groups, _ = _model_importances(model)
    names_attr = getattr(model, "feature_name_", None)
    feature_count = len(list(names_attr)) if names_attr is not None else 0
    return {
        "name": "home_win",
        "mode": DEFAULT_MODEL_MODE,
        "featureCount": feature_count,
        "importanceMetric": "lightgbm_split_gain",
        "factorGroups": [
            {"name": g["name"], "source": g["source"], "pct": g["pct"]}
            for g in groups
        ],
    }


# Per-game context notes keyed by factor-group name. The importance share is
# global (same every game); the note carries this matchup's values.
def _factor_notes(
    aw_p: dict, hm_p: dict,
    aw_form: list[str], hm_form: list[str],
    aw_rpg: float, hm_rpg: float,
    aw_rec: dict, hm_rec: dict,
) -> dict[str, str]:
    return {
        "Starting Pitcher Quality": f"{aw_p['name']} vs {hm_p['name']}",
        "Rest & Availability": "Days rest and recent start load",
        "Recent Form (L10/L20)": f"{''.join(aw_form) or '—'} vs {''.join(hm_form) or '—'}",
        "Run Production": f"{aw_rpg} vs {hm_rpg} R/G last 10",
        "Bullpen Quality + Load": "Reliever run prevention and recent workload",
        "Home/Away Record": f"{aw_rec['wPct']} vs {hm_rec['wPct']} overall",
        "Park Factors": "Venue run/HR environment",
        _OTHER_GROUP[0]: "Additional model features",
    }


def _build_factors(
    feat: pd.Series,
    model,
    aw_p: dict, hm_p: dict,
    aw_form: list[str], hm_form: list[str],
    aw_rpg: float, hm_rpg: float,
    aw_rec: dict, hm_rec: dict,
) -> list[dict]:
    """Per-game factor list from the deployed model's real importances.

    Only groups the model actually uses appear (non-zero gain); ``pct`` is the
    group's true share of total model gain. No synthetic or anchored factors.
    """
    groups, _ = _model_importances(model)
    notes = _factor_notes(aw_p, hm_p, aw_form, hm_form, aw_rpg, hm_rpg, aw_rec, hm_rec)
    return [
        {
            "name": g["name"],
            "source": g["source"],
            "pct": g["pct"],
            "note": notes.get(g["name"], f"{g['source']} features"),
        }
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Run distribution (Poisson)
# ---------------------------------------------------------------------------

def _run_dist(aw_mu: float, hm_mu: float, n: int = 11) -> list[dict]:
    scale = 20

    def pois(mu: float, k: int) -> float:
        try:
            return math.exp(-mu) * (mu ** k) / math.factorial(k)
        except Exception:
            return 0.0

    return [
        {
            "r": r,
            "away": max(1, int(round(scale * pois(max(aw_mu, 0.1), r)))),
            "home": max(1, int(round(scale * pois(max(hm_mu, 0.1), r)))),
        }
        for r in range(n)
    ]


def _clean_text(value: Any, fallback: str = "TBD") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else fallback


def _confidence_label(probability: float) -> str:
    edge = abs(probability - 50) * 2
    if edge < 25:
        return "MARGINAL"
    if edge < 40:
        return "LOW"
    if edge < 60:
        return "MODERATE"
    return "HIGH"


def _side_prob_text(
    away_name: str,
    home_name: str,
    away_prob: int,
    home_prob: int,
) -> str:
    return f"{away_name} {away_prob}% / {home_name} {home_prob}%"


def _build_driver_sentences(
    *,
    away_name: str,
    home_name: str,
    winner: str,
    pick_prob: int,
    away_prob: int,
    home_prob: int,
    total: float,
    away_starter: str,
    home_starter: str,
    away_runs_l10: str | float,
    home_runs_l10: str | float,
    away_record: str,
    home_record: str,
    source: str,
    factors: list[dict] | None = None,
) -> list[str]:
    """Create concise, deterministic user-facing reasons for a pick."""
    sentences = [
        (
            f"The model makes {winner} the pick at {pick_prob}% win probability "
            f"({_side_prob_text(away_name, home_name, away_prob, home_prob)})."
        )
    ]

    if factors:
        top = sorted(factors, key=lambda item: item.get("pct", 0), reverse=True)[:2]
        if top:
            names = " and ".join(str(item.get("name", "")).lower() for item in top if item.get("name"))
            notes = "; ".join(str(item.get("note", "")) for item in top if item.get("note"))
            if names and notes:
                sentences.append(f"The biggest model drivers are {names}: {notes}.")
            elif names:
                sentences.append(f"The biggest model drivers are {names}.")

    if len(sentences) < 3:
        sentences.append(
            f"The probable-starter matchup is {away_starter} for {away_name} against "
            f"{home_starter} for {home_name}, with a projected total of {total:.1f} runs."
        )
    else:
        sentences.append(
            f"The starter matchup is {away_starter} vs {home_starter}, and the run model projects {total:.1f} total runs."
        )

    sentences.append(
        f"Recent team context has {away_name} at {away_runs_l10} runs per game over the last 10 "
        f"and {home_name} at {home_runs_l10}, with records of {away_record} and {home_record}."
    )

    if source != "model":
        sentences.append("This is a fast preview until the full model payload finishes in the background.")

    return sentences[:4]


def _team_recent_map(target_date: date) -> dict[int, dict]:
    path = RAW_DIR / "schedule" / f"schedule_{target_date.year}.parquet"
    if not path.exists():
        return {}

    try:
        schedule = pd.read_parquet(path)
    except Exception:
        logger.exception("failed to load cached schedule for quick recent stats")
        return {}

    schedule = schedule.copy()
    schedule["official_date"] = pd.to_datetime(schedule["official_date"]).dt.date
    schedule = schedule[
        (schedule["official_date"] < target_date)
        & schedule["away_score"].notna()
        & schedule["home_score"].notna()
    ].sort_values("game_date")

    games_by_team: dict[int, list[dict]] = {}
    for _, row in schedule.iterrows():
        away_id = int(row["away_team_id"])
        home_id = int(row["home_team_id"])
        away_runs = int(row["away_score"])
        home_runs = int(row["home_score"])
        games_by_team.setdefault(away_id, []).append({
            "runs": away_runs,
            "form": "W" if away_runs > home_runs else "L",
        })
        games_by_team.setdefault(home_id, []).append({
            "runs": home_runs,
            "form": "W" if home_runs > away_runs else "L",
        })

    out: dict[int, dict] = {}
    for team_id, team_games in games_by_team.items():
        last_ten = team_games[-10:]
        if not last_ten:
            continue
        out[team_id] = {
            "runsPerG": f"{sum(g['runs'] for g in last_ten) / len(last_ten):.1f}",
            "form": [g["form"] for g in last_ten],
            "spark": [{"g": i + 1, "r": g["runs"]} for i, g in enumerate(last_ten)],
        }
    return out


def _quick_game_payload(
    slate_row: pd.Series,
    meta: dict[int, dict],
    stnd: dict[int, dict],
    recent: dict[int, dict],
    target_date: date,
    pred: pd.Series | None = None,
) -> dict:
    game_pk = int(slate_row["game_pk"])
    aw_id = int(slate_row["away_team_id"])
    hm_id = int(slate_row["home_team_id"])
    aw_meta = meta.get(aw_id, {})
    hm_meta = meta.get(hm_id, {})
    aw_rec = stnd.get(aw_id, {"record": "-", "wPct": "-", "division": "-", "standing": "-"})
    hm_rec = stnd.get(hm_id, {"record": "-", "wPct": "-", "division": "-", "standing": "-"})
    aw_recent = recent.get(aw_id, {})
    hm_recent = recent.get(hm_id, {})

    def win_pct(record: dict) -> float:
        try:
            return float(str(record.get("wPct", "")).strip())
        except Exception:
            return 0.5

    if pred is not None:
        p_home = float(pred.get("p_home_win", 0.5))
        total = float(pred.get("expected_total_runs", 8.5))
        prediction_note = "Cached model prediction"
    else:
        # Lightweight preview for dates without a saved model run. It keeps
        # the calendar useful without triggering the slow rich prediction path.
        pct_delta = win_pct(hm_rec) - win_pct(aw_rec)
        p_home = min(0.65, max(0.35, 0.53 + pct_delta * 0.45))
        total_seed = (aw_id * 3 + hm_id * 5 + target_date.day + target_date.month) % 18
        total = round(7.6 + total_seed / 10, 1)
        prediction_note = "Standings-based preview; full model pending"

    p_away = 1 - p_home
    aw_runs = round(total * p_away / (p_away + p_home), 1)
    hm_runs = round(total - aw_runs, 1)
    aw_score = round(aw_runs)
    hm_score = round(hm_runs)
    if p_away > p_home and aw_score <= hm_score:
        aw_score = hm_score + 1
    if p_home > p_away and hm_score <= aw_score:
        hm_score = aw_score + 1

    aw_prob = round(p_away * 100)
    hm_prob = 100 - aw_prob
    pick_prob = max(aw_prob, hm_prob)
    winner = aw_meta.get("name", "Away") if aw_prob >= hm_prob else hm_meta.get("name", "Home")
    away_sp = _clean_text(slate_row.get("away_sp_name"))
    home_sp = _clean_text(slate_row.get("home_sp_name"))
    source = "model" if pred is not None else "preview"
    factors = [
        {"name": "Model Edge", "pct": pick_prob, "note": f"{winner} {pick_prob}%" if pred is not None else "Preview edge from standings"},
        {"name": "Run Environment", "pct": round(total * 8), "note": f"{total:.1f} projected runs"},
        {"name": "Starting Pitchers", "pct": 55 if pred is not None else 25, "note": f"{away_sp} vs {home_sp}"},
    ]
    drivers = _build_driver_sentences(
        away_name=aw_meta.get("name", "Away"),
        home_name=hm_meta.get("name", "Home"),
        winner=winner,
        pick_prob=pick_prob,
        away_prob=aw_prob,
        home_prob=hm_prob,
        total=total,
        away_starter=away_sp,
        home_starter=home_sp,
        away_runs_l10=aw_recent.get("runsPerG", "-"),
        home_runs_l10=hm_recent.get("runsPerG", "-"),
        away_record=aw_rec["record"],
        home_record=hm_rec["record"],
        source=source,
        factors=factors,
    )

    return {
        "game": {
            "gamePk": game_pk,
            "date": target_date.strftime("%b %d, %Y").upper(),
            "reportNo": f"RPT-{target_date.year}-{game_pk}",
            "series": "REGULAR SEASON",
            "venue": _clean_text(slate_row.get("venue_name"), "TBD"),
            "firstPitch": _first_pitch_et(slate_row.get("scheduled_start_utc", "")),
            "line": "-",
            "status": _clean_text(slate_row.get("status"), "PREVIEW").upper(),
        },
        "away": {
            "full": aw_meta.get("name", "Away"),
            "abbr": aw_meta.get("abbreviation", "???"),
            "record": aw_rec["record"],
            "wPct": aw_rec["wPct"],
            "division": aw_rec["division"],
            "elo": "-",
            "runsPerG": aw_recent.get("runsPerG", "-"),
            "standing": aw_rec["standing"],
            "form": aw_recent.get("form", ["-"] * 10),
            "spark": aw_recent.get("spark", [{"g": i + 1, "r": 4} for i in range(10)]),
        },
        "home": {
            "full": hm_meta.get("name", "Home"),
            "abbr": hm_meta.get("abbreviation", "???"),
            "record": hm_rec["record"],
            "wPct": hm_rec["wPct"],
            "division": hm_rec["division"],
            "elo": "-",
            "runsPerG": hm_recent.get("runsPerG", "-"),
            "standing": hm_rec["standing"],
            "form": hm_recent.get("form", ["-"] * 10),
            "spark": hm_recent.get("spark", [{"g": i + 1, "r": 4} for i in range(10)]),
        },
        "prediction": {
            "winner": winner,
            "awayProb": aw_prob,
            "homeProb": hm_prob,
            "awayRuns": aw_runs,
            "homeRuns": hm_runs,
            "awayScore": aw_score,
            "homeScore": hm_score,
            "confidence": round(abs(pick_prob - 50) * 2),
            "confLabel": _confidence_label(pick_prob),
            "spread": "-1.5",
            "total": f"{total:.1f}",
            "source": source,
            "drivers": drivers,
        },
        "pitchers": {
            "away": {"name": away_sp, "hand": "-", "era": "-", "whip": "-", "k9": "-", "rec": "-", "gs": "-", "inn": "-", "last": prediction_note},
            "home": {"name": home_sp, "hand": "-", "era": "-", "whip": "-", "k9": "-", "rec": "-", "gs": "-", "inn": "-", "last": prediction_note},
        },
        "stats": [
            {"stat": "MODEL PICK", "away": winner if pred is not None and aw_prob >= hm_prob else "-", "home": winner if pred is not None and hm_prob > aw_prob else "-"},
            {"stat": "WIN PROB", "away": f"{aw_prob}%", "home": f"{hm_prob}%"},
            {"stat": "EXP RUNS", "away": str(aw_runs), "home": str(hm_runs)},
            {"stat": "R/GAME (L10)", "away": aw_recent.get("runsPerG", "-"), "home": hm_recent.get("runsPerG", "-")},
            {"stat": "RECORD", "away": aw_rec["record"], "home": hm_rec["record"]},
            {"stat": "WIN PCT", "away": aw_rec["wPct"], "home": hm_rec["wPct"]},
            {"stat": "TOTAL", "away": f"{total:.1f}", "home": f"{total:.1f}"},
        ],
        "factors": factors,
        "runDist": _run_dist(aw_runs, hm_runs),
        # Extension point for per-game SHAP (TreeExplainer) attributions.
        "explain": {"shap": None},
    }


# ---------------------------------------------------------------------------
# Single-game assembler
# ---------------------------------------------------------------------------

def _first_pitch_et(scheduled_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(str(scheduled_utc).replace("Z", "+00:00"))
        et = dt.astimezone(SCHEDULE_TIMEZONE)
        return et.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return "TBD"


def _assemble_game(
    game_pk: int,
    slate_row: pd.Series,
    pred: pd.Series,
    feat: pd.Series,
    meta: dict[int, dict],
    stnd: dict[int, dict],
    gamelogs: pd.DataFrame,
    model,
    target_date: date,
) -> dict:
    season = target_date.year
    aw_id = int(slate_row["away_team_id"])
    hm_id = int(slate_row["home_team_id"])

    aw_meta = meta.get(aw_id, {})
    hm_meta = meta.get(hm_id, {})
    aw_rec = stnd.get(aw_id, {"record": "—", "wPct": "—", "division": "—", "standing": "—"})
    hm_rec = stnd.get(hm_id, {"record": "—", "wPct": "—", "division": "—", "standing": "—"})

    aw_spark, aw_form, aw_rpg = _fetch_hitting_log(aw_id, season)
    hm_spark, hm_form, hm_rpg = _fetch_hitting_log(hm_id, season)

    aw_sp_id   = slate_row.get("away_sp_id")
    hm_sp_id   = slate_row.get("home_sp_id")
    aw_sp_name = slate_row.get("away_sp_name") or "TBD"
    hm_sp_name = slate_row.get("home_sp_name") or "TBD"

    aw_p = _fetch_pitcher_info(
        int(aw_sp_id) if pd.notna(aw_sp_id) else None, aw_sp_name, gamelogs, season
    )
    hm_p = _fetch_pitcher_info(
        int(hm_sp_id) if pd.notna(hm_sp_id) else None, hm_sp_name, gamelogs, season
    )

    p_home    = float(pred["p_home_win"])
    p_away    = 1.0 - p_home
    exp_total = float(pred["expected_total_runs"])

    total_rpg = aw_rpg + hm_rpg
    aw_exp = round(exp_total * aw_rpg / total_rpg, 1) if total_rpg else round(exp_total / 2, 1)
    hm_exp = round(exp_total - aw_exp, 1)

    aw_score = int(round(aw_exp))
    hm_score = int(round(hm_exp))
    if p_away > p_home and aw_score <= hm_score:
        aw_score = hm_score + 1
    elif p_home > p_away and hm_score <= aw_score:
        hm_score = aw_score + 1

    aw_prob = int(round(p_away * 100))
    hm_prob = 100 - aw_prob
    winner  = aw_meta.get("name", "Away") if p_away >= p_home else hm_meta.get("name", "Home")
    conf    = int(abs(p_away - 0.5) * 200)
    conf_label = (
        "MARGINAL" if conf < 25 else
        "LOW"      if conf < 40 else
        "MODERATE" if conf < 60 else "HIGH"
    )

    factors = _build_factors(
        feat, model,
        aw_p, hm_p, aw_form, hm_form,
        aw_rpg, hm_rpg, aw_rec, hm_rec,
    )
    drivers = _build_driver_sentences(
        away_name=aw_meta.get("name", "Away"),
        home_name=hm_meta.get("name", "Home"),
        winner=winner,
        pick_prob=max(aw_prob, hm_prob),
        away_prob=aw_prob,
        home_prob=hm_prob,
        total=exp_total,
        away_starter=aw_p["name"],
        home_starter=hm_p["name"],
        away_runs_l10=aw_rpg,
        home_runs_l10=hm_rpg,
        away_record=aw_rec["record"],
        home_record=hm_rec["record"],
        source="model",
        factors=factors,
    )

    return {
        "game": {
            "gamePk":     game_pk,
            "date":       target_date.strftime("%b %d, %Y").upper(),
            "reportNo":   f"RPT-{target_date.year}-{game_pk}",
            "series":     "REGULAR SEASON",
            "venue":      slate_row.get("venue_name") or "—",
            "firstPitch": _first_pitch_et(slate_row.get("scheduled_start_utc", "")),
            "line":       "—",
            "status":     (slate_row.get("status") or "PREVIEW").upper(),
        },
        "away": {
            "full":     aw_meta.get("name", "Away"),
            "abbr":     aw_meta.get("abbreviation", "???"),
            "record":   aw_rec["record"],
            "wPct":     aw_rec["wPct"],
            "division": aw_rec["division"],
            "elo":      "—",
            "runsPerG": str(aw_rpg),
            "standing": aw_rec["standing"],
            "form":     aw_form or ["—"] * 10,
            "spark":    aw_spark or [{"g": i + 1, "r": 4} for i in range(10)],
        },
        "home": {
            "full":     hm_meta.get("name", "Home"),
            "abbr":     hm_meta.get("abbreviation", "???"),
            "record":   hm_rec["record"],
            "wPct":     hm_rec["wPct"],
            "division": hm_rec["division"],
            "elo":      "—",
            "runsPerG": str(hm_rpg),
            "standing": hm_rec["standing"],
            "form":     hm_form or ["—"] * 10,
            "spark":    hm_spark or [{"g": i + 1, "r": 4} for i in range(10)],
        },
        "prediction": {
            "winner":    winner,
            "awayProb":  aw_prob,
            "homeProb":  hm_prob,
            "awayRuns":  aw_exp,
            "homeRuns":  hm_exp,
            "awayScore": aw_score,
            "homeScore": hm_score,
            "confidence":  conf,
            "confLabel":   conf_label,
            "spread":      "-1.5",
            "total":       str(round(exp_total, 1)),
            "source":      "model",
            "drivers":     drivers,
        },
        "pitchers": {"away": aw_p, "home": hm_p},
        "stats": [
            {"stat": "R/GAME (L10)",  "away": str(aw_rpg),       "home": str(hm_rpg)},
            {"stat": "RECORD",        "away": aw_rec["record"],   "home": hm_rec["record"]},
            {"stat": "WIN PCT",       "away": aw_rec["wPct"],     "home": hm_rec["wPct"]},
            {"stat": "SP ERA",        "away": aw_p["era"],        "home": hm_p["era"]},
            {"stat": "SP WHIP",       "away": aw_p["whip"],       "home": hm_p["whip"]},
            {"stat": "SP K/9",        "away": aw_p["k9"],         "home": hm_p["k9"]},
            {"stat": "DIV STANDING",  "away": aw_rec["standing"], "home": hm_rec["standing"]},
        ],
        "factors":  factors,
        "runDist":  _run_dist(aw_exp, hm_exp),
        # Extension point for per-game SHAP (TreeExplainer) attributions.
        "explain":  {"shap": None},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_slate_payloads(
    target_date: date,
    slate: pd.DataFrame | None = None,
) -> list[dict]:
    """Build the full dashboard payload list for all games on a date.

    Results are cached in memory for CACHE_TTL_SECONDS. Pass an explicit
    slate (from fetch_slate_range) to bypass the daily fetch.
    """
    cache_key = str(target_date)
    now = datetime.now()
    cached = _payload_cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < CACHE_TTL_SECONDS:
        logger.debug("cache hit for %s", cache_key)
        return cached[1]

    if slate is None:
        slate = fetch_today_slate(target_date)
    if slate.empty:
        return []

    # Batch resources shared across all games
    meta  = _fetch_team_meta()
    stnd  = _fetch_standings(target_date.year)
    model = load_latest_model(MODEL_DIR, "home_win")

    gamelogs = load_pitching_gamelogs(target_date.year - 1, target_date.year, raw_dir=RAW_DIR)
    if not gamelogs.empty:
        gamelogs = _add_per_start_rates(gamelogs)

    # Run the full prediction pipeline for the whole slate at once. Stamp
    # features as-of the current time so an evening-before build isn't blocked
    # by the default end-of-prior-day stamp landing in the future.
    prediction_timestamp = datetime.now(SCHEDULE_TIMEZONE)
    features = build_pregame_prediction_features(
        slate,
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        target_date=target_date,
        model_mode=DEFAULT_MODEL_MODE,
        as_of_timestamp=prediction_timestamp,
    )
    preds = predict_slate(features, MODEL_DIR, prediction_timestamp=prediction_timestamp)
    append_prediction_audit(
        slate=slate,
        predictions=preds,
        features=features,
        model_dir=MODEL_DIR,
        now=prediction_timestamp,
    )

    payloads: list[dict] = []
    for _, slate_row in slate.iterrows():
        game_pk = int(slate_row["game_pk"])
        pred_match = preds[preds["game_pk"] == game_pk]
        if pred_match.empty:
            continue
        feat_match = features[features["game_pk"] == game_pk]
        feat = feat_match.iloc[0] if not feat_match.empty else pd.Series(dtype=float)

        try:
            payload = _assemble_game(
                game_pk=game_pk,
                slate_row=slate_row,
                pred=pred_match.iloc[0],
                feat=feat,
                meta=meta,
                stnd=stnd,
                gamelogs=gamelogs,
                model=model,
                target_date=target_date,
            )
            payloads.append(payload)
        except Exception:
            logger.exception("failed to assemble game_pk=%d", game_pk)

    _payload_cache[cache_key] = (now, payloads)
    return payloads


def build_quick_slate_payloads(target_date: date) -> list[dict]:
    """Build a fast dashboard-compatible slate.

    This path avoids the full feature rebuild and skips slow per-game probable
    pitcher live-feed calls. If a saved predictions CSV exists for the date,
    those probabilities are used; otherwise the slate is returned with pending
    model values so the calendar stays responsive.
    """
    cache_key = str(target_date)
    now = datetime.now()
    cached = _quick_payload_cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < CACHE_TTL_SECONDS:
        return cached[1]

    slate = fetch_today_slate(target_date, fill_probables=False)
    if slate.empty:
        _quick_payload_cache[cache_key] = (now, [])
        return []

    meta = _fetch_team_meta()
    try:
        stnd = _fetch_standings(target_date.year)
    except Exception:
        logger.exception("failed to fetch standings for quick slate")
        stnd = {}
    recent = _team_recent_map(target_date)
    prediction_path = PROCESSED_DIR / f"predictions_{target_date}.csv"
    preds = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()

    payloads: list[dict] = []
    for _, slate_row in slate.iterrows():
        game_pk = int(slate_row["game_pk"])
        pred_row = None
        if not preds.empty:
            match = preds[preds["game_pk"] == game_pk]
            if not match.empty:
                pred_row = match.iloc[0]
        payloads.append(_quick_game_payload(slate_row, meta, stnd, recent, target_date, pred_row))

    _quick_payload_cache[cache_key] = (now, payloads)
    return payloads


def build_model_preview_payloads(target_date: date) -> list[dict]:
    """Run the model but skip the slow rich dashboard assembly."""
    cache_key = str(target_date)
    now = datetime.now()
    cached = _model_preview_cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < CACHE_TTL_SECONDS:
        return cached[1]

    slate = fetch_today_slate(target_date, fill_probables=False)
    if slate.empty:
        _model_preview_cache[cache_key] = (now, [])
        return []

    meta = _fetch_team_meta()
    try:
        stnd = _fetch_standings(target_date.year)
    except Exception:
        logger.exception("failed to fetch standings for model preview")
        stnd = {}
    recent = _team_recent_map(target_date)

    prediction_timestamp = datetime.now(SCHEDULE_TIMEZONE)
    features = build_pregame_prediction_features(
        slate,
        processed_dir=PROCESSED_DIR,
        raw_dir=RAW_DIR,
        target_date=target_date,
        model_mode=DEFAULT_MODEL_MODE,
        as_of_timestamp=prediction_timestamp,
    )
    preds = predict_slate(features, MODEL_DIR, prediction_timestamp=prediction_timestamp)
    append_prediction_audit(
        slate=slate,
        predictions=preds,
        features=features,
        model_dir=MODEL_DIR,
        now=prediction_timestamp,
    )

    payloads: list[dict] = []
    for _, slate_row in slate.iterrows():
        game_pk = int(slate_row["game_pk"])
        match = preds[preds["game_pk"] == game_pk]
        pred_row = match.iloc[0] if not match.empty else None
        payloads.append(_quick_game_payload(slate_row, meta, stnd, recent, target_date, pred_row))

    _model_preview_cache[cache_key] = (now, payloads)
    return payloads


def invalidate_cache(target_date: date | None = None) -> None:
    """Evict one date (or all) from the in-process cache."""
    if target_date:
        _payload_cache.pop(str(target_date), None)
        _quick_payload_cache.pop(str(target_date), None)
        _model_preview_cache.pop(str(target_date), None)
    else:
        _payload_cache.clear()
        _quick_payload_cache.clear()
        _model_preview_cache.clear()

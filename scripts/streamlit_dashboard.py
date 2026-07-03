"""Interactive Streamlit dashboard for selecting and previewing MLB games.

Run:
    streamlit run scripts/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dashboard_data import build as build_dashboard_data  # noqa: E402
from src.data.update import fetch_today_slate, today_in_schedule_timezone  # noqa: E402

BASE_URL = "https://statsapi.mlb.com/api/v1"


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_names() -> dict[int, str]:
    response = requests.get(f"{BASE_URL}/teams", params={"sportId": 1}, timeout=15)
    response.raise_for_status()
    return {team["id"]: team["name"] for team in response.json()["teams"]}


def _team_label(row: pd.Series) -> str:
    away = row.get("away_team_name") or row.get("away_team_id")
    home = row.get("home_team_name") or row.get("home_team_id")
    return f"{away} @ {home}"


@st.cache_data(ttl=300, show_spinner=False)
def load_slate(target: date) -> pd.DataFrame:
    slate = fetch_today_slate(target)
    if slate.empty:
        return slate

    team_names = load_team_names()
    slate = slate.copy()
    slate["away_team_name"] = slate["away_team_id"].map(team_names)
    slate["home_team_name"] = slate["home_team_id"].map(team_names)
    slate["label"] = slate.apply(_team_label, axis=1)
    return slate


def build_payload(target: date, game_pk: int) -> dict:
    return build_dashboard_data(target, game_pk)


def render_dashboard(data: dict) -> None:
    game = data["game"]
    away = data["away"]
    home = data["home"]
    prediction = data["prediction"]
    pitchers = data["pitchers"]

    st.caption(f"{game['date']} | {game['venue']} | {game['firstPitch']}")
    st.title(f"{away['full']} @ {home['full']}")

    away_col, pick_col, home_col = st.columns([1, 1.2, 1])
    with away_col:
        st.subheader(away["abbr"])
        st.metric("Win probability", f"{prediction['awayProb']}%")
        st.metric("Projected runs", prediction["awayRuns"])
        st.write(f"Record: **{away['record']}**")
        st.write(f"Last 10 R/G: **{away['runsPerG']}**")
    with pick_col:
        st.subheader("Model Pick")
        st.metric("Winner", prediction["winner"])
        st.metric("Confidence", prediction["confLabel"], f"{prediction['confidence']} / 100")
        st.metric("Projected Score", f"{prediction['awayScore']} - {prediction['homeScore']}")
        st.metric("Total Runs", prediction["total"])
    with home_col:
        st.subheader(home["abbr"])
        st.metric("Win probability", f"{prediction['homeProb']}%")
        st.metric("Projected runs", prediction["homeRuns"])
        st.write(f"Record: **{home['record']}**")
        st.write(f"Last 10 R/G: **{home['runsPerG']}**")

    st.divider()

    sp_left, sp_right = st.columns(2)
    with sp_left:
        st.markdown(f"### {away['abbr']} Starter")
        st.write(f"**{pitchers['away']['name']}** ({pitchers['away']['hand']})")
        st.write(
            f"ERA {pitchers['away']['era']} | WHIP {pitchers['away']['whip']} | "
            f"K/9 {pitchers['away']['k9']}"
        )
        st.caption(pitchers["away"]["last"])
    with sp_right:
        st.markdown(f"### {home['abbr']} Starter")
        st.write(f"**{pitchers['home']['name']}** ({pitchers['home']['hand']})")
        st.write(
            f"ERA {pitchers['home']['era']} | WHIP {pitchers['home']['whip']} | "
            f"K/9 {pitchers['home']['k9']}"
        )
        st.caption(pitchers["home"]["last"])

    st.divider()

    stat_df = (
        pd.DataFrame(data["stats"])
        .rename(columns={"away": away["full"], "home": home["full"]})
        .set_index("stat")
    )
    factor_df = pd.DataFrame(data["factors"])
    run_df = pd.DataFrame(data["runDist"]).set_index("r")

    stats_col, factors_col = st.columns([1, 1])
    with stats_col:
        st.markdown("### Matchup Stats")
        st.dataframe(stat_df, use_container_width=True)
    with factors_col:
        st.markdown("### Model Factors")
        st.bar_chart(factor_df.set_index("name")["pct"])
        for factor in data["factors"]:
            st.caption(f"{factor['name']}: {factor['note']}")

    st.markdown("### Run Distribution")
    st.line_chart(run_df)


def main() -> None:
    st.set_page_config(page_title="MLB Predictor Dashboard", layout="wide")
    st.sidebar.title("MLB Predictor")

    target = st.sidebar.date_input("Game date", value=today_in_schedule_timezone())
    slate = load_slate(target)
    if slate.empty:
        st.warning(f"No games found for {target}.")
        return

    labels = slate["label"].tolist()
    default_index = next(
        (
            idx
            for idx, label in enumerate(labels)
            if "Nationals" in label or "Mets" in label
        ),
        0,
    )
    selected_label = st.sidebar.selectbox("Game", labels, index=default_index)
    selected = slate.loc[slate["label"] == selected_label].iloc[0]

    if st.sidebar.button("Refresh slate"):
        load_slate.clear()
        st.rerun()

    with st.spinner("Running model and building dashboard..."):
        try:
            data = build_payload(target, int(selected["game_pk"]))
        except Exception as exc:
            st.error(str(exc))
            return

    render_dashboard(data)


if __name__ == "__main__":
    main()

"""Team-level features.

All rolling windows must use only games strictly BEFORE the target game's date
to avoid target leakage. Use df.groupby('team_id').rolling(...).shift(1) or
equivalent.
"""

from __future__ import annotations

import pandas as pd


def last_n_record(team_games: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Wins, losses, and run differential over the previous n games per team.

    Args:
        team_games: long-format game log with columns
            [game_pk, game_date, team_id, runs_for, runs_against, won]
        n: window size

    Returns:
        DataFrame keyed by (game_pk, team_id) with columns
            wins_l{n}, losses_l{n}, run_diff_l{n}, win_pct_l{n}
    """
    df = team_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["won"] = df["won"].astype(int)
    df["run_diff"] = df["runs_for"] - df["runs_against"]
    df = df.sort_values(["team_id", "game_date", "game_pk"]).reset_index(drop=True)

    grouped = df.groupby("team_id", group_keys=False)
    wins = grouped["won"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    games = grouped["won"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).count())
    run_diff = grouped["run_diff"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())

    out = df[["game_pk", "team_id"]].copy()
    out[f"wins_l{n}"] = wins.fillna(0)
    out[f"losses_l{n}"] = (games - wins).fillna(0)
    out[f"run_diff_l{n}"] = run_diff.fillna(0)
    out[f"win_pct_l{n}"] = out[f"wins_l{n}"] / n
    out[f"n_games_l{n}"] = games.fillna(0).astype(int)
    return out


def season_to_date(team_games: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date runs/game (offense and defense) per (game_pk, team_id).

    Uses shift(1).expanding() so the current game is always excluded.
    Returns NaN for opening day (no prior games in season).
    """
    df = team_games[["game_pk", "team_id", "game_date", "runs_for", "runs_against"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["_year"] = df["game_date"].dt.year
    df = df.sort_values(["team_id", "_year", "game_date"]).reset_index(drop=True)

    grouped = df.groupby(["team_id", "_year"], group_keys=False)
    df["_cum_rf"] = grouped["runs_for"].transform(lambda x: x.shift(1).expanding().sum())
    df["_cum_ra"] = grouped["runs_against"].transform(lambda x: x.shift(1).expanding().sum())
    df["_cum_g"] = grouped["runs_for"].transform(lambda x: x.shift(1).expanding().count())

    safe_g = df["_cum_g"].clip(lower=1)
    df["runs_per_game_std"] = df["_cum_rf"] / safe_g
    df["ra_per_game_std"] = df["_cum_ra"] / safe_g
    df.loc[df["_cum_g"] == 0, ["runs_per_game_std", "ra_per_game_std"]] = float("nan")

    return df[["game_pk", "team_id", "runs_per_game_std", "ra_per_game_std"]]


def home_away_split(team_games: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date home and away win% per (game_pk, team_id).

    win_pct_home_std: team's win% in home games played before this game this season.
    win_pct_away_std: team's win% in road games played before this game this season.
    Both are NaN until the team has played at least one game of that type.

    Requires team_games to have an `is_home` column (1=home, 0=away) and a
    `won` column, as produced by compute_team_rolling_features() in build.py.
    """
    df = team_games[["game_pk", "team_id", "game_date", "is_home", "won"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["_year"] = df["game_date"].dt.year
    df = df.sort_values(["team_id", "_year", "game_date", "game_pk"]).reset_index(drop=True)

    out_rows = []
    for (team_id, year), grp in df.groupby(["team_id", "_year"]):
        home_wins = home_g = away_wins = away_g = 0
        for _, row in grp.iterrows():
            wpct_home = home_wins / home_g if home_g > 0 else float("nan")
            wpct_away = away_wins / away_g if away_g > 0 else float("nan")
            out_rows.append({
                "game_pk": row["game_pk"],
                "team_id": row["team_id"],
                "win_pct_home_std": wpct_home,
                "win_pct_away_std": wpct_away,
            })
            if row["is_home"] == 1:
                home_g += 1
                home_wins += int(row["won"])
            else:
                away_g += 1
                away_wins += int(row["won"])

    return pd.DataFrame(out_rows)


def days_rest(team_games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game per (game_pk, team_id).

    NaN for the first game of the season (no prior game on record).
    0 means doubleheader (same calendar date as previous game).
    """
    df = team_games[["game_pk", "team_id", "game_date"]].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "game_date", "game_pk"]).reset_index(drop=True)
    df["_prev_date"] = df.groupby("team_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["_prev_date"]).dt.days
    return df[["game_pk", "team_id", "days_rest"]]

"""Compare legacy, pregame-safe, and simple MLB prediction baselines."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.features.build import (
    _join_park_factors,
    compute_team_rolling_features,
    join_bullpen_features,
    join_gamelog_pitcher_features,
    join_pitch_quality_features,
    join_pitcher_features,
    join_posted_lineup_features,
    join_team_statcast_features,
    load_processed_games,
)
from src.features.team import days_rest, home_away_split, season_to_date
from src.models.calibration import fit_probability_calibrator
from src.models.feature_config import MODEL_MODE_LEGACY_FULL, MODEL_MODE_PREGAME_SAFE
from src.models.train import get_features
from src.utils.logging import configure_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "diagnostics" / "backtests" / "leakage_safe_comparison"
MATRIX_PATH = OUT_DIR / "backtest_training_matrix_2018_2025.parquet"
FULL_MATRIX_PATH = OUT_DIR / "backtest_training_matrix_2018_2025_full.parquet"
SAFE_FEATURES_USED_PATH = OUT_DIR / "leakage_safe_feature_cols_used_in_backtest.json"
EPS = 1e-6
FULL_STAGE_DIR = OUT_DIR / "full_feature_build_stages"


@dataclass(frozen=True)
class BacktestWindow:
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"
    eval_start: str = "2024-01-01"
    eval_end: str = "2025-12-31"


def _load_matrix(rebuild: bool) -> pd.DataFrame:
    if MATRIX_PATH.exists() and not rebuild:
        logger.info("loading cached backtest matrix: %s", MATRIX_PATH)
        return pd.read_parquet(MATRIX_PATH)

    logger.info("building backtest matrix from cached processed/raw data")
    df = _build_comparison_matrix()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MATRIX_PATH, index=False)
    return df


def _load_full_matrix(rebuild: bool) -> pd.DataFrame:
    if FULL_MATRIX_PATH.exists() and not rebuild:
        logger.info("loading cached full backtest matrix: %s", FULL_MATRIX_PATH)
        return pd.read_parquet(FULL_MATRIX_PATH)

    logger.info("building full all-feature matrix from cached processed/raw data")
    df = _build_full_feature_matrix(rebuild=rebuild)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FULL_MATRIX_PATH, index=False)
    return df


def _stage_path(stage_name: str) -> Path:
    return FULL_STAGE_DIR / f"{stage_name}.parquet"


def _load_or_run_stage(
    stage_name: str,
    func,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    path = _stage_path(stage_name)
    if path.exists() and not rebuild:
        logger.info("loading full-build stage %s: %s", stage_name, path)
        return pd.read_parquet(path)
    logger.info("running full-build stage %s", stage_name)
    df = func()
    FULL_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("saved full-build stage %s: %s shape=%s", stage_name, path, df.shape)
    return df


def _build_full_feature_matrix(rebuild: bool = False) -> pd.DataFrame:
    """Build and cache the all-feature historical matrix stage by stage."""

    def base_stage() -> pd.DataFrame:
        existing_fast = MATRIX_PATH
        if existing_fast.exists() and not rebuild:
            logger.info("seeding full build from existing fast matrix: %s", existing_fast)
            return pd.read_parquet(existing_fast)
        return _build_comparison_matrix()

    df = _load_or_run_stage("00_base_team_gamelog_pitch_quality", base_stage, rebuild=rebuild)
    df = _load_or_run_stage("01_starter_statcast", lambda: join_pitcher_features(df, RAW_DIR), rebuild=rebuild)
    df = _load_or_run_stage("02_team_statcast", lambda: join_team_statcast_features(df, RAW_DIR), rebuild=rebuild)
    df = _load_or_run_stage("03_posted_lineups", lambda: join_posted_lineup_features(df, RAW_DIR), rebuild=rebuild)
    df = _load_or_run_stage("04_bullpen", lambda: join_bullpen_features(df, RAW_DIR), rebuild=rebuild)
    df = _load_or_run_stage("05_lineup_matchup", lambda: _join_lineup_matchup_features_fast(df), rebuild=rebuild)
    return df.sort_values("game_date").reset_index(drop=True)


def _join_lineup_matchup_features_fast(games: pd.DataFrame) -> pd.DataFrame:
    """Vectorized all-history lineup matchup join for the full diagnostic build."""
    from src.data.statcast import (
        aggregate_game_lineups,
        aggregate_pitcher_starts,
        compute_batter_season_splits,
        load_statcast,
    )

    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    pitches = load_statcast(games["game_date"].min().date(), games["game_date"].max().date(), raw_dir=RAW_DIR)
    out_cols = [
        "home_lineup_xwoba_vs_sp",
        "away_lineup_xwoba_vs_sp",
        "home_bvp_xwoba",
        "away_bvp_xwoba",
    ]
    if pitches.empty or "inning_topbot" not in pitches.columns:
        for col in out_cols:
            games[col] = float("nan")
        return games

    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    lineups = aggregate_game_lineups(pitches)
    starts = aggregate_pitcher_starts(pitches)
    sp_throws = (
        starts[["pitcher", "game_pk", "p_throws"]]
        .rename(columns={"pitcher": "pitcher_id"})
        .drop_duplicates(subset=["game_pk", "pitcher_id"])
    )

    wide = games[["game_pk", "game_date", "home_sp_id", "away_sp_id"]].copy()
    wide["game_year"] = wide["game_date"].dt.year
    wide = wide.merge(lineups, on="game_pk", how="left")
    home_throws = (
        games[["game_pk", "home_sp_id"]].dropna(subset=["home_sp_id"])
        .assign(pitcher_id=lambda d: d["home_sp_id"].astype(int))
        .merge(sp_throws, on=["game_pk", "pitcher_id"], how="left")
        .rename(columns={"p_throws": "home_sp_throws"})[["game_pk", "home_sp_throws"]]
    )
    away_throws = (
        games[["game_pk", "away_sp_id"]].dropna(subset=["away_sp_id"])
        .assign(pitcher_id=lambda d: d["away_sp_id"].astype(int))
        .merge(sp_throws, on=["game_pk", "pitcher_id"], how="left")
        .rename(columns={"p_throws": "away_sp_throws"})[["game_pk", "away_sp_throws"]]
    )
    wide = wide.merge(home_throws, on="game_pk", how="left")
    wide = wide.merge(away_throws, on="game_pk", how="left")

    splits = [
        compute_batter_season_splits(pitches[pitches["game_date"].dt.year == year], int(year))
        for year in sorted(pitches["game_date"].dt.year.unique())
    ]
    splits_all = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame()

    def lineup_xwoba(lineup_col: str, hand_col: str) -> pd.Series:
        expanded = (
            wide[["game_pk", "game_year", lineup_col, hand_col]]
            .explode(lineup_col)
            .dropna(subset=[lineup_col, hand_col])
            .rename(columns={lineup_col: "batter", hand_col: "hand"})
        )
        if expanded.empty or splits_all.empty:
            return pd.Series(dtype=float, name="val")
        expanded["batter"] = expanded["batter"].astype(int)
        expanded["year"] = expanded["game_year"].astype(int) - 1
        joined = expanded.merge(splits_all, on=["batter", "year"], how="left")
        joined["xwoba"] = np.where(joined["hand"] == "L", joined["xwoba_vs_L"], joined["xwoba_vs_R"])
        return joined.groupby("game_pk")["xwoba"].mean().rename("val")

    def bvp(lineup_col: str, sp_col: str) -> pd.Series:
        bvp_pa = (
            pitches.loc[
                pitches["woba_denom"].fillna(0).astype(float).eq(1),
                ["game_date", "batter", "pitcher", "estimated_woba_using_speedangle"],
            ]
            .dropna(subset=["game_date", "batter", "pitcher"])
            .assign(
                batter=lambda d: d["batter"].astype(int),
                pitcher=lambda d: d["pitcher"].astype(int),
            )
            .rename(columns={"estimated_woba_using_speedangle": "_xwoba"})
            .sort_values(["batter", "pitcher", "game_date"])
        )
        if bvp_pa.empty:
            return pd.Series(dtype=float, name="val")
        grouped = bvp_pa.groupby(["batter", "pitcher"], group_keys=False)
        bvp_pa["pa_count"] = grouped.cumcount() + 1
        bvp_pa["_xwoba_sum"] = grouped["_xwoba"].cumsum()
        bvp_pa["xwoba_bvp"] = bvp_pa["_xwoba_sum"] / bvp_pa["pa_count"].clip(lower=1)
        lookup = bvp_pa[["game_date", "batter", "pitcher", "pa_count", "xwoba_bvp"]]

        left = (
            wide[["game_pk", "game_date", lineup_col, sp_col]]
            .dropna(subset=[sp_col])
            .explode(lineup_col)
            .dropna(subset=[lineup_col])
            .rename(columns={lineup_col: "batter", sp_col: "pitcher"})
        )
        if left.empty:
            return pd.Series(dtype=float, name="val")
        left["batter"] = left["batter"].astype(int)
        left["pitcher"] = left["pitcher"].astype(int)
        merged = pd.merge_asof(
            left.sort_values("game_date"),
            lookup.sort_values("game_date"),
            on="game_date",
            by=["batter", "pitcher"],
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged[merged["pa_count"].fillna(0) >= 20].copy()
        if merged.empty:
            return pd.Series(dtype=float, name="val")
        merged["_weight"] = merged["pa_count"].clip(upper=60) / 60
        merged["_weighted_xwoba"] = merged["_weight"] * merged["xwoba_bvp"]
        sums = merged.groupby("game_pk")[["_weighted_xwoba", "_weight"]].sum()
        return (sums["_weighted_xwoba"] / sums["_weight"]).rename("val")

    features = pd.DataFrame({"game_pk": games["game_pk"]})
    for col_name, values in {
        "home_lineup_xwoba_vs_sp": lineup_xwoba("home_lineup_ids", "away_sp_throws"),
        "away_lineup_xwoba_vs_sp": lineup_xwoba("away_lineup_ids", "home_sp_throws"),
        "home_bvp_xwoba": bvp("home_lineup_ids", "away_sp_id"),
        "away_bvp_xwoba": bvp("away_lineup_ids", "home_sp_id"),
    }.items():
        features = features.merge(values.rename(col_name).reset_index(), on="game_pk", how="left")

    return games.merge(features, on="game_pk", how="left")


def _build_comparison_matrix() -> pd.DataFrame:
    """Build the comparison feature matrix without fetching external data.

    The matrix materializes the fast pregame-safe families available from
    processed games and cached starter gamelogs, plus the legacy pitch-quality
    family. Historical lineup-derived columns and heavy Statcast aggregates are
    intentionally not rebuilt here; the report records the safe feature subset
    used in this local backtest.
    """
    games = load_processed_games(PROCESSED_DIR, 2018, 2025)
    team_features = compute_team_rolling_features(games)

    std_feats = season_to_date(team_features)
    team_features = team_features.merge(
        std_feats[["game_pk", "team_id", "runs_per_game_std", "ra_per_game_std"]],
        on=["game_pk", "team_id"],
        how="left",
    )
    ha_feats = home_away_split(team_features)
    team_features = team_features.merge(
        ha_feats[["game_pk", "team_id", "win_pct_home_std", "win_pct_away_std"]],
        on=["game_pk", "team_id"],
        how="left",
    )
    rest_feats = days_rest(team_features)
    team_features = team_features.merge(
        rest_feats[["game_pk", "team_id", "days_rest"]],
        on=["game_pk", "team_id"],
        how="left",
    )

    home_features = team_features[team_features["is_home"] == 1].copy()
    home_cols = [
        col for col in home_features.columns
        if col not in [
            "game_pk",
            "game_date",
            "team_id",
            "runs_for",
            "runs_against",
            "venue_id",
            "is_home",
            "won",
            "run_diff",
        ]
    ]
    home_features = home_features.rename(columns={col: f"home_{col}" for col in home_cols})
    home_features = home_features[["game_pk"] + [f"home_{col}" for col in home_cols]]

    away_features = team_features[team_features["is_home"] == 0].copy()
    away_features = away_features.rename(columns={col: f"away_{col}" for col in home_cols})
    away_features = away_features[["game_pk"] + [f"away_{col}" for col in home_cols]]

    result = games.merge(home_features, on="game_pk", how="left")
    result = result.merge(away_features, on="game_pk", how="left")
    logger.info("joined team features: %s", result.shape)

    result = _join_park_factors(result, RAW_DIR, PROCESSED_DIR)
    logger.info("joined park factors: %s", result.shape)
    result = join_gamelog_pitcher_features(result, RAW_DIR)
    logger.info("joined gamelog pitcher features: %s", result.shape)
    result = join_pitch_quality_features(result, RAW_DIR)
    logger.info("joined pitch-quality features: %s", result.shape)

    return result.sort_values("game_date").reset_index(drop=True)


def _write_safe_features_used(df: pd.DataFrame) -> tuple[int, int]:
    safe_path = PROCESSED_DIR / "leakage_safe_feature_cols.json"
    safe_features = json.loads(safe_path.read_text(encoding="utf-8"))
    available = set(df.columns)
    used = []
    for feature in safe_features:
        if feature in available or (
            feature.endswith("_missing") and feature.removesuffix("_missing") in available
        ):
            used.append(feature)
    SAFE_FEATURES_USED_PATH.write_text(json.dumps(used, indent=2) + "\n", encoding="utf-8")
    return len(used), len(safe_features)


def _train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, model_mode: str) -> lgb.LGBMClassifier:
    safe_path = SAFE_FEATURES_USED_PATH if model_mode == MODEL_MODE_PREGAME_SAFE else PROCESSED_DIR / "leakage_safe_feature_cols.json"
    x_train = get_features(train_df, model_mode=model_mode, safe_features_path=safe_path)
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = train_df["target_home_win"].astype(int)
    x_val = get_features(val_df, model_mode=model_mode, safe_features_path=safe_path)
    x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val = val_df["target_home_win"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    if model_mode == MODEL_MODE_PREGAME_SAFE:
        model = fit_probability_calibrator(model, x_val, y_val)
    return model


def _model_predictions(model: lgb.LGBMClassifier, eval_df: pd.DataFrame, model_mode: str) -> pd.DataFrame:
    safe_path = SAFE_FEATURES_USED_PATH if model_mode == MODEL_MODE_PREGAME_SAFE else PROCESSED_DIR / "leakage_safe_feature_cols.json"
    x_eval = get_features(eval_df, model_mode=model_mode, safe_features_path=safe_path)
    x_eval = x_eval.apply(pd.to_numeric, errors="coerce").fillna(0)
    out = _prediction_frame(eval_df, model_mode)
    out["p_home_win"] = model.predict_proba(x_eval)[:, 1]
    return out


def _prediction_frame(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    cols = [
        "game_pk",
        "game_date",
        "official_date",
        "home_team_id",
        "away_team_id",
        "target_home_win",
        "home_score",
        "away_score",
    ]
    available = [col for col in cols if col in df.columns]
    out = df[available].copy()
    out["model_name"] = model_name
    out["excluded"] = False
    out["exclusion_reason"] = ""
    return out


def _prior_record_features(df: pd.DataFrame) -> pd.DataFrame:
    games = df[
        [
            "game_pk",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
        ]
    ].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    home = games[["game_pk", "game_date", "home_team_id", "home_score", "away_score"]].rename(
        columns={"home_team_id": "team_id", "home_score": "runs_for", "away_score": "runs_against"}
    )
    away = games[["game_pk", "game_date", "away_team_id", "away_score", "home_score"]].rename(
        columns={"away_team_id": "team_id", "away_score": "runs_for", "home_score": "runs_against"}
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["won"] = (team_games["runs_for"] > team_games["runs_against"]).astype(int)
    team_games["_year"] = team_games["game_date"].dt.year
    team_games = team_games.sort_values(["team_id", "_year", "game_date", "game_pk"]).reset_index(drop=True)
    grouped = team_games.groupby(["team_id", "_year"], group_keys=False)
    team_games["prior_wins"] = grouped["won"].transform(lambda s: s.shift(1).expanding().sum())
    team_games["prior_games"] = grouped["won"].transform(lambda s: s.shift(1).expanding().count())
    team_games["prior_win_pct"] = team_games["prior_wins"] / team_games["prior_games"].replace(0, np.nan)
    return team_games[["game_pk", "team_id", "prior_win_pct", "prior_games"]]


def _add_prior_records(df: pd.DataFrame) -> pd.DataFrame:
    rec = _prior_record_features(df)
    out = df.copy()
    home = rec.rename(
        columns={
            "team_id": "home_team_id",
            "prior_win_pct": "home_prior_win_pct",
            "prior_games": "home_prior_games",
        }
    )
    away = rec.rename(
        columns={
            "team_id": "away_team_id",
            "prior_win_pct": "away_prior_win_pct",
            "prior_games": "away_prior_games",
        }
    )
    out = out.merge(home, on=["game_pk", "home_team_id"], how="left")
    out = out.merge(away, on=["game_pk", "away_team_id"], how="left")
    return out


def _pick_accuracy(df: pd.DataFrame, pick_home: pd.Series) -> float:
    return float((pick_home.astype(int) == df["target_home_win"].astype(int)).mean())


def _calibrated_pick_probability(train_df: pd.DataFrame, pick_home: pd.Series) -> float:
    acc = _pick_accuracy(train_df, pick_home)
    return float(np.clip(acc, 0.5001, 0.75))


def _pick_to_probability(pick_home: pd.Series, confidence: float) -> pd.Series:
    return pd.Series(np.where(pick_home, confidence, 1.0 - confidence), index=pick_home.index)


def _baseline_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[pd.DataFrame]:
    predictions = []

    home_conf = float(np.clip(train_df["target_home_win"].mean(), 0.5001, 0.75))
    home = _prediction_frame(eval_df, "home_team_baseline")
    home["p_home_win"] = home_conf
    predictions.append(home)

    train_pick = train_df["home_prior_win_pct"].fillna(0.5) >= train_df["away_prior_win_pct"].fillna(0.5)
    eval_pick = eval_df["home_prior_win_pct"].fillna(0.5) >= eval_df["away_prior_win_pct"].fillna(0.5)
    conf = _calibrated_pick_probability(train_df, train_pick)
    record = _prediction_frame(eval_df, "better_record_baseline")
    record["p_home_win"] = _pick_to_probability(eval_pick, conf)
    predictions.append(record)

    train_pick = train_df["home_wins_l10"].fillna(0) >= train_df["away_wins_l10"].fillna(0)
    eval_pick = eval_df["home_wins_l10"].fillna(0) >= eval_df["away_wins_l10"].fillna(0)
    conf = _calibrated_pick_probability(train_df, train_pick)
    last10 = _prediction_frame(eval_df, "better_last_10_baseline")
    last10["p_home_win"] = _pick_to_probability(eval_pick, conf)
    predictions.append(last10)

    if {"home_run_diff_l20", "away_run_diff_l20"}.issubset(eval_df.columns):
        train_pick = train_df["home_run_diff_l20"].fillna(0) >= train_df["away_run_diff_l20"].fillna(0)
        eval_pick = eval_df["home_run_diff_l20"].fillna(0) >= eval_df["away_run_diff_l20"].fillna(0)
        conf = _calibrated_pick_probability(train_df, train_pick)
        run_diff = _prediction_frame(eval_df, "better_run_differential_baseline")
        run_diff["p_home_win"] = _pick_to_probability(eval_pick, conf)
        predictions.append(run_diff)

    predictions.append(_elo_predictions(pd.concat([train_df, eval_df], ignore_index=True), eval_df))
    return predictions


def _elo_predictions(all_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[int, float] = {}
    rows = []
    eval_ids = set(eval_df["game_pk"].astype(int))
    for _, game in all_df.sort_values(["game_date", "game_pk"]).iterrows():
        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])
        home_rating = ratings.get(home_id, 1500.0)
        away_rating = ratings.get(away_id, 1500.0)
        p_home = 1.0 / (1.0 + 10.0 ** (-((home_rating + 35.0) - away_rating) / 400.0))

        if int(game["game_pk"]) in eval_ids:
            rows.append(
                {
                    "game_pk": game["game_pk"],
                    "game_date": game["game_date"],
                    "official_date": game.get("official_date"),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "target_home_win": game["target_home_win"],
                    "home_score": game.get("home_score"),
                    "away_score": game.get("away_score"),
                    "model_name": "elo_baseline",
                    "excluded": False,
                    "exclusion_reason": "",
                    "p_home_win": p_home,
                }
            )

        actual = float(game["target_home_win"])
        margin = abs(float(game.get("home_score", 0)) - float(game.get("away_score", 0)))
        if np.isnan(margin):
            margin = 1.0
        k = 20.0 * np.log1p(max(margin, 1.0))
        ratings[home_id] = home_rating + k * (actual - p_home)
        ratings[away_id] = away_rating + k * ((1.0 - actual) - (1.0 - p_home))
    return pd.DataFrame(rows)


def _annotate_predictions(preds: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "game_pk",
        "home_sp_unknown",
        "away_sp_unknown",
    ]
    meta = source_df[[col for col in meta_cols if col in source_df.columns]].copy()
    out = preds.merge(meta, on="game_pk", how="left")
    out["p_home_win"] = pd.to_numeric(out["p_home_win"], errors="coerce").fillna(0.5).clip(EPS, 1.0 - EPS)
    out["predicted_home_win"] = (out["p_home_win"] >= 0.5).astype(int)
    out["predicted_winner_team_id"] = np.where(
        out["predicted_home_win"] == 1,
        out["home_team_id"],
        out["away_team_id"],
    )
    out["winner_confidence"] = np.where(out["predicted_home_win"] == 1, out["p_home_win"], 1.0 - out["p_home_win"])
    out["correct"] = out["predicted_home_win"] == out["target_home_win"].astype(int)
    out["brier_loss"] = (out["p_home_win"] - out["target_home_win"].astype(float)) ** 2
    out["log_loss"] = -(
        out["target_home_win"].astype(float) * np.log(out["p_home_win"])
        + (1.0 - out["target_home_win"].astype(float)) * np.log(1.0 - out["p_home_win"])
    )
    out["month"] = pd.to_datetime(out["game_date"]).dt.to_period("M").astype(str)
    out["pitcher_availability"] = np.where(
        (out.get("home_sp_unknown", 1).fillna(1).astype(int) == 0)
        & (out.get("away_sp_unknown", 1).fillna(1).astype(int) == 0),
        "both_probable_pitchers_known",
        "one_or_more_probable_pitchers_missing",
    )
    return out


def _metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, grp in preds.groupby("model_name"):
        y = grp["target_home_win"].astype(int)
        p = grp["p_home_win"].clip(EPS, 1.0 - EPS)
        high = grp[grp["winner_confidence"] >= 0.60]
        rows.append(
            {
                "model_name": model_name,
                "accuracy": accuracy_score(y, grp["predicted_home_win"]),
                "brier_score": brier_score_loss(y, p),
                "log_loss": log_loss(y, p, labels=[0, 1]),
                "high_confidence_accuracy": high["correct"].mean() if len(high) else np.nan,
                "high_confidence_games": len(high),
                "games_predicted": len(grp),
                "games_excluded": int(grp["excluded"].sum()),
                "avg_winner_confidence": grp["winner_confidence"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def _confidence_buckets(preds: pd.DataFrame) -> pd.DataFrame:
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["50-60", "60-70", "70-80", "80-90", "90-100"]
    out = preds.copy()
    out["confidence_bucket"] = pd.cut(out["winner_confidence"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for (model_name, bucket), grp in out.groupby(["model_name", "confidence_bucket"], observed=False):
        if grp.empty:
            continue
        rows.append(
            {
                "model_name": model_name,
                "confidence_bucket": str(bucket),
                "games": len(grp),
                "avg_confidence": grp["winner_confidence"].mean(),
                "accuracy": grp["correct"].mean(),
                "calibration_gap_confidence_minus_accuracy": grp["winner_confidence"].mean() - grp["correct"].mean(),
                "brier_score": grp["brier_loss"].mean(),
                "log_loss": grp["log_loss"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _segment_metrics(preds: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for (model_name, segment), grp in preds.groupby(["model_name", segment_col]):
        rows.append(
            {
                "model_name": model_name,
                segment_col: segment,
                "games": len(grp),
                "accuracy": grp["correct"].mean(),
                "brier_score": grp["brier_loss"].mean(),
                "log_loss": grp["log_loss"].mean(),
                "avg_winner_confidence": grp["winner_confidence"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _team_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in preds.iterrows():
        rows.append(
            {
                "model_name": row["model_name"],
                "team_id": int(row["home_team_id"]),
                "side": "home",
                "game_pk": row["game_pk"],
                "team_won": int(row["target_home_win"]),
                "model_p_team_win": row["p_home_win"],
                "model_picked_team": int(row["predicted_home_win"] == 1),
                "model_correct": bool(row["correct"]),
            }
        )
        rows.append(
            {
                "model_name": row["model_name"],
                "team_id": int(row["away_team_id"]),
                "side": "away",
                "game_pk": row["game_pk"],
                "team_won": int(1 - int(row["target_home_win"])),
                "model_p_team_win": 1.0 - row["p_home_win"],
                "model_picked_team": int(row["predicted_home_win"] == 0),
                "model_correct": bool(row["correct"]),
            }
        )
    teams = pd.DataFrame(rows)
    return (
        teams.groupby(["model_name", "team_id"], as_index=False)
        .agg(
            games=("game_pk", "count"),
            accuracy_when_in_game=("model_correct", "mean"),
            avg_model_p_team_win=("model_p_team_win", "mean"),
            model_pick_rate=("model_picked_team", "mean"),
            actual_team_win_rate=("team_won", "mean"),
        )
        .sort_values(["model_name", "accuracy_when_in_game", "games"], ascending=[True, True, False])
    )


def _excluded_games(preds: pd.DataFrame) -> pd.DataFrame:
    cols = ["model_name", "game_pk", "game_date", "home_team_id", "away_team_id", "exclusion_reason"]
    excluded = preds[preds["excluded"]].copy()
    if excluded.empty:
        return pd.DataFrame(columns=cols)
    return excluded[cols]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    cols = list(df.columns)
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_summary(
    metrics: pd.DataFrame,
    confidence: pd.DataFrame,
    by_month: pd.DataFrame,
    by_team: pd.DataFrame,
    by_pitchers: pd.DataFrame,
    preds: pd.DataFrame,
    window: BacktestWindow,
    safe_features_used: int,
    safe_features_total: int,
    full_feature_matrix: bool,
) -> None:
    legacy = metrics[metrics["model_name"] == MODEL_MODE_LEGACY_FULL].iloc[0]
    safe = metrics[metrics["model_name"] == MODEL_MODE_PREGAME_SAFE].iloc[0]
    baselines = metrics[~metrics["model_name"].isin([MODEL_MODE_LEGACY_FULL, MODEL_MODE_PREGAME_SAFE])]
    best_baseline = baselines.sort_values("log_loss").iloc[0]

    accuracy_drop = legacy["accuracy"] - safe["accuracy"]
    brier_delta = safe["brier_score"] - legacy["brier_score"]
    logloss_delta = safe["log_loss"] - legacy["log_loss"]
    safe_beats_best_baseline = safe["log_loss"] < best_baseline["log_loss"]

    safe_conf = confidence[confidence["model_name"] == MODEL_MODE_PREGAME_SAFE]
    over_buckets = safe_conf[safe_conf["calibration_gap_confidence_minus_accuracy"] > 0.03]
    overall_conf_gap = safe["avg_winner_confidence"] - safe["accuracy"]
    overconfident = len(over_buckets) > 0 or overall_conf_gap > 0.02

    safe_month = by_month[by_month["model_name"] == MODEL_MODE_PREGAME_SAFE].sort_values("log_loss", ascending=False).head(5)
    safe_team = by_team[by_team["model_name"] == MODEL_MODE_PREGAME_SAFE].sort_values(
        ["accuracy_when_in_game", "games"], ascending=[True, False]
    ).head(8)
    safe_pitchers = by_pitchers[by_pitchers["model_name"] == MODEL_MODE_PREGAME_SAFE].sort_values("log_loss", ascending=False)

    table = _markdown_table(metrics)
    month_table = _markdown_table(safe_month)
    team_table = _markdown_table(safe_team)
    pitcher_table = _markdown_table(safe_pitchers)
    matrix_note = (
        "- Full all-feature historical matrix was used, with all audited safe features materialized."
        if full_feature_matrix
        else "- Heavy Statcast aggregate families and historical lineup-derived families were not rebuilt for this timed diagnostic run; absent legacy columns use the existing `legacy_full` missing-feature behavior."
    )

    md = f"""# Leakage-Safe Model Comparison Backtest

Generated: 2026-05-27

## Setup

- Training window: 2018 through {window.train_end}
- Early-stopping validation window: 2023 through {window.val_end}
- Evaluation window: {window.eval_start} through {window.eval_end}
- Split type: chronological only
- Odds: not used
- Pregame-safe probabilities: isotonic calibration fitted on the 2023 validation slice
- Evaluation games: {int(safe["games_predicted"])}
- Pregame-safe feature coverage in this local matrix: {safe_features_used} of {safe_features_total} audited safe features
{matrix_note}

## Model Comparison

{table}

## Explicit Answers

1. How much did performance drop after removing leakage?

Accuracy dropped by {accuracy_drop * 100:.2f} percentage points for `pregame_safe` versus `legacy_full`. Brier changed by {brier_delta:.4f}, and log loss changed by {logloss_delta:.4f}. Lower Brier/log loss is better.

2. Does pregame_safe beat the simple baselines?

`pregame_safe` {'beats' if safe_beats_best_baseline else 'does not beat'} the best simple baseline by log loss. Best baseline by log loss is `{best_baseline["model_name"]}` with log loss {best_baseline["log_loss"]:.4f}; `pregame_safe` log loss is {safe["log_loss"]:.4f}.

3. Is the model still overconfident?

`pregame_safe` is {'still materially overconfident in at least one confidence check' if overconfident else 'not materially overconfident by the bucket checks used here'}. Average winner confidence is {safe["avg_winner_confidence"]:.4f} versus accuracy {safe["accuracy"]:.4f}, a gap of {overall_conf_gap:.4f}.

4. Which segments perform worst?

Worst `pregame_safe` months by log loss:

{month_table}

Worst `pregame_safe` teams by accuracy when they appear:

{team_table}

Probable-pitcher availability:

{pitcher_table}

5. What should be fixed next?

- Keep `pregame_safe` as the default candidate for deployable pregame evaluation.
- Add timestamped first-pitch source freshness to the feature matrix so doubleheaders and late lineup changes can be audited at the game level.
- Replace excluded lineup matchup and pitch-quality families with true as-of snapshots before allowing them back into a pregame model.
- Review the worst months/teams above for missing pitcher data, schedule quirks, and systematic team-specific bias.
- Keep the calibration layer in place and monitor it against fresh audited prediction rows before tuning features.

## Output Files

- `model_comparison_metrics.csv`
- `performance_by_confidence_bucket.csv`
- `performance_by_month.csv`
- `performance_by_team.csv`
- `performance_by_pitcher_availability.csv`
- `worst_50_pregame_safe_predictions.csv`
- `excluded_game_reasons.csv`
"""
    (OUT_DIR / "model_comparison_summary.md").write_text(md, encoding="utf-8")


def run(rebuild_matrix: bool = False, full_feature_matrix: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_full_matrix(rebuild_matrix) if full_feature_matrix else _load_matrix(rebuild_matrix)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["target_home_win"]).copy()
    df = _add_prior_records(df)
    safe_features_used, safe_features_total = _write_safe_features_used(df)

    window = BacktestWindow()
    train_df = df[df["game_date"] <= pd.Timestamp(window.train_end)].copy()
    val_df = df[
        (df["game_date"] > pd.Timestamp(window.train_end))
        & (df["game_date"] <= pd.Timestamp(window.val_end))
    ].copy()
    eval_df = df[
        (df["game_date"] >= pd.Timestamp(window.eval_start))
        & (df["game_date"] <= pd.Timestamp(window.eval_end))
    ].copy()
    if train_df.empty or val_df.empty or eval_df.empty:
        raise ValueError(
            f"Backtest split is empty: train={len(train_df)} val={len(val_df)} eval={len(eval_df)}"
        )

    logger.info("split sizes train=%d val=%d eval=%d", len(train_df), len(val_df), len(eval_df))
    legacy_model = _train_model(train_df, val_df, MODEL_MODE_LEGACY_FULL)
    safe_model = _train_model(train_df, val_df, MODEL_MODE_PREGAME_SAFE)

    pred_frames = [
        _model_predictions(legacy_model, eval_df, MODEL_MODE_LEGACY_FULL),
        _model_predictions(safe_model, eval_df, MODEL_MODE_PREGAME_SAFE),
        *_baseline_predictions(train_df, eval_df),
    ]
    preds = pd.concat(pred_frames, ignore_index=True)
    preds = _annotate_predictions(preds, eval_df)

    metrics = _metrics(preds)
    confidence = _confidence_buckets(preds)
    by_month = _segment_metrics(preds, "month")
    by_pitchers = _segment_metrics(preds, "pitcher_availability")
    by_team = _team_metrics(preds)
    excluded = _excluded_games(preds)
    worst_safe = (
        preds[preds["model_name"] == MODEL_MODE_PREGAME_SAFE]
        .sort_values("brier_loss", ascending=False)
        .head(50)
    )

    metrics.to_csv(OUT_DIR / "model_comparison_metrics.csv", index=False)
    confidence.to_csv(OUT_DIR / "performance_by_confidence_bucket.csv", index=False)
    by_month.to_csv(OUT_DIR / "performance_by_month.csv", index=False)
    by_team.to_csv(OUT_DIR / "performance_by_team.csv", index=False)
    by_pitchers.to_csv(OUT_DIR / "performance_by_pitcher_availability.csv", index=False)
    worst_safe.to_csv(OUT_DIR / "worst_50_pregame_safe_predictions.csv", index=False)
    excluded.to_csv(OUT_DIR / "excluded_game_reasons.csv", index=False)
    _write_summary(
        metrics,
        confidence,
        by_month,
        by_team,
        by_pitchers,
        preds,
        window,
        safe_features_used,
        safe_features_total,
        full_feature_matrix,
    )
    shutil.copy2(PROCESSED_DIR / "leakage_safe_feature_cols.json", OUT_DIR / "leakage_safe_feature_cols.json")
    logger.info("wrote backtest outputs to %s", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-matrix", action="store_true", help="Rebuild the cached backtest feature matrix")
    parser.add_argument(
        "--full-feature-matrix",
        action="store_true",
        help="Build and use the all-feature historical matrix with resumable stage caches.",
    )
    args = parser.parse_args()
    configure_logging()
    run(rebuild_matrix=args.rebuild_matrix, full_feature_matrix=args.full_feature_matrix)


if __name__ == "__main__":
    main()

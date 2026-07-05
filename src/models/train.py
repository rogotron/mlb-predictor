"""Train models on the assembled feature matrix.

Default model: LightGBM binary classifier for home_win, with a separate
regressor for total_runs. Easy to swap.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from src.models.calibration import fit_probability_calibrator
from src.models.feature_config import (
    DEFAULT_MODEL_MODE,
    LEAKAGE_SAFE_FEATURE_COLS_PATH,
    MODEL_MODE_PREGAME_SAFE,
    get_model_feature_cols,
    model_artifact_name,
    validate_model_mode,
)
from src.models.feature_groups import group_importances, group_name_and_source

logger = logging.getLogger(__name__)


def write_feature_importance_artifact(
    model,
    name: str,
    model_mode: str,
    model_dir: Path,
) -> Path:
    """Write models/feature_importance_{artifact}.json for dashboard/CLI use.

    Captures the real LightGBM split-gain importances at train time — feature
    name, gain, share, and source group — so factor transparency always
    reflects the deployed model rather than a hand-maintained list.
    """
    names = list(getattr(model, "feature_name_", []) or [])
    gains = [float(g) for g in getattr(model, "feature_importances_", [])]
    total = sum(gains) or 1.0

    features = sorted(
        (
            {
                "feature": feat,
                "gain": gain,
                "pct": round(gain / total * 100, 3),
                "group": group_name_and_source(feat)[0],
                "source": group_name_and_source(feat)[1],
            }
            for feat, gain in zip(names, gains, strict=False)
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    groups, _ = group_importances(names, gains)

    artifact = model_artifact_name(name, model_mode)
    out_path = model_dir / f"feature_importance_{artifact}.json"
    payload = {
        "model": name,
        "mode": model_mode,
        "featureCount": len(names),
        "importanceMetric": "lightgbm_split_gain",
        "generatedAt": datetime.now().isoformat(),
        "groups": [
            {"name": g["name"], "source": g["source"], "pct": g["pct"]}
            for g in groups
        ],
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote feature importance artifact %s", out_path)
    return out_path

# Team rolling features (30)
_TEAM_FEATURE_COLS = [
    "home_wins_l5", "home_run_diff_l5", "home_avg_runs_for_l5", "home_avg_runs_against_l5", "home_win_pct_l5",
    "home_wins_l10", "home_run_diff_l10", "home_avg_runs_for_l10", "home_avg_runs_against_l10", "home_win_pct_l10",
    "home_wins_l20", "home_run_diff_l20", "home_avg_runs_for_l20", "home_avg_runs_against_l20", "home_win_pct_l20",
    "away_wins_l5", "away_run_diff_l5", "away_avg_runs_for_l5", "away_avg_runs_against_l5", "away_win_pct_l5",
    "away_wins_l10", "away_run_diff_l10", "away_avg_runs_for_l10", "away_avg_runs_against_l10", "away_win_pct_l10",
    "away_wins_l20", "away_run_diff_l20", "away_avg_runs_for_l20", "away_avg_runs_against_l20", "away_win_pct_l20",
]

# Stats API gamelog starter features: rolling 3-start (14) + season-to-date (14)
_SP_FEATURE_COLS = [
    # Rolling 3-start averages
    "home_sp_era_l3", "home_sp_whip_l3", "home_sp_k_per_9_l3",
    "home_sp_bb_per_9_l3", "home_sp_k_minus_bb_pct_l3", "home_sp_hr_per_9_l3",
    "home_sp_ip_per_start_l3",
    "away_sp_era_l3", "away_sp_whip_l3", "away_sp_k_per_9_l3",
    "away_sp_bb_per_9_l3", "away_sp_k_minus_bb_pct_l3", "away_sp_hr_per_9_l3",
    "away_sp_ip_per_start_l3",
    # Season-to-date cumulative rates
    "home_sp_era_std", "home_sp_whip_std", "home_sp_k_per_9_std",
    "home_sp_bb_per_9_std", "home_sp_k_minus_bb_pct_std", "home_sp_hr_per_9_std",
    "home_sp_ip_total_std",
    "away_sp_era_std", "away_sp_whip_std", "away_sp_k_per_9_std",
    "away_sp_bb_per_9_std", "away_sp_k_minus_bb_pct_std", "away_sp_hr_per_9_std",
    "away_sp_ip_total_std",
]

# Schedule / rest context (4)
_SCHEDULE_FEATURE_COLS = [
    "home_days_rest", "away_days_rest",
    "home_sp_days_rest", "away_sp_days_rest",
]

# Probable/starting pitcher availability and sample-size context (8)
_SP_AVAILABILITY_FEATURE_COLS = [
    "home_sp_season_starts_prior", "home_sp_recent_starts_l60d",
    "home_sp_short_history", "home_sp_unknown",
    "away_sp_season_starts_prior", "away_sp_recent_starts_l60d",
    "away_sp_short_history", "away_sp_unknown",
]

# Park factors — one per venue, not side-specific (2)
_PARK_FEATURE_COLS = [
    "pf_runs", "pf_hr",
]

# Season-to-date team rates and home/away splits (8)
_TEAM_STD_FEATURE_COLS = [
    "home_runs_per_game_std", "home_ra_per_game_std",
    "away_runs_per_game_std", "away_ra_per_game_std",
    "home_win_pct_home_std",   # home team's win% in home games this season
    "away_win_pct_away_std",   # away team's win% in road games this season
]

# SP Statcast features: rolling 3-start xwOBA against, whiff, barrel, platoon splits (14)
_SP_STATCAST_FEATURE_COLS = [
    "home_sp_xwoba_against_l3", "home_sp_whiff_rate_l3", "home_sp_barrel_rate_l3",
    "home_sp_xwoba_against_vs_L_l3", "home_sp_xwoba_against_vs_R_l3",
    "home_sp_whiff_rate_vs_L_l3", "home_sp_whiff_rate_vs_R_l3",
    "away_sp_xwoba_against_l3", "away_sp_whiff_rate_l3", "away_sp_barrel_rate_l3",
    "away_sp_xwoba_against_vs_L_l3", "away_sp_xwoba_against_vs_R_l3",
    "away_sp_whiff_rate_vs_L_l3", "away_sp_whiff_rate_vs_R_l3",
]

# Team batting Statcast: rolling 10-game offensive xwOBA and barrel rate (4)
_TEAM_HIT_FEATURE_COLS = [
    "home_xwoba_off_l10", "home_barrel_rate_off_l10",
    "away_xwoba_off_l10", "away_barrel_rate_off_l10",
]

# Lineup matchup: batter splits vs SP handedness + BvP (4)
_LINEUP_FEATURE_COLS = [
    "home_lineup_xwoba_vs_sp",  # home batters' prior-year xwOBA vs away SP's hand
    "away_lineup_xwoba_vs_sp",  # away batters' prior-year xwOBA vs home SP's hand
    "home_bvp_xwoba",           # home lineup weighted BvP xwOBA vs away SP
    "away_bvp_xwoba",           # away lineup weighted BvP xwOBA vs home SP
]

# Posted lineup Statcast features with team-level fallback (13)
_POSTED_LINEUP_FEATURE_COLS = [
    "home_lineup_xwoba_vs_hand_L30", "home_lineup_xwoba_weighted",
    "home_lineup_xwoba_top5", "home_lineup_barrel_rate_vs_hand_L30",
    "home_lineup_barrel_rate_weighted", "home_lineup_barrel_rate_top5",
    "away_lineup_xwoba_vs_hand_L30", "away_lineup_xwoba_weighted",
    "away_lineup_xwoba_top5", "away_lineup_barrel_rate_vs_hand_L30",
    "away_lineup_barrel_rate_weighted", "away_lineup_barrel_rate_top5",
    "lineup_features_missing",
]

# Bullpen rolling quality and short-term fatigue/availability (22)
_BULLPEN_FEATURE_COLS = [
    "home_bullpen_xwoba_against_l14", "home_bullpen_whiff_rate_l14",
    "home_bullpen_barrel_rate_l14", "home_bullpen_pitches_l3d",
    "home_bullpen_games_l3d",
    "home_bullpen_pitches_l1d", "home_bullpen_games_l1d",
    "home_bullpen_pitches_l2d", "home_bullpen_games_l2d",
    "home_bullpen_back_to_back_l2d", "home_bullpen_heavy_work_l2d",
    "away_bullpen_xwoba_against_l14", "away_bullpen_whiff_rate_l14",
    "away_bullpen_barrel_rate_l14", "away_bullpen_pitches_l3d",
    "away_bullpen_games_l3d",
    "away_bullpen_pitches_l1d", "away_bullpen_games_l1d",
    "away_bullpen_pitches_l2d", "away_bullpen_games_l2d",
    "away_bullpen_back_to_back_l2d", "away_bullpen_heavy_work_l2d",
]

# Baseball Savant pitch-arsenal quality for starters and bullpens (16)
_PITCH_QUALITY_FEATURE_COLS = [
    "home_sp_rv_per_100", "home_sp_xwoba_arsenal", "home_sp_whiff_arsenal",
    "away_sp_rv_per_100", "away_sp_xwoba_arsenal", "away_sp_whiff_arsenal",
    "home_bp_rv_per_100_weighted", "home_bp_xwoba_arsenal_weighted",
    "home_bp_whiff_arsenal_weighted",
    "away_bp_rv_per_100_weighted", "away_bp_xwoba_arsenal_weighted",
    "away_bp_whiff_arsenal_weighted",
    "home_sp_pitch_quality_missing", "away_sp_pitch_quality_missing",
    "home_bp_pitch_quality_missing", "away_bp_pitch_quality_missing",
]

_MISSINGNESS_SOURCE_COLS = [
    "home_sp_era_l3", "away_sp_era_l3",
    "home_sp_xwoba_against_l3", "away_sp_xwoba_against_l3",
    "home_sp_days_rest", "away_sp_days_rest",
    "home_lineup_xwoba_vs_sp", "away_lineup_xwoba_vs_sp",
    "home_lineup_xwoba_vs_hand_L30", "away_lineup_xwoba_vs_hand_L30",
    "home_bvp_xwoba", "away_bvp_xwoba",
    "home_bullpen_xwoba_against_l14", "away_bullpen_xwoba_against_l14",
    "home_sp_rv_per_100", "away_sp_rv_per_100",
    "home_bp_rv_per_100_weighted", "away_bp_rv_per_100_weighted",
]

_MISSINGNESS_FEATURE_COLS = [f"{col}_missing" for col in _MISSINGNESS_SOURCE_COLS]

FEATURE_COLS = (
    _TEAM_FEATURE_COLS
    + _SP_FEATURE_COLS
    + _SCHEDULE_FEATURE_COLS
    + _SP_AVAILABILITY_FEATURE_COLS
    + _PARK_FEATURE_COLS
    + _TEAM_STD_FEATURE_COLS
    + _SP_STATCAST_FEATURE_COLS
    + _TEAM_HIT_FEATURE_COLS
    + _POSTED_LINEUP_FEATURE_COLS
    + _LINEUP_FEATURE_COLS
    + _BULLPEN_FEATURE_COLS
    + _PITCH_QUALITY_FEATURE_COLS
    + _MISSINGNESS_FEATURE_COLS
)


def add_missing_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit missing-data indicators before numeric imputation."""
    out = df.copy()
    for col in _MISSINGNESS_SOURCE_COLS:
        out[f"{col}_missing"] = out[col].isna().astype("int8") if col in out.columns else 1
    return out


def get_features(
    df: pd.DataFrame,
    *,
    model_mode: str = DEFAULT_MODEL_MODE,
    safe_features_path: Path = LEAKAGE_SAFE_FEATURE_COLS_PATH,
) -> pd.DataFrame:
    """Extract feature matrix, handling missing values."""
    mode = validate_model_mode(model_mode)
    df = add_missing_indicator_features(df)
    feature_cols = get_model_feature_cols(
        mode,
        legacy_feature_cols=FEATURE_COLS,
        safe_features_path=safe_features_path,
    )
    missing = [col for col in feature_cols if col not in df.columns]
    if mode == MODEL_MODE_PREGAME_SAFE and missing:
        sample = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(f"Missing required pregame_safe features: {sample}{more}")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = float("nan")
    x = df[feature_cols].copy()
    # Fill NaN with 0 (no prior data = neutral)
    x = x.fillna(0)
    return x


def time_based_split(df: pd.DataFrame, train_end: str, val_end: str) -> tuple:
    """Split chronologically. Returns (train, val, test) DataFrames.

    Example: train_end='2022-12-31', val_end='2023-12-31'.
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train = df[df["game_date"] <= train_end_dt].copy()
    val = df[(df["game_date"] > train_end_dt) & (df["game_date"] <= val_end_dt)].copy()
    test = df[df["game_date"] > val_end_dt].copy()

    return train, val, test


def train_home_win_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
    *,
    model_mode: str = DEFAULT_MODEL_MODE,
    calibrate: bool | None = None,
) -> dict:
    """Train the binary home-win classifier.

    Args:
        train_df: feature matrix with target_home_win column
        val_df: held-out validation slice (later seasons)
        model_dir: where to save the artifact

    Returns:
        dict with keys: model_path, val_log_loss, val_brier, val_accuracy
    """
    model_mode = validate_model_mode(model_mode)
    logger.info("training home-win model mode=%s", model_mode)

    x_train = get_features(train_df, model_mode=model_mode)
    y_train = train_df["target_home_win"]
    x_val = get_features(val_df, model_mode=model_mode)
    y_val = val_df["target_home_win"]

    # LightGBM binary classifier — params from grid search (tune_hyperparams.py)
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
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    should_calibrate = model_mode == MODEL_MODE_PREGAME_SAFE if calibrate is None else calibrate
    calibration_metrics: dict[str, float | str | bool] = {"calibrated": bool(should_calibrate)}
    if should_calibrate:
        model = fit_probability_calibrator(model, x_val, y_val)
        calibration_metrics.update(
            {
                "calibration_method": model.calibration_method,
                "calibration_log_loss_before": model.calibration_log_loss_before,
                "calibration_log_loss_after": model.calibration_log_loss_after,
                "calibration_brier_before": model.calibration_brier_before,
                "calibration_brier_after": model.calibration_brier_after,
            }
        )

    # Predictions
    y_pred_proba = model.predict_proba(x_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    val_log_loss = log_loss(y_val, y_pred_proba)
    val_brier = brier_score_loss(y_val, y_pred_proba)
    val_accuracy = accuracy_score(y_val, y_pred)

    logger.info("home_win metrics: log_loss=%.4f brier=%.4f accuracy=%.4f",
                val_log_loss, val_brier, val_accuracy)

    # Save model
    today = datetime.now().strftime("%Y%m%d")
    artifact = model_artifact_name("home_win", model_mode)
    model_path = model_dir / f"{artifact}_{today}.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, model_path)

    # Also save as "latest"
    latest_path = model_dir / f"{artifact}_latest.pkl"
    pd.to_pickle(model, latest_path)

    write_feature_importance_artifact(model, "home_win", model_mode, model_dir)

    return {
        "model_path": str(model_path),
        "model_mode": model_mode,
        "feature_count": len(x_train.columns),
        "val_log_loss": val_log_loss,
        "val_brier": val_brier,
        "val_accuracy": val_accuracy,
        **calibration_metrics,
    }


def train_total_runs_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
    *,
    model_mode: str = DEFAULT_MODEL_MODE,
) -> dict:
    """Train a regressor for total game runs."""
    model_mode = validate_model_mode(model_mode)
    logger.info("training total-runs model mode=%s", model_mode)

    # Drop rows with NaN targets
    train_df = train_df.dropna(subset=["target_total_runs"])
    val_df = val_df.dropna(subset=["target_total_runs"])

    x_train = get_features(train_df, model_mode=model_mode)
    y_train = train_df["target_total_runs"]
    x_val = get_features(val_df, model_mode=model_mode)
    y_val = val_df["target_total_runs"]

    # LightGBM regressor — params from grid search (tune_hyperparams.py)
    model = lgb.LGBMRegressor(
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
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Predictions
    y_pred = model.predict(x_val)

    # Metrics
    val_mae = mean_absolute_error(y_val, y_pred)

    logger.info("total_runs metrics: MAE=%.4f", val_mae)

    # Save model
    today = datetime.now().strftime("%Y%m%d")
    artifact = model_artifact_name("total_runs", model_mode)
    model_path = model_dir / f"{artifact}_{today}.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, model_path)

    # Also save as "latest"
    latest_path = model_dir / f"{artifact}_latest.pkl"
    pd.to_pickle(model, latest_path)

    write_feature_importance_artifact(model, "total_runs", model_mode, model_dir)

    return {
        "model_path": str(model_path),
        "model_mode": model_mode,
        "feature_count": len(x_train.columns),
        "val_mae": val_mae,
    }

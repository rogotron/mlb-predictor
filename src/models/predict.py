"""Inference: load latest model and score today's slate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.as_of import (
    AS_OF_VALIDATION_REPORT_PATH,
    STATIC_FEATURE_ALLOWLIST_PATH,
    validate_feature_as_of_timestamps,
)
from src.models.feature_config import (
    DEFAULT_MODEL_MODE,
    LEAKAGE_SAFE_FEATURE_COLS_PATH,
    MODEL_MODE_PREGAME_SAFE,
    get_model_feature_cols,
    model_artifact_name,
    validate_model_mode,
)

# Feature columns — must stay in sync with train.py
from src.models.train import (  # noqa: F401  (re-exported)
    FEATURE_COLS,
    add_missing_indicator_features,
)


def load_latest_model(
    model_dir: Path,
    name: str,
    *,
    model_mode: str = DEFAULT_MODEL_MODE,
):
    """Load models/{name}_latest from disk.

    Raises FileNotFoundError if no trained model exists yet.
    """
    artifact = model_artifact_name(name, model_mode)
    model_path = model_dir / f"{artifact}_latest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return pd.read_pickle(model_path)


def predict_slate(
    features: pd.DataFrame,
    model_dir: Path,
    *,
    model_mode: str = DEFAULT_MODEL_MODE,
    safe_features_path: Path = LEAKAGE_SAFE_FEATURE_COLS_PATH,
    prediction_timestamp: object | None = None,
    static_allowlist_path: Path = STATIC_FEATURE_ALLOWLIST_PATH,
    as_of_report_path: Path = AS_OF_VALIDATION_REPORT_PATH,
) -> pd.DataFrame:
    """Score a slate.

    Args:
        features: output of features.build.build_prediction_input

    Returns:
        DataFrame with one row per game and columns:
            game_pk, game_date, home_team_id, away_team_id,
            p_home_win, expected_total_runs
    """
    model_mode = validate_model_mode(model_mode)

    # Load models
    home_win_model = load_latest_model(model_dir, "home_win", model_mode=model_mode)
    total_runs_model = load_latest_model(model_dir, "total_runs", model_mode=model_mode)

    if model_mode == MODEL_MODE_PREGAME_SAFE:
        home_features = get_model_feature_cols(
            model_mode,
            legacy_feature_cols=FEATURE_COLS,
            safe_features_path=safe_features_path,
        )
        run_features = home_features
    else:
        # Use the feature names each model was actually trained on. This lets
        # prediction keep working when one model has been retrained before the
        # other or when feature sets diverge.
        home_features = getattr(home_win_model, "feature_name_", None) or FEATURE_COLS
        run_features = getattr(total_runs_model, "feature_name_", None) or FEATURE_COLS
        if hasattr(home_features, "tolist"):
            home_features = home_features.tolist()
        if hasattr(run_features, "tolist"):
            run_features = run_features.tolist()

    features = add_missing_indicator_features(features)

    missing = sorted((set(home_features) | set(run_features)) - set(features.columns))
    if model_mode == MODEL_MODE_PREGAME_SAFE and missing:
        sample = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(f"Missing required pregame_safe features: {sample}{more}")

    for col in missing:
        if col not in features.columns:
            features = features.copy()
            features[col] = float("nan")

    if model_mode == MODEL_MODE_PREGAME_SAFE:
        validate_feature_as_of_timestamps(
            features,
            sorted(set(home_features) | set(run_features)),
            prediction_timestamp=prediction_timestamp,
            static_allowlist_path=static_allowlist_path,
            report_path=as_of_report_path,
        )

    # Predictions
    # Use predict_proba for classifier to get probabilities
    p_home_win = home_win_model.predict_proba(features[home_features].fillna(0))[:, 1]
    expected_runs = total_runs_model.predict(features[run_features].fillna(0))

    # Build output
    result = features[["game_pk", "game_date"]].copy()
    result["p_home_win"] = p_home_win
    result["expected_total_runs"] = expected_runs

    return result

"""Helpers for preparing audited pregame prediction feature snapshots."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.data.update import assert_processed_games_fresh
from src.features.build import (
    build_bullpen_prediction_features,
    build_gamelog_pitcher_prediction_features,
    build_lineup_prediction_features,
    build_pitch_quality_prediction_features,
    build_pitcher_prediction_features,
    build_posted_lineup_prediction_features,
    build_prediction_input,
    build_team_statcast_prediction_features,
)
from src.models.as_of import add_default_as_of_timestamps
from src.models.feature_config import (
    DEFAULT_MODEL_MODE,
    MODEL_MODE_PREGAME_SAFE,
    get_model_feature_cols,
)
from src.models.train import FEATURE_COLS, add_missing_indicator_features


def attach_slate_metadata(features: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Carry first-pitch and identity metadata into the feature snapshot."""
    metadata_cols = [
        "game_pk",
        "scheduled_start_utc",
        "official_date",
        "home_team_id",
        "away_team_id",
        "venue_id",
    ]
    available = [col for col in metadata_cols if col in slate.columns]
    new_cols = [col for col in available if col == "game_pk" or col not in features.columns]
    if new_cols == ["game_pk"]:
        return features
    return features.merge(slate[new_cols], on="game_pk", how="left")


def prepare_pregame_feature_snapshot(
    features: pd.DataFrame,
    slate: pd.DataFrame,
    *,
    target_date: date,
    model_mode: str = DEFAULT_MODEL_MODE,
    as_of_timestamp: datetime | None = None,
) -> pd.DataFrame:
    """Prepare feature sidecars required by audited pregame-safe prediction.

    ``as_of_timestamp`` overrides the default end-of-prior-day stamp. The live
    slate builder passes the real current time so a slate can be built the
    evening before a game day, when the prior-day default would be future-dated.
    """
    if model_mode != MODEL_MODE_PREGAME_SAFE:
        return features

    feature_cols = get_model_feature_cols(
        model_mode,
        legacy_feature_cols=FEATURE_COLS,
    )
    out = attach_slate_metadata(features, slate)
    out = add_missing_indicator_features(out)
    return add_default_as_of_timestamps(
        out,
        feature_cols,
        target_date=target_date,
        timestamp=as_of_timestamp,
    )


def build_pregame_prediction_features(
    slate: pd.DataFrame,
    *,
    processed_dir,
    raw_dir,
    target_date: date,
    model_mode: str = DEFAULT_MODEL_MODE,
    as_of_timestamp: datetime | None = None,
) -> pd.DataFrame:
    """Build prediction features and prepare pregame-safe audit sidecars."""
    assert_processed_games_fresh(target_date, raw_dir=raw_dir, processed_dir=processed_dir)
    features = build_prediction_input(slate, processed_dir, target_date=target_date, raw_dir=raw_dir)

    def _merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        if extra.empty:
            return base
        new_cols = [col for col in extra.columns if col != "game_pk" and col not in base.columns]
        return base.merge(extra[["game_pk"] + new_cols], on="game_pk", how="left")

    features = _merge(
        features,
        build_gamelog_pitcher_prediction_features(slate, raw_dir, target_date=target_date),
    )
    features = _merge(features, build_pitcher_prediction_features(slate, raw_dir, target_date=target_date))
    features = _merge(
        features,
        build_team_statcast_prediction_features(slate, raw_dir, processed_dir, target_date=target_date),
    )
    features = _merge(features, build_posted_lineup_prediction_features(slate, raw_dir, target_date=target_date))
    features = _merge(
        features,
        build_bullpen_prediction_features(slate, raw_dir, processed_dir, target_date=target_date),
    )
    features = _merge(features, build_pitch_quality_prediction_features(slate, raw_dir, target_date=target_date))
    features = _merge(features, build_lineup_prediction_features(slate, raw_dir, target_date=target_date))
    return prepare_pregame_feature_snapshot(
        features,
        slate,
        target_date=target_date,
        model_mode=model_mode,
        as_of_timestamp=as_of_timestamp,
    )

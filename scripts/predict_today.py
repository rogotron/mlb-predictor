"""Score today's MLB slate using the latest trained model.

Example:
    python scripts/predict_today.py
    python scripts/predict_today.py --date 2025-04-15 --out predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.data.update import fetch_today_slate, today_in_schedule_timezone
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
from src.models.audit import append_prediction_audit
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.predict import predict_slate
from src.models.pregame import prepare_pregame_feature_snapshot
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--model-mode",
        choices=["legacy_full", "pregame_safe"],
        default=DEFAULT_MODEL_MODE,
        help="Feature configuration to score with.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()
    logger.info("predicting slate for %s", target)

    slate = fetch_today_slate(target)
    if slate.empty:
        logger.warning("no games on %s", target)
        return

    team_features = build_prediction_input(slate, PROCESSED_DIR, target_date=target, raw_dir=RAW_DIR)

    def _merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        if extra.empty:
            return base
        new_cols = [c for c in extra.columns if c != "game_pk" and c not in base.columns]
        return base.merge(extra[["game_pk"] + new_cols], on="game_pk", how="left")

    import pandas as pd

    # Gamelog SP features: traditional rate stats (ERA, WHIP, K/9, …)
    team_features = _merge(
        team_features,
        build_gamelog_pitcher_prediction_features(slate, RAW_DIR, target_date=target),
    )

    # Statcast SP features: xwOBA against, whiff rate, barrel rate, platoon splits
    team_features = _merge(
        team_features,
        build_pitcher_prediction_features(slate, RAW_DIR, target_date=target),
    )

    # Team batting Statcast features: rolling xwOBA offense and barrel rate
    team_features = _merge(
        team_features,
        build_team_statcast_prediction_features(slate, RAW_DIR, PROCESSED_DIR, target_date=target),
    )

    # Posted lineup batter quality features
    team_features = _merge(
        team_features,
        build_posted_lineup_prediction_features(slate, RAW_DIR, target_date=target),
    )

    # Bullpen quality and recent workload
    team_features = _merge(
        team_features,
        build_bullpen_prediction_features(slate, RAW_DIR, PROCESSED_DIR, target_date=target),
    )

    # FanGraphs Stuff+/Location+/Pitching+ for SP and bullpen
    team_features = _merge(
        team_features,
        build_pitch_quality_prediction_features(slate, RAW_DIR, target_date=target),
    )

    # Lineup matchup features: batter splits vs SP handedness + BvP
    team_features = _merge(
        team_features,
        build_lineup_prediction_features(slate, RAW_DIR, target_date=target),
    )

    prediction_timestamp = datetime.now(tz=ZoneInfo("UTC"))
    team_features = prepare_pregame_feature_snapshot(
        team_features,
        slate,
        target_date=target,
        model_mode=args.model_mode,
    )

    preds = predict_slate(
        team_features,
        MODEL_DIR,
        model_mode=args.model_mode,
        prediction_timestamp=prediction_timestamp,
    )
    append_prediction_audit(
        slate=slate,
        predictions=preds,
        features=team_features,
        model_dir=MODEL_DIR,
        model_mode=args.model_mode,
    )

    # Sort by p_home_win for readability
    preds = preds.sort_values("p_home_win", ascending=False)
    print(preds.to_string(index=False))

    if args.out:
        preds.to_csv(args.out, index=False)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

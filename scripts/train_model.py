"""Train the home-win classifier and total-runs regressor.

Example:
    python scripts/train_model.py
    python scripts/train_model.py --train-end 2022 --val-end 2023 --test-end 2024
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.features.build import build_training_set
from src.models.feature_config import DEFAULT_MODEL_MODE
from src.models.train import (
    time_based_split,
    train_home_win_model,
    train_total_runs_model,
)
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--val-end", default="2024-12-31")
    parser.add_argument(
        "--model-mode",
        choices=["legacy_full", "pregame_safe"],
        default=DEFAULT_MODEL_MODE,
        help="Feature configuration to train with.",
    )
    parser.add_argument("--skip-runs-model", action="store_true")
    parser.add_argument(
        "--input-matrix",
        default=None,
        help="Optional cached parquet feature matrix to train from instead of rebuilding features.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    if args.input_matrix:
        logger.info("loading training matrix from %s", args.input_matrix)
        df = pd.read_parquet(args.input_matrix)
    else:
        logger.info("building training set from %s", PROCESSED_DIR)
        df = build_training_set(PROCESSED_DIR, raw_dir=RAW_DIR)

    train, val, test = time_based_split(df, args.train_end, args.val_end)
    logger.info("split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    win_metrics = train_home_win_model(train, val, MODEL_DIR, model_mode=args.model_mode)
    logger.info("home_win model: %s", win_metrics)

    if not args.skip_runs_model:
        runs_metrics = train_total_runs_model(train, val, MODEL_DIR, model_mode=args.model_mode)
        logger.info("total_runs model: %s", runs_metrics)


if __name__ == "__main__":
    main()

"""Evaluation: backtesting, calibration, and metrics.

For time series, the right backtest is walk-forward: train on everything up to
date T, predict day T+1, advance, repeat. Cheap approximation: evaluate on a
held-out future season (set in src/models/train.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def metrics_summary(y_true: pd.Series, y_prob: pd.Series) -> dict:
    """Standard binary metrics: log_loss, brier, accuracy, auc."""
    raise NotImplementedError


def calibration_table(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data. Columns: bin_lower, bin_upper, n, mean_pred, frac_pos."""
    raise NotImplementedError


def walk_forward_backtest(
    df: pd.DataFrame,
    train_start: str,
    test_start: str,
    refit_freq_days: int = 30,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest. Refits every refit_freq_days.

    Returns a DataFrame with one row per scored game:
        game_pk, game_date, y_true, y_prob.
    """
    raise NotImplementedError


def baseline_home_team(df: pd.DataFrame) -> dict:
    """Bar to beat: 'always pick home'. Reports the same metrics."""
    raise NotImplementedError

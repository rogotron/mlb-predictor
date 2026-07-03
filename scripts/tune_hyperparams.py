"""Grid search for LightGBM hyperparameters.

Searches over learning_rate, num_leaves, and min_child_samples using the
time-based train/val split. Prints a ranked results table and reports the
best configuration to paste into src/models/train.py.

Example:
    python scripts/tune_hyperparams.py
"""

from __future__ import annotations

import itertools
import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error

from src.features.build import build_training_set
from src.models.train import get_features, time_based_split
from src.utils.logging import configure_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"

GRID = {
    "learning_rate":     [0.01, 0.02, 0.05],
    "num_leaves":        [31, 63, 127],
    "min_child_samples": [10, 20],
    "max_depth":         [6],
    "subsample":         [0.8],
    "colsample_bytree":  [0.8],
}

FIXED = dict(n_estimators=1000, random_state=42, verbose=-1)


def _combinations(grid: dict) -> list[dict]:
    keys, values = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main() -> None:
    configure_logging()

    logger.info("building training set (2018-2025)")
    df = build_training_set(PROCESSED_DIR, raw_dir=RAW_DIR)
    train, val, _ = time_based_split(df, TRAIN_END, VAL_END)
    logger.info("train=%d  val=%d", len(train), len(val))

    X_tr = get_features(train); y_tr = train["target_home_win"]
    X_va = get_features(val);   y_va = val["target_home_win"]
    X_tr2 = get_features(train.dropna(subset=["target_total_runs"]))
    y_tr2 = train.dropna(subset=["target_total_runs"])["target_total_runs"]
    X_va2 = get_features(val.dropna(subset=["target_total_runs"]))
    y_va2 = val.dropna(subset=["target_total_runs"])["target_total_runs"]

    combos = _combinations(GRID)
    logger.info("searching %d hyperparameter combinations", len(combos))

    results = []
    for i, params in enumerate(combos, 1):
        full = {**FIXED, **params}

        clf = lgb.LGBMClassifier(**full)
        clf.fit(X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        ll = log_loss(y_va, clf.predict_proba(X_va)[:, 1])

        reg = lgb.LGBMRegressor(**full)
        reg.fit(X_tr2, y_tr2,
                eval_set=[(X_va2, y_va2)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        mae = mean_absolute_error(y_va2, reg.predict(X_va2))

        row = {**params, "val_log_loss": ll, "val_mae": mae,
               "win_iters": clf.best_iteration_, "run_iters": reg.best_iteration_}
        results.append(row)
        logger.info("[%d/%d] lr=%.3f leaves=%d min_child=%d  ll=%.4f mae=%.4f  iters=%d/%d",
                    i, len(combos),
                    params["learning_rate"], params["num_leaves"], params["min_child_samples"],
                    ll, mae, clf.best_iteration_, reg.best_iteration_)

    df_res = pd.DataFrame(results).sort_values("val_log_loss")
    print("\n=== RESULTS (sorted by val log_loss) ===")
    print(df_res.to_string(index=False))

    best = df_res.iloc[0]
    print("\n=== BEST CONFIGURATION ===")
    for k in GRID:
        print(f"    {k}={best[k]}")
    print(f"    val_log_loss={best['val_log_loss']:.4f}  val_mae={best['val_mae']:.4f}")
    print(f"    best_iter (win/run): {int(best['win_iters'])}/{int(best['run_iters'])}")


if __name__ == "__main__":
    main()

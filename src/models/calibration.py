"""Probability calibration helpers for model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6


@dataclass
class ProbabilityCalibratedClassifier:
    """Wrap a fitted classifier and calibrate its positive-class probability."""

    base_model: Any
    calibrator: IsotonicRegression
    calibration_method: str = "isotonic"
    calibration_log_loss_before: float | None = None
    calibration_log_loss_after: float | None = None
    calibration_brier_before: float | None = None
    calibration_brier_after: float | None = None

    @property
    def feature_name_(self) -> list[str]:
        names = getattr(self.base_model, "feature_name_", [])
        return names.tolist() if hasattr(names, "tolist") else list(names)

    @property
    def feature_importances_(self):
        return getattr(self.base_model, "feature_importances_", [])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.base_model.predict_proba(x)[:, 1], dtype=float)
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        calibrated = np.clip(calibrated, EPS, 1.0 - EPS)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def fit_probability_calibrator(
    base_model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    method: str = "isotonic",
) -> ProbabilityCalibratedClassifier:
    """Fit an out-of-sample probability calibrator on validation predictions."""
    if method != "isotonic":
        raise ValueError(f"Unsupported calibration method: {method}")

    y = y_val.astype(int)
    raw = np.asarray(base_model.predict_proba(x_val)[:, 1], dtype=float)
    raw = np.clip(raw, EPS, 1.0 - EPS)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
    calibrator.fit(raw, y)
    calibrated = np.clip(calibrator.predict(raw), EPS, 1.0 - EPS)

    return ProbabilityCalibratedClassifier(
        base_model=base_model,
        calibrator=calibrator,
        calibration_method=method,
        calibration_log_loss_before=float(log_loss(y, raw, labels=[0, 1])),
        calibration_log_loss_after=float(log_loss(y, calibrated, labels=[0, 1])),
        calibration_brier_before=float(brier_score_loss(y, raw)),
        calibration_brier_after=float(brier_score_loss(y, calibrated)),
    )

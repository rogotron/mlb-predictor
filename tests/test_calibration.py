from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.calibration import ProbabilityCalibratedClassifier, fit_probability_calibrator


class _DummyClassifier:
    feature_name_ = ["x"]
    feature_importances_ = [1]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = x["x"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def test_fit_probability_calibrator_wraps_classifier() -> None:
    x_val = pd.DataFrame({"x": [0.1, 0.2, 0.8, 0.9]})
    y_val = pd.Series([0, 0, 1, 1])

    calibrated = fit_probability_calibrator(_DummyClassifier(), x_val, y_val)

    assert isinstance(calibrated, ProbabilityCalibratedClassifier)
    assert calibrated.calibration_method == "isotonic"
    assert calibrated.feature_name_ == ["x"]
    probs = calibrated.predict_proba(pd.DataFrame({"x": [0.25, 0.75]}))[:, 1]
    assert probs[0] < probs[1]
    assert np.all((probs > 0) & (probs < 1))

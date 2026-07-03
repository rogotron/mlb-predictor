"""Feature eligibility and model-mode tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.feature_config import (
    MODEL_MODE_LEGACY_FULL,
    MODEL_MODE_PREGAME_SAFE,
    write_feature_eligibility_artifacts,
)
from src.models.train import FEATURE_COLS, get_features


def _inventory(path: Path) -> None:
    rows = [
        {
            "feature_name": "home_wins_l5",
            "source_file": "src/features/build.py",
            "source_data": "prior games",
            "timestamp_available": "before first pitch",
            "safe_before_first_pitch": "yes",
            "leakage_risk": "Safe",
            "explanation": "shifted prior-game window",
            "recommended_fix": "none",
        },
        {
            "feature_name": "home_lineup_xwoba_vs_hand_L30",
            "source_file": "src/features/build.py",
            "source_data": "historical lineups",
            "timestamp_available": "not proven",
            "safe_before_first_pitch": "no",
            "leakage_risk": "Possible leakage",
            "explanation": "historical lineup availability is not audited",
            "recommended_fix": "use pregame snapshots",
        },
        {
            "feature_name": "home_lineup_xwoba_vs_sp",
            "source_file": "src/features/build.py",
            "source_data": "same-game Statcast",
            "timestamp_available": "after game starts",
            "safe_before_first_pitch": "no",
            "leakage_risk": "Definite leakage",
            "explanation": "actual lineup derived from same-game plate appearances",
            "recommended_fix": "remove or replace with pregame source",
        },
        {
            "feature_name": "mystery_feature",
            "source_file": "unknown",
            "source_data": "unknown",
            "timestamp_available": "unknown",
            "safe_before_first_pitch": "no",
            "leakage_risk": "Unknown",
            "explanation": "not traced",
            "recommended_fix": "trace before use",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pregame_safe_artifacts_exclude_leaky_features(tmp_path: Path) -> None:
    inventory_path = tmp_path / "feature_inventory.csv"
    safe_path = tmp_path / "leakage_safe_feature_cols.json"
    report_path = tmp_path / "feature_eligibility_report.md"
    excluded_path = tmp_path / "excluded_features.csv"
    _inventory(inventory_path)

    safe_features = write_feature_eligibility_artifacts(
        inventory_path=inventory_path,
        safe_features_path=safe_path,
        report_path=report_path,
        excluded_features_path=excluded_path,
    )

    assert safe_features == ["home_wins_l5"]
    assert json.loads(safe_path.read_text(encoding="utf-8")) == ["home_wins_l5"]

    excluded = pd.read_csv(excluded_path)
    assert "home_lineup_xwoba_vs_sp" in set(excluded["feature_name"])
    assert "home_lineup_xwoba_vs_hand_L30" in set(excluded["feature_name"])
    assert "mystery_feature" in set(excluded["feature_name"])
    assert "home_wins_l5" not in set(excluded["feature_name"])
    assert "Pregame-safe features included | 1" in report_path.read_text(encoding="utf-8")


def test_pregame_safe_get_features_uses_only_safe_columns(tmp_path: Path) -> None:
    safe_path = tmp_path / "leakage_safe_feature_cols.json"
    safe_path.write_text(json.dumps(["home_wins_l5"]) + "\n", encoding="utf-8")
    df = pd.DataFrame(
        {
            "home_wins_l5": [3],
            "home_lineup_xwoba_vs_sp": [0.340],
            "home_lineup_xwoba_vs_hand_L30": [0.330],
        }
    )

    features = get_features(
        df,
        model_mode=MODEL_MODE_PREGAME_SAFE,
        safe_features_path=safe_path,
    )

    assert list(features.columns) == ["home_wins_l5"]
    assert "home_lineup_xwoba_vs_sp" not in features.columns
    assert "home_lineup_xwoba_vs_hand_L30" not in features.columns


def test_pregame_safe_fails_when_safe_feature_is_missing(tmp_path: Path) -> None:
    safe_path = tmp_path / "leakage_safe_feature_cols.json"
    safe_path.write_text(json.dumps(["home_wins_l5"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required pregame_safe features: home_wins_l5"):
        get_features(
            pd.DataFrame({"away_wins_l5": [2]}),
            model_mode=MODEL_MODE_PREGAME_SAFE,
            safe_features_path=safe_path,
        )


def test_legacy_full_mode_still_allows_full_feature_set() -> None:
    features = get_features(pd.DataFrame({"home_wins_l5": [1]}), model_mode=MODEL_MODE_LEGACY_FULL)

    assert list(features.columns) == FEATURE_COLS
    assert "home_lineup_xwoba_vs_sp" in features.columns
    assert "home_sp_rv_per_100" in features.columns

"""Feature eligibility and model-mode helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.utils.paths import PROCESSED_DIR, REPO_ROOT

MODEL_MODE_LEGACY_FULL = "legacy_full"
MODEL_MODE_PREGAME_SAFE = "pregame_safe"
DEFAULT_MODEL_MODE = MODEL_MODE_PREGAME_SAFE
VALID_MODEL_MODES = {MODEL_MODE_LEGACY_FULL, MODEL_MODE_PREGAME_SAFE}

FEATURE_INVENTORY_PATH = REPO_ROOT / "diagnostics" / "feature_inventory.csv"
FEATURE_ELIGIBILITY_REPORT_PATH = REPO_ROOT / "diagnostics" / "feature_eligibility_report.md"
EXCLUDED_FEATURES_PATH = REPO_ROOT / "diagnostics" / "excluded_features.csv"
LEAKAGE_SAFE_FEATURE_COLS_PATH = PROCESSED_DIR / "leakage_safe_feature_cols.json"

_TRUE_VALUES = {"true", "1", "yes", "y"}


def validate_model_mode(model_mode: str) -> str:
    """Return a normalized model mode or raise a clear error."""
    if model_mode not in VALID_MODEL_MODES:
        valid = ", ".join(sorted(VALID_MODEL_MODES))
        raise ValueError(f"Unknown model_mode={model_mode!r}; expected one of: {valid}")
    return model_mode


def _is_pre_first_pitch_available(value: object) -> bool:
    return str(value).strip().lower() in _TRUE_VALUES


def load_feature_inventory(path: Path = FEATURE_INVENTORY_PATH) -> pd.DataFrame:
    """Load the leakage audit feature inventory."""
    if not path.exists():
        raise FileNotFoundError(f"Feature inventory not found: {path}")
    inventory = pd.read_csv(path)
    required = {
        "feature_name",
        "source_file",
        "source_data",
        "timestamp_available",
        "safe_before_first_pitch",
        "leakage_risk",
        "explanation",
        "recommended_fix",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Feature inventory is missing required columns: {missing}")
    return inventory


def safe_features_from_inventory(inventory: pd.DataFrame) -> list[str]:
    """Return feature names eligible for pregame-safe model mode."""
    eligible = inventory[
        (inventory["leakage_risk"] == "Safe")
        & inventory["safe_before_first_pitch"].map(_is_pre_first_pitch_available)
    ]
    return eligible["feature_name"].dropna().astype(str).tolist()


def excluded_features_from_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return features excluded from pregame-safe mode, with reasons."""
    safe_names = set(safe_features_from_inventory(inventory))
    excluded = inventory[~inventory["feature_name"].isin(safe_names)].copy()
    excluded["exclusion_reason"] = excluded.apply(_exclusion_reason, axis=1)
    return excluded[
        [
            "feature_name",
            "leakage_risk",
            "safe_before_first_pitch",
            "exclusion_reason",
            "source_file",
            "explanation",
            "recommended_fix",
        ]
    ]


def write_feature_eligibility_artifacts(
    *,
    inventory_path: Path = FEATURE_INVENTORY_PATH,
    safe_features_path: Path = LEAKAGE_SAFE_FEATURE_COLS_PATH,
    report_path: Path = FEATURE_ELIGIBILITY_REPORT_PATH,
    excluded_features_path: Path = EXCLUDED_FEATURES_PATH,
) -> list[str]:
    """Write the pregame-safe feature list and human-readable eligibility report."""
    inventory = load_feature_inventory(inventory_path)
    safe_features = safe_features_from_inventory(inventory)
    excluded = excluded_features_from_inventory(inventory)

    safe_features_path.parent.mkdir(parents=True, exist_ok=True)
    safe_features_path.write_text(json.dumps(safe_features, indent=2) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_eligibility_report(inventory, safe_features, excluded), encoding="utf-8")

    excluded_features_path.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(excluded_features_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return safe_features


def load_leakage_safe_feature_cols(
    path: Path = LEAKAGE_SAFE_FEATURE_COLS_PATH,
    *,
    inventory_path: Path = FEATURE_INVENTORY_PATH,
    create_if_missing: bool = True,
) -> list[str]:
    """Load the pregame-safe feature list, generating it from the inventory if needed."""
    should_create = not path.exists()
    if path == LEAKAGE_SAFE_FEATURE_COLS_PATH:
        should_create = should_create or not EXCLUDED_FEATURES_PATH.exists()
        if inventory_path.exists() and path.exists():
            should_create = should_create or inventory_path.stat().st_mtime > path.stat().st_mtime

    if should_create:
        if not create_if_missing:
            raise FileNotFoundError(f"Leakage-safe feature list not found: {path}")
        write_feature_eligibility_artifacts(
            inventory_path=inventory_path,
            safe_features_path=path,
            report_path=FEATURE_ELIGIBILITY_REPORT_PATH,
            excluded_features_path=EXCLUDED_FEATURES_PATH,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Leakage-safe feature list must be a JSON array of strings: {path}")
    return data


def get_model_feature_cols(
    model_mode: str,
    *,
    legacy_feature_cols: Iterable[str],
    safe_features_path: Path = LEAKAGE_SAFE_FEATURE_COLS_PATH,
) -> list[str]:
    """Return the feature columns for a model mode."""
    mode = validate_model_mode(model_mode)
    if mode == MODEL_MODE_LEGACY_FULL:
        return list(legacy_feature_cols)
    return load_leakage_safe_feature_cols(safe_features_path)


def model_artifact_name(base_name: str, model_mode: str) -> str:
    """Return the persisted model artifact stem for a model mode."""
    mode = validate_model_mode(model_mode)
    if mode == MODEL_MODE_LEGACY_FULL:
        return base_name
    return f"{base_name}_{mode}"


def _exclusion_reason(row: pd.Series) -> str:
    risk = str(row.get("leakage_risk", "")).strip()
    pregame = _is_pre_first_pitch_available(row.get("safe_before_first_pitch"))
    if risk != "Safe":
        return f"leakage_risk={risk or 'missing'}"
    if not pregame:
        return "safe_before_first_pitch is not true"
    return "missing clear pre-first-pitch availability rule"


def _eligibility_report(
    inventory: pd.DataFrame,
    safe_features: list[str],
    excluded: pd.DataFrame,
) -> str:
    counts = inventory["leakage_risk"].value_counts().to_dict()
    safe_count = len(safe_features)
    excluded_count = len(excluded)
    excluded_by_reason = excluded["exclusion_reason"].value_counts().to_dict()

    reason_lines = "\n".join(
        f"| {reason} | {count} |" for reason, count in sorted(excluded_by_reason.items())
    )
    if not reason_lines:
        reason_lines = "| None | 0 |"

    return f"""# Feature Eligibility Report

Generated: 2026-05-25

## Model Modes

- `legacy_full`: Uses the existing `FEATURE_COLS` list for comparison only.
- `pregame_safe`: Uses only features with `leakage_risk == "Safe"` and `safe_before_first_pitch == true` in `diagnostics/feature_inventory.csv`.

## Eligibility Summary

| Category | Count |
|---|---:|
| Inventory features reviewed | {len(inventory)} |
| Pregame-safe features included | {safe_count} |
| Features excluded from pregame-safe mode | {excluded_count} |
| Safe in inventory | {counts.get("Safe", 0)} |
| Possible leakage in inventory | {counts.get("Possible leakage", 0)} |
| Definite leakage in inventory | {counts.get("Definite leakage", 0)} |
| Unknown in inventory | {counts.get("Unknown", 0)} |

## Exclusion Reasons

| Reason | Count |
|---|---:|
{reason_lines}

## Artifacts

- Safe feature list: `data/processed/leakage_safe_feature_cols.json`
- Excluded feature list: `diagnostics/excluded_features.csv`
- Source inventory: `diagnostics/feature_inventory.csv`

## Enforcement

In `pregame_safe` mode, training and prediction use exactly the safe feature list.
If any required safe feature is absent after missing-indicator generation, feature extraction raises a clear `ValueError`.
`legacy_full` mode remains available so old model runs can be compared without deleting or rewriting the existing model code.
"""

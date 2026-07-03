# Feature Eligibility Report

Generated: 2026-05-25

## Model Modes

- `legacy_full`: Uses the existing `FEATURE_COLS` list for comparison only.
- `pregame_safe`: Uses only features with `leakage_risk == "Safe"` and `safe_before_first_pitch == true` in `diagnostics/feature_inventory.csv`.

## Eligibility Summary

| Category | Count |
|---|---:|
| Inventory features reviewed | 169 |
| Pregame-safe features included | 126 |
| Features excluded from pregame-safe mode | 43 |
| Safe in inventory | 126 |
| Possible leakage in inventory | 15 |
| Definite leakage in inventory | 28 |
| Unknown in inventory | 0 |

## Exclusion Reasons

| Reason | Count |
|---|---:|
| leakage_risk=Definite leakage | 28 |
| leakage_risk=Possible leakage | 15 |

## Artifacts

- Safe feature list: `data/processed/leakage_safe_feature_cols.json`
- Excluded feature list: `diagnostics/excluded_features.csv`
- Source inventory: `diagnostics/feature_inventory.csv`

## Enforcement

In `pregame_safe` mode, training and prediction use exactly the safe feature list.
If any required safe feature is absent after missing-indicator generation, feature extraction raises a clear `ValueError`.
`legacy_full` mode remains available so old model runs can be compared without deleting or rewriting the existing model code.

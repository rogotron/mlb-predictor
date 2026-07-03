"""Prediction audit logging and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.update import SCHEDULE_TIMEZONE
from src.models.feature_config import DEFAULT_MODEL_MODE, model_artifact_name
from src.utils.paths import PROCESSED_DIR, REPO_ROOT

PREDICTION_AUDIT_PATH = PROCESSED_DIR / "prediction_audit.csv"
PREDICTION_AUDIT_REPORT_PATH = REPO_ROOT / "diagnostics" / "prediction_audit_report.md"

AUDIT_COLUMNS = [
    "game_id",
    "game_date",
    "prediction_timestamp",
    "prediction_date_et",
    "home_team",
    "away_team",
    "probable_home_pitcher",
    "probable_away_pitcher",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "confidence_bucket",
    "model_version",
    "features_snapshot_id",
    "data_freshness_status",
    "exclusion_reason",
    "scheduled_start_utc",
]

_PREDICTION_WINDOW = "min"
_PROB_TOLERANCE = 0.01


def confidence_bucket(home_win_probability: float) -> str:
    """Return the dashboard confidence bucket for a win probability."""
    edge = abs(home_win_probability - 0.5) * 200
    if edge < 25:
        return "MARGINAL"
    if edge < 40:
        return "LOW"
    if edge < 60:
        return "MODERATE"
    return "HIGH"


def model_version_id(model_dir: Path, *, model_mode: str = DEFAULT_MODEL_MODE) -> str:
    """Hash the latest model artifacts used to produce the prediction."""
    parts = []
    for base_name in ("home_win", "total_runs"):
        artifact = model_artifact_name(base_name, model_mode)
        name = f"{artifact}_latest.pkl"
        path = model_dir / name
        if not path.exists():
            parts.append(f"{name}=missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        parts.append(f"{name.removesuffix('.pkl')}={digest.hexdigest()[:12]}")
    return ";".join(parts)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def features_snapshot_id(feature_row: pd.Series | None) -> str:
    """Hash the feature values for one game into a stable snapshot ID."""
    if feature_row is None or feature_row.empty:
        return "missing"

    values = {
        str(key): _json_safe(value)
        for key, value in feature_row.sort_index().items()
        if str(key) != "game_pk"
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _now_et(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(SCHEDULE_TIMEZONE)
    if now.tzinfo is None:
        return now.replace(tzinfo=SCHEDULE_TIMEZONE)
    return now.astimezone(SCHEDULE_TIMEZONE)


def _parse_first_pitch(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SCHEDULE_TIMEZONE)
    return parsed.astimezone(SCHEDULE_TIMEZONE)


def _clean_text(value: Any, fallback: str = "TBD") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else fallback


def _team_name(row: pd.Series, side: str) -> str:
    return _clean_text(row.get(f"{side}_team_name"), str(row.get(f"{side}_team_id", side)))


def _freshness_status(slate_row: pd.Series, prediction_time_et: datetime) -> str:
    first_pitch = _parse_first_pitch(slate_row.get("scheduled_start_utc") or slate_row.get("game_date"))
    if first_pitch is None:
        return "missing_first_pitch"
    return "pre_first_pitch" if prediction_time_et < first_pitch else "after_first_pitch"


def _prediction_timestamp_window(timestamp: str) -> str:
    parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.floor(_PREDICTION_WINDOW).isoformat()


def _load_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    return pd.read_csv(path, dtype={"game_id": "string"})


def _key_text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _prediction_row(
    *,
    slate_row: pd.Series,
    pred_row: pd.Series | None,
    feature_row: pd.Series | None,
    prediction_time_et: datetime,
    model_version: str,
    exclusion_reason: str = "",
) -> dict[str, Any]:
    p_home = None if pred_row is None else float(pred_row.get("p_home_win"))
    p_away = None if p_home is None else 1.0 - p_home
    home_team = _team_name(slate_row, "home")
    away_team = _team_name(slate_row, "away")
    predicted_winner = ""
    bucket = ""
    if p_home is not None:
        predicted_winner = home_team if p_home >= p_away else away_team
        bucket = confidence_bucket(p_home)

    return {
        "game_id": str(_json_safe(slate_row.get("game_pk")) or ""),
        "game_date": str(_json_safe(slate_row.get("official_date")) or _json_safe(slate_row.get("game_date")) or ""),
        "prediction_timestamp": prediction_time_et.isoformat(),
        "prediction_date_et": prediction_time_et.date().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "probable_home_pitcher": _clean_text(slate_row.get("home_sp_name")),
        "probable_away_pitcher": _clean_text(slate_row.get("away_sp_name")),
        "home_win_probability": p_home,
        "away_win_probability": p_away,
        "predicted_winner": predicted_winner,
        "confidence_bucket": bucket,
        "model_version": model_version,
        "features_snapshot_id": features_snapshot_id(feature_row),
        "data_freshness_status": _freshness_status(slate_row, prediction_time_et),
        "exclusion_reason": exclusion_reason,
        "scheduled_start_utc": _json_safe(slate_row.get("scheduled_start_utc") or slate_row.get("game_date")),
    }


def append_prediction_audit(
    *,
    slate: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    model_dir: Path,
    audit_path: Path = PREDICTION_AUDIT_PATH,
    report_path: Path = PREDICTION_AUDIT_REPORT_PATH,
    now: datetime | None = None,
    model_version: str | None = None,
    model_mode: str = DEFAULT_MODEL_MODE,
    exclusions: dict[int | str, str] | None = None,
) -> pd.DataFrame:
    """Append new prediction audit rows without overwriting started games."""
    prediction_time_et = _now_et(now)
    version = model_version or model_version_id(model_dir, model_mode=model_mode)
    exclusions = exclusions or {}

    existing = _load_audit(audit_path)
    existing_keys = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            existing_keys.add(
                (
                    _key_text(row.get("game_id", "")),
                    _key_text(row.get("model_version", "")),
                    _key_text(row.get("features_snapshot_id", "")),
                    _key_text(row.get("exclusion_reason", "")),
                )
            )

    rows: list[dict[str, Any]] = []
    pred_by_game = {str(row["game_pk"]): row for _, row in predictions.iterrows()} if not predictions.empty else {}
    feat_by_game = {str(row["game_pk"]): row for _, row in features.iterrows()} if not features.empty else {}

    for _, slate_row in slate.iterrows():
        game_id = str(_json_safe(slate_row.get("game_pk")) or "")
        pred_row = pred_by_game.get(game_id)
        feature_row = feat_by_game.get(game_id)
        explicit_exclusion = exclusions.get(game_id) or exclusions.get(slate_row.get("game_pk"))

        if pred_row is None:
            row = _prediction_row(
                slate_row=slate_row,
                pred_row=None,
                feature_row=feature_row,
                prediction_time_et=prediction_time_et,
                model_version=version,
                exclusion_reason=explicit_exclusion or "model_prediction_missing",
            )
        else:
            row = _prediction_row(
                slate_row=slate_row,
                pred_row=pred_row,
                feature_row=feature_row,
                prediction_time_et=prediction_time_et,
                model_version=version,
                exclusion_reason=explicit_exclusion or "",
            )
            if row["data_freshness_status"] == "after_first_pitch" and not explicit_exclusion:
                continue

        key = (
            str(row["game_id"]),
            str(row["model_version"]),
            str(row["features_snapshot_id"]),
            str(row["exclusion_reason"]),
        )
        if key in existing_keys:
            continue
        existing_keys.add(key)
        rows.append(row)

    if rows:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out = out.reindex(columns=AUDIT_COLUMNS)
        out.to_csv(audit_path, index=False)
    else:
        out = existing.reindex(columns=AUDIT_COLUMNS)

    validate_prediction_audit(audit_path=audit_path, report_path=report_path)
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def validate_prediction_audit(
    *,
    audit_path: Path = PREDICTION_AUDIT_PATH,
    report_path: Path = PREDICTION_AUDIT_REPORT_PATH,
) -> dict[str, list[str]]:
    """Validate audit CSV and write a Markdown diagnostics report."""
    if not audit_path.exists():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    df = _load_audit(audit_path)
    issues: dict[str, list[str]] = {
        "missing_game_id": [],
        "duplicate_game_id_model_version_timestamp_window": [],
        "missing_probabilities": [],
        "probabilities_not_sum_to_one": [],
        "prediction_after_first_pitch": [],
    }

    if not df.empty:
        active = df[df["exclusion_reason"].fillna("").eq("")]
        issues["missing_game_id"] = [
            str(idx) for idx, row in df.iterrows() if not str(row.get("game_id", "")).strip()
        ]
        issues["missing_probabilities"] = [
            str(row.get("game_id", idx))
            for idx, row in active.iterrows()
            if pd.isna(row.get("home_win_probability")) or pd.isna(row.get("away_win_probability"))
        ]

        for idx, row in active.iterrows():
            home_p = pd.to_numeric(row.get("home_win_probability"), errors="coerce")
            away_p = pd.to_numeric(row.get("away_win_probability"), errors="coerce")
            if pd.isna(home_p) or pd.isna(away_p):
                continue
            if abs(float(home_p) + float(away_p) - 1.0) > _PROB_TOLERANCE:
                issues["probabilities_not_sum_to_one"].append(str(row.get("game_id", idx)))

        with_windows = active.copy()
        with_windows["_timestamp_window"] = with_windows["prediction_timestamp"].map(_prediction_timestamp_window)
        duplicates = with_windows[
            with_windows.duplicated(
                subset=["game_id", "model_version", "_timestamp_window"],
                keep=False,
            )
        ]
        issues["duplicate_game_id_model_version_timestamp_window"] = sorted(
            set(duplicates["game_id"].dropna().astype(str).tolist())
        )

        for idx, row in active.iterrows():
            first_pitch = _parse_first_pitch(row.get("scheduled_start_utc"))
            predicted_at = _parse_first_pitch(row.get("prediction_timestamp"))
            if first_pitch is None or predicted_at is None:
                continue
            if predicted_at >= first_pitch:
                issues["prediction_after_first_pitch"].append(str(row.get("game_id", idx)))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prediction Audit Report",
        "",
        f"- audit_file: `{audit_path}`",
        f"- generated_at_et: `{datetime.now(SCHEDULE_TIMEZONE).isoformat()}`",
        f"- rows_checked: {len(df)}",
        "",
        "## Checks",
    ]
    for name, values in issues.items():
        status = "PASS" if not values else "FAIL"
        lines.append(f"- {name}: {status} ({len(values)})")
        if values:
            lines.append(f"  - {', '.join(values[:20])}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return issues

"""As-of timestamp validation for pregame feature snapshots."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.paths import REPO_ROOT

AS_OF_SUFFIX = "__as_of_timestamp"
STATIC_FEATURE_ALLOWLIST_PATH = REPO_ROOT / "diagnostics" / "static_feature_allowlist.json"
AS_OF_VALIDATION_REPORT_PATH = REPO_ROOT / "diagnostics" / "as_of_validation_report.md"

DEFAULT_STATIC_FEATURE_ALLOWLIST = [
    "home_team_id",
    "away_team_id",
    "venue_id",
    "pf_runs",
    "pf_hr",
]

_ET = ZoneInfo("America/New_York")


def as_of_column(feature_name: str) -> str:
    """Return the timestamp sidecar column name for a model feature."""
    return f"{feature_name}{AS_OF_SUFFIX}"


def ensure_static_feature_allowlist(
    path: Path = STATIC_FEATURE_ALLOWLIST_PATH,
) -> list[str]:
    """Create the static feature allowlist artifact if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(DEFAULT_STATIC_FEATURE_ALLOWLIST, indent=2) + "\n",
            encoding="utf-8",
        )
    return load_static_feature_allowlist(path)


def load_static_feature_allowlist(path: Path = STATIC_FEATURE_ALLOWLIST_PATH) -> list[str]:
    """Load the static feature allowlist."""
    if not path.exists():
        return list(DEFAULT_STATIC_FEATURE_ALLOWLIST)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Static feature allowlist must be a JSON array of strings: {path}")
    return data


def default_prior_day_as_of_timestamp(target_date: date) -> datetime:
    """Use the last ET second of the prior day for cached historical features."""
    return datetime.combine(target_date - timedelta(days=1), time(23, 59, 59), tzinfo=_ET)


def add_default_as_of_timestamps(
    features: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    target_date: date,
    static_allowlist_path: Path = STATIC_FEATURE_ALLOWLIST_PATH,
    timestamp: datetime | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Attach prior-day as-of sidecars for feature columns that can be timestamped.

    This is intended for live pregame snapshots built from cached historical data.
    Static metadata remains exempt only through the explicit allowlist.
    """
    ensure_static_feature_allowlist(static_allowlist_path)
    static_features = set(load_static_feature_allowlist(static_allowlist_path))
    as_of_value = timestamp or default_prior_day_as_of_timestamp(target_date)
    out = features.copy()
    for feature in feature_cols:
        if feature in static_features or feature not in out.columns:
            continue
        col = as_of_column(feature)
        if col not in out.columns:
            out[col] = as_of_value
    return out


def validate_feature_as_of_timestamps(
    features: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    prediction_timestamp: datetime | str | pd.Timestamp | None = None,
    first_pitch_col: str = "scheduled_start_utc",
    fallback_first_pitch_col: str = "game_date",
    static_allowlist_path: Path = STATIC_FEATURE_ALLOWLIST_PATH,
    report_path: Path = AS_OF_VALIDATION_REPORT_PATH,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate that every non-static model feature is timestamp-clean."""
    prediction_ts = _to_utc_timestamp(prediction_timestamp or datetime.now(tz=ZoneInfo("UTC")))
    static_features = set(ensure_static_feature_allowlist(static_allowlist_path))
    issues: list[dict[str, object]] = []
    checked_features = list(dict.fromkeys(feature_cols))

    for row_idx, row in features.iterrows():
        game_id = row.get("game_pk", row.get("game_id", row_idx))
        first_pitch = _row_first_pitch(row, first_pitch_col, fallback_first_pitch_col)
        if pd.isna(first_pitch):
            issues.append(
                _issue(game_id, None, "missing_first_pitch_time", prediction_ts, first_pitch, None)
            )
        elif prediction_ts >= first_pitch:
            issues.append(
                _issue(
                    game_id,
                    None,
                    "prediction_not_before_first_pitch",
                    prediction_ts,
                    first_pitch,
                    None,
                )
            )

        for feature in checked_features:
            if feature in static_features:
                continue
            timestamp_col = as_of_column(feature)
            if timestamp_col not in features.columns:
                issues.append(
                    _issue(
                        game_id,
                        feature,
                        "missing_as_of_timestamp",
                        prediction_ts,
                        first_pitch,
                        None,
                    )
                )
                continue

            as_of_ts = _to_utc_timestamp(row.get(timestamp_col))
            if pd.isna(as_of_ts):
                issues.append(
                    _issue(
                        game_id,
                        feature,
                        "missing_as_of_timestamp",
                        prediction_ts,
                        first_pitch,
                        as_of_ts,
                    )
                )
                continue
            if as_of_ts > prediction_ts:
                issues.append(
                    _issue(
                        game_id,
                        feature,
                        "feature_after_prediction_timestamp",
                        prediction_ts,
                        first_pitch,
                        as_of_ts,
                    )
                )
            if not pd.isna(first_pitch) and as_of_ts >= first_pitch:
                issues.append(
                    _issue(
                        game_id,
                        feature,
                        "feature_after_first_pitch_time",
                        prediction_ts,
                        first_pitch,
                        as_of_ts,
                    )
                )

    issues_df = pd.DataFrame(issues)
    _write_validation_report(
        issues_df=issues_df,
        checked_features=checked_features,
        static_features=static_features,
        prediction_timestamp=prediction_ts,
        rows_validated=len(features),
        report_path=report_path,
    )

    if raise_on_error and not issues_df.empty:
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in Counter(issues_df["reason"]).most_common()
        )
        raise ValueError(f"As-of timestamp validation failed for pregame_safe: {reasons}")
    return issues_df


def _row_first_pitch(
    row: pd.Series,
    first_pitch_col: str,
    fallback_first_pitch_col: str,
) -> pd.Timestamp:
    if first_pitch_col in row and pd.notna(row.get(first_pitch_col)):
        return _to_utc_timestamp(row.get(first_pitch_col))
    if fallback_first_pitch_col in row and pd.notna(row.get(fallback_first_pitch_col)):
        return _to_utc_timestamp(row.get(fallback_first_pitch_col))
    return pd.NaT


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        raise TypeError("Expected scalar timestamp value")
    return ts


def _issue(
    game_id: object,
    feature_name: str | None,
    reason: str,
    prediction_timestamp: pd.Timestamp,
    first_pitch_time: pd.Timestamp,
    feature_as_of_timestamp: pd.Timestamp | None,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "feature_name": feature_name or "",
        "reason": reason,
        "prediction_timestamp": _format_ts(prediction_timestamp),
        "first_pitch_time": _format_ts(first_pitch_time),
        "feature_as_of_timestamp": _format_ts(feature_as_of_timestamp),
    }


def _format_ts(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return value.isoformat()


def _write_validation_report(
    *,
    issues_df: pd.DataFrame,
    checked_features: list[str],
    static_features: set[str],
    prediction_timestamp: pd.Timestamp,
    rows_validated: int,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(issues_df["reason"]) if not issues_df.empty else Counter()
    count_lines = "\n".join(f"| {reason} | {count} |" for reason, count in counts.items())
    if not count_lines:
        count_lines = "| None | 0 |"

    sample_lines = ""
    if not issues_df.empty:
        sample = issues_df.head(50)
        sample_lines = "\n".join(
            "| {game_id} | {feature_name} | {reason} | {feature_as_of_timestamp} | {first_pitch_time} |".format(
                **row
            )
            for row in sample.to_dict(orient="records")
        )
    if not sample_lines:
        sample_lines = "| None |  |  |  |  |"

    report_path.write_text(
        f"""# As-Of Validation Report

Generated: {datetime.now(tz=ZoneInfo("UTC")).isoformat()}
Prediction timestamp: {_format_ts(prediction_timestamp)}

## Summary

| Check | Value |
|---|---:|
| Rows validated | {rows_validated} |
| Feature columns checked | {len(checked_features)} |
| Static features exempted | {len(set(checked_features) & static_features)} |
| Issues found | {len(issues_df)} |

## Issue Counts

| Reason | Count |
|---|---:|
{count_lines}

## Sample Issues

| Game | Feature | Reason | Feature as-of | First pitch |
|---|---|---|---|---|
{sample_lines}
""",
        encoding="utf-8",
    )

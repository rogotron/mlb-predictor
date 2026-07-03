from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.models.as_of import (
    add_default_as_of_timestamps,
    as_of_column,
    validate_feature_as_of_timestamps,
)


def _features(**extra: object) -> pd.DataFrame:
    row = {
        "game_pk": 123,
        "game_date": "2026-05-24T23:05:00Z",
        "scheduled_start_utc": "2026-05-24T23:05:00Z",
        "home_wins_l10": 6,
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_as_of_validation_allows_valid_prior_day_feature(tmp_path: Path) -> None:
    features = _features(**{as_of_column("home_wins_l10"): "2026-05-23T23:59:59Z"})
    report_path = tmp_path / "as_of_validation_report.md"

    issues = validate_feature_as_of_timestamps(
        features,
        ["home_wins_l10"],
        prediction_timestamp=datetime(2026, 5, 24, 20, 0, tzinfo=UTC),
        static_allowlist_path=tmp_path / "static_feature_allowlist.json",
        report_path=report_path,
    )

    assert issues.empty
    assert "Issues found | 0" in report_path.read_text(encoding="utf-8")


def test_as_of_validation_fails_postgame_feature(tmp_path: Path) -> None:
    features = _features(**{as_of_column("home_wins_l10"): "2026-05-25T01:00:00Z"})

    with pytest.raises(ValueError, match="feature_after_first_pitch_time"):
        validate_feature_as_of_timestamps(
            features,
            ["home_wins_l10"],
            prediction_timestamp=datetime(2026, 5, 24, 20, 0, tzinfo=UTC),
            static_allowlist_path=tmp_path / "static_feature_allowlist.json",
            report_path=tmp_path / "as_of_validation_report.md",
        )


def test_as_of_validation_fails_missing_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_as_of_timestamp"):
        validate_feature_as_of_timestamps(
            _features(),
            ["home_wins_l10"],
            prediction_timestamp=datetime(2026, 5, 24, 20, 0, tzinfo=UTC),
            static_allowlist_path=tmp_path / "static_feature_allowlist.json",
            report_path=tmp_path / "as_of_validation_report.md",
        )


def test_as_of_validation_respects_static_feature_allowlist(tmp_path: Path) -> None:
    allowlist_path = tmp_path / "static_feature_allowlist.json"
    allowlist_path.write_text(json.dumps(["venue_id"]) + "\n", encoding="utf-8")
    features = _features(venue_id=15)

    issues = validate_feature_as_of_timestamps(
        features,
        ["venue_id"],
        prediction_timestamp=datetime(2026, 5, 24, 20, 0, tzinfo=UTC),
        static_allowlist_path=allowlist_path,
        report_path=tmp_path / "as_of_validation_report.md",
    )

    assert issues.empty


def test_add_default_as_of_timestamps_skips_static_features(tmp_path: Path) -> None:
    allowlist_path = tmp_path / "static_feature_allowlist.json"
    allowlist_path.write_text(json.dumps(["venue_id"]) + "\n", encoding="utf-8")

    stamped = add_default_as_of_timestamps(
        _features(venue_id=15),
        ["home_wins_l10", "venue_id"],
        target_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        static_allowlist_path=allowlist_path,
    )

    assert as_of_column("home_wins_l10") in stamped.columns
    assert as_of_column("venue_id") not in stamped.columns

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.models.audit import append_prediction_audit, validate_prediction_audit


def _slate(first_pitch: str = "2026-05-24T23:05:00Z", *, pitchers: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 123,
                "game_date": first_pitch,
                "official_date": "2026-05-24",
                "home_team_id": 20,
                "away_team_id": 10,
                "home_team_name": "Home Club",
                "away_team_name": "Away Club",
                "home_sp_name": "Home Starter" if pitchers else None,
                "away_sp_name": "Away Starter" if pitchers else None,
                "scheduled_start_utc": first_pitch,
            }
        ]
    )


def _predictions(p_home: float = 0.61) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 123,
                "game_date": "2026-05-24T23:05:00Z",
                "p_home_win": p_home,
                "expected_total_runs": 8.4,
            }
        ]
    )


def _features(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 123,
                "game_date": "2026-05-24T23:05:00Z",
                "home_wins_l10": value,
                "away_wins_l10": 4,
            }
        ]
    )


def test_prediction_audit_does_not_append_after_first_pitch(tmp_path) -> None:
    audit_path = tmp_path / "prediction_audit.csv"
    report_path = tmp_path / "prediction_audit_report.md"
    slate = _slate()

    append_prediction_audit(
        slate=slate,
        predictions=_predictions(0.61),
        features=_features(1.0),
        model_dir=tmp_path,
        audit_path=audit_path,
        report_path=report_path,
        now=datetime(2026, 5, 24, 22, 30, tzinfo=UTC),
        model_version="model-a",
    )
    append_prediction_audit(
        slate=slate,
        predictions=_predictions(0.72),
        features=_features(2.0),
        model_dir=tmp_path,
        audit_path=audit_path,
        report_path=report_path,
        now=datetime(2026, 5, 24, 23, 30, tzinfo=UTC),
        model_version="model-b",
    )

    audit = pd.read_csv(audit_path)
    assert len(audit) == 1
    assert audit.loc[0, "model_version"] == "model-a"
    assert audit.loc[0, "home_win_probability"] == 0.61


def test_prediction_audit_prevents_duplicate_same_model_and_snapshot(tmp_path) -> None:
    audit_path = tmp_path / "prediction_audit.csv"
    report_path = tmp_path / "prediction_audit_report.md"

    for _ in range(2):
        append_prediction_audit(
            slate=_slate(),
            predictions=_predictions(),
            features=_features(),
            model_dir=tmp_path,
            audit_path=audit_path,
            report_path=report_path,
            now=datetime(2026, 5, 24, 21, 0, tzinfo=UTC),
            model_version="model-a",
        )

    audit = pd.read_csv(audit_path)
    assert len(audit) == 1
    assert audit.loc[0, "exclusion_reason"] != audit.loc[0, "exclusion_reason"]


def test_prediction_audit_allows_missing_probable_pitchers(tmp_path) -> None:
    audit_path = tmp_path / "prediction_audit.csv"

    append_prediction_audit(
        slate=_slate(pitchers=False),
        predictions=_predictions(),
        features=_features(),
        model_dir=tmp_path,
        audit_path=audit_path,
        report_path=tmp_path / "prediction_audit_report.md",
        now=datetime(2026, 5, 24, 21, 0, tzinfo=UTC),
        model_version="model-a",
    )

    audit = pd.read_csv(audit_path)
    assert len(audit) == 1
    assert audit.loc[0, "probable_home_pitcher"] == "TBD"
    assert audit.loc[0, "probable_away_pitcher"] == "TBD"
    assert audit.loc[0, "predicted_winner"] == "Home Club"


def test_prediction_audit_records_exclusion_reason(tmp_path) -> None:
    audit_path = tmp_path / "prediction_audit.csv"
    report_path = tmp_path / "prediction_audit_report.md"

    append_prediction_audit(
        slate=_slate(),
        predictions=pd.DataFrame(),
        features=pd.DataFrame(),
        model_dir=tmp_path,
        audit_path=audit_path,
        report_path=report_path,
        now=datetime(2026, 5, 24, 21, 0, tzinfo=UTC),
        model_version="model-a",
        exclusions={123: "postponed"},
    )

    audit = pd.read_csv(audit_path)
    issues = validate_prediction_audit(audit_path=audit_path, report_path=report_path)

    assert len(audit) == 1
    assert audit.loc[0, "exclusion_reason"] == "postponed"
    assert issues["missing_probabilities"] == []
    assert report_path.exists()

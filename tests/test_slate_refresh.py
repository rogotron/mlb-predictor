"""Tests for slate-input cache refresh and staleness guards."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import src.data.statcast as statcast_module
import src.data.update as update_module
from src.data.update import assert_processed_games_fresh, refresh_schedule_cache
from src.features.build import build_bullpen_prediction_features


def _write_schedule(raw_dir, year, rows):
    path = raw_dir / "schedule" / f"schedule_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_processed_games(processed_dir, year, rows):
    path = processed_dir / "games" / f"games_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _schedule_row(game_pk, official_date, status="Final", home_score=4, away_score=2):
    return {
        "game_pk": game_pk,
        "game_date": f"{official_date}T23:05:00Z",
        "official_date": official_date,
        "game_type": "R",
        "status": status,
        "away_team_id": 1,
        "away_team_name": "Away",
        "home_team_id": 2,
        "home_team_name": "Home",
        "away_score": away_score,
        "home_score": home_score,
        "venue_id": 10,
        "venue_name": "Park",
        "scheduled_innings": 9,
    }


class TestAssertProcessedGamesFresh:
    def test_raises_when_completed_games_missing(self, tmp_path):
        raw = tmp_path / "raw"
        processed = tmp_path / "processed"
        _write_schedule(raw, 2026, [
            _schedule_row(100, "2026-07-01"),
            _schedule_row(101, "2026-07-02"),
        ])
        _write_processed_games(processed, 2026, [
            {"game_pk": 100, "official_date": "2026-07-01"},
        ])

        with pytest.raises(RuntimeError, match="stale"):
            assert_processed_games_fresh(
                date(2026, 7, 3), raw_dir=raw, processed_dir=processed
            )

    def test_passes_when_all_finals_processed(self, tmp_path):
        raw = tmp_path / "raw"
        processed = tmp_path / "processed"
        _write_schedule(raw, 2026, [
            _schedule_row(100, "2026-07-01"),
            _schedule_row(101, "2026-07-02"),
            _schedule_row(102, "2026-07-03", status="Preview", home_score=None, away_score=None),
        ])
        _write_processed_games(processed, 2026, [
            {"game_pk": 100, "official_date": "2026-07-01"},
            {"game_pk": 101, "official_date": "2026-07-02"},
        ])

        assert_processed_games_fresh(
            date(2026, 7, 3), raw_dir=raw, processed_dir=processed
        )

    def test_ignores_finals_without_scores(self, tmp_path):
        # MLB marks some postponed games Final with blank scores; ingest drops
        # them, so the freshness check must not demand them.
        raw = tmp_path / "raw"
        processed = tmp_path / "processed"
        _write_schedule(raw, 2026, [
            _schedule_row(100, "2026-07-01"),
            _schedule_row(101, "2026-07-02", home_score=None, away_score=None),
        ])
        _write_processed_games(processed, 2026, [
            {"game_pk": 100, "official_date": "2026-07-01"},
        ])

        assert_processed_games_fresh(
            date(2026, 7, 3), raw_dir=raw, processed_dir=processed
        )

    def test_skips_when_caches_absent(self, tmp_path):
        assert_processed_games_fresh(
            date(2026, 7, 3),
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )


class TestRefreshScheduleCache:
    def test_merges_fresh_window_over_cached_rows(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        _write_schedule(raw, 2026, [
            _schedule_row(100, "2026-06-01"),
            _schedule_row(101, "2026-07-01", status="Preview", home_score=None, away_score=None),
        ])

        fresh = pd.DataFrame([
            _schedule_row(101, "2026-07-01"),
            _schedule_row(102, "2026-07-02"),
        ])
        captured = {}

        def fake_fetch(start_date, end_date):
            captured["window"] = (start_date, end_date)
            return fresh

        monkeypatch.setattr(update_module, "_fetch_raw_schedule_range", fake_fetch)

        merged = refresh_schedule_cache(date(2026, 7, 3), raw_dir=raw, overlap_days=10)

        # Old row outside the window is kept; the Preview row is replaced.
        assert set(merged["game_pk"]) == {100, 101, 102}
        updated = merged.loc[merged["game_pk"] == 101].iloc[0]
        assert updated["status"] == "Final"
        # Window reaches back overlap_days before the cache max, capped at target.
        assert captured["window"] == (date(2026, 6, 21), date(2026, 7, 3))

        on_disk = pd.read_parquet(raw / "schedule" / "schedule_2026.parquet")
        assert set(on_disk["game_pk"]) == {100, 101, 102}

    def test_keeps_cached_rows_after_target_when_backfilling(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        _write_schedule(raw, 2026, [
            _schedule_row(100, "2026-06-01"),
            _schedule_row(103, "2026-07-04", status="Preview", home_score=None, away_score=None),
        ])
        monkeypatch.setattr(
            update_module,
            "_fetch_raw_schedule_range",
            lambda start, end: pd.DataFrame([_schedule_row(101, "2026-06-25")]),
        )

        merged = refresh_schedule_cache(date(2026, 6, 26), raw_dir=raw, overlap_days=10)

        assert set(merged["game_pk"]) == {100, 101, 103}


class TestBullpenWorkloadStalenessGuard:
    _BULLPEN_STAT_COLS = ["bullpen_xwoba_against", "bullpen_whiff_rate", "bullpen_barrel_rate"]

    def _pen_frame(self, rows):
        records = []
        for game_pk, game_date, pitches in rows:
            record = {
                "game_pk": game_pk,
                "game_date": pd.Timestamp(game_date),
                "side": "home",
                "bullpen_pa": 10,
                "bullpen_pitches": pitches,
            }
            for col in self._BULLPEN_STAT_COLS:
                record[col] = 0.3
            records.append(record)
        return pd.DataFrame(records)

    def _slate(self):
        return pd.DataFrame([
            {"game_pk": 900, "home_team_id": 1, "away_team_id": 2},
        ])

    def _setup(self, tmp_path, monkeypatch, pen):
        processed = tmp_path / "processed"
        _write_processed_games(processed, 2026, [
            {"game_pk": 100, "home_team_id": 1, "away_team_id": 2, "official_date": "2026-06-30"},
        ])
        monkeypatch.setattr(
            statcast_module, "load_statcast", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(
            statcast_module, "aggregate_team_game_bullpen", lambda pitches: pen
        )
        return processed

    def test_unmapped_recent_appearance_yields_nan_not_zero(self, tmp_path, monkeypatch):
        # game_pk 101 happened 2026-07-02 but is missing from processed games:
        # every workload window covering that date must be NaN, not 0.
        pen = self._pen_frame([(100, "2026-06-30", 40), (101, "2026-07-02", 55)])
        processed = self._setup(tmp_path, monkeypatch, pen)

        result = build_bullpen_prediction_features(
            self._slate(), tmp_path / "raw", processed, target_date=date(2026, 7, 3)
        )

        row = result.iloc[0]
        for days in (1, 2, 3):
            assert pd.isna(row[f"home_bullpen_pitches_l{days}d"])
            assert pd.isna(row[f"home_bullpen_games_l{days}d"])
        assert pd.isna(row["home_bullpen_back_to_back_l2d"])
        assert pd.isna(row["home_bullpen_heavy_work_l2d"])
        # Rolling quality means still come from the mapped appearance.
        assert row["home_bullpen_xwoba_against_l14"] == pytest.approx(0.3)

    def test_fully_mapped_appearances_yield_real_zeros(self, tmp_path, monkeypatch):
        # No appearances in the L1/L2 windows and nothing unmapped: 0 is honest.
        pen = self._pen_frame([(100, "2026-06-30", 40)])
        processed = self._setup(tmp_path, monkeypatch, pen)

        result = build_bullpen_prediction_features(
            self._slate(), tmp_path / "raw", processed, target_date=date(2026, 7, 3)
        )

        row = result.iloc[0]
        assert row["home_bullpen_pitches_l1d"] == 0
        assert row["home_bullpen_pitches_l2d"] == 0
        assert row["home_bullpen_pitches_l3d"] == 40
        assert row["home_bullpen_back_to_back_l2d"] == 0
        assert row["home_bullpen_heavy_work_l2d"] == 0

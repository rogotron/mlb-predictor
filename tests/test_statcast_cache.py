"""Tests for Statcast month-cache staleness detection.

A month parquet last fetched mid-month is truncated; the calendar-based
completeness check alone would skip it forever. fetch_statcast_month must
compare the cache against the schedule and re-fetch truncated months.

Uses 2024 so the calendar check always sees a past month regardless of when
the tests run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import src.data.statcast as statcast_module
from src.data.statcast import _cache_covers_month, fetch_statcast_month


def _write_schedule(raw_dir, year, official_dates):
    path = raw_dir / "schedule" / f"schedule_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "game_pk": range(100, 100 + len(official_dates)),
        "game_date": [f"{d}T23:05:00Z" for d in official_dates],
        "official_date": official_dates,
        "status": "Final",
    }).to_parquet(path, index=False)


def _pitch_frame(dates):
    return pd.DataFrame({
        "game_pk": range(1, len(dates) + 1),
        "game_date": pd.to_datetime(dates),
        "pitcher": 500,
        "batter": 600,
        "woba_denom": 1.0,
    })


def _write_pitches(raw_dir, year, month, dates):
    path = raw_dir / "statcast" / f"pitches_{year}_{month:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    _pitch_frame(dates).to_parquet(path, index=False)
    return path


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Isolated raw dir with a fresh in-memory month cache per test."""
    monkeypatch.setattr(statcast_module, "_MONTH_CACHE", {})
    return tmp_path / "raw"


@pytest.fixture
def fake_pybaseball(monkeypatch):
    """Stub pybaseball; counts statcast() calls and returns a full April 2024."""
    full_month = _pitch_frame(["2024-04-01", "2024-04-15", "2024-04-30"])
    calls = SimpleNamespace(count=0)

    def fake_statcast(start_dt, end_dt, verbose=False):
        calls.count += 1
        return full_month.copy()

    monkeypatch.setattr(
        statcast_module,
        "pybaseball",
        SimpleNamespace(
            statcast=fake_statcast,
            cache=SimpleNamespace(enable=lambda: None),
        ),
    )
    return calls


class TestCacheCoversMonth:
    def test_truncated_cache_not_covered(self, raw_dir):
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-29", "2024-04-30"])
        cached = _pitch_frame(["2024-04-01", "2024-04-29"])
        assert not _cache_covers_month(cached, 2024, 4, raw_dir)

    def test_full_cache_covered(self, raw_dir):
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        cached = _pitch_frame(["2024-04-01", "2024-04-30"])
        assert _cache_covers_month(cached, 2024, 4, raw_dir)

    def test_offseason_month_covered_by_empty_cache(self, raw_dir):
        # Schedule exists for the year but has no January games.
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        cached = pd.DataFrame(columns=["game_pk", "game_date"])
        assert _cache_covers_month(cached, 2024, 1, raw_dir)

    def test_covers_to_last_scheduled_game_not_calendar_end(self, raw_dir):
        # Season ends Oct 27; a cache reaching that date is complete even
        # though the calendar month runs to Oct 31.
        _write_schedule(raw_dir, 2024, ["2024-10-01", "2024-10-27"])
        cached = _pitch_frame(["2024-10-27"])
        assert _cache_covers_month(cached, 2024, 10, raw_dir)

    def test_no_schedule_falls_back_to_calendar_month_end(self, raw_dir):
        truncated = _pitch_frame(["2024-04-29"])
        full = _pitch_frame(["2024-04-30"])
        assert not _cache_covers_month(truncated, 2024, 4, raw_dir)
        assert _cache_covers_month(full, 2024, 4, raw_dir)

    def test_empty_cache_for_scheduled_month_not_covered(self, raw_dir):
        _write_schedule(raw_dir, 2024, ["2024-04-30"])
        cached = pd.DataFrame(columns=["game_pk", "game_date"])
        assert not _cache_covers_month(cached, 2024, 4, raw_dir)


class TestFetchStatcastMonthStaleness:
    def test_truncated_past_month_is_refetched(self, raw_dir, fake_pybaseball):
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        path = _write_pitches(raw_dir, 2024, 4, ["2024-04-01", "2024-04-29"])

        df = fetch_statcast_month(2024, 4, raw_dir=raw_dir)

        assert fake_pybaseball.count == 1
        assert df["game_date"].max() == pd.Timestamp("2024-04-30")
        # The on-disk cache is repaired too.
        on_disk = pd.read_parquet(path)
        assert pd.to_datetime(on_disk["game_date"]).max() == pd.Timestamp("2024-04-30")

    def test_complete_past_month_is_not_refetched(self, raw_dir, fake_pybaseball):
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        _write_pitches(raw_dir, 2024, 4, ["2024-04-01", "2024-04-30"])

        df = fetch_statcast_month(2024, 4, raw_dir=raw_dir)

        assert fake_pybaseball.count == 0
        assert len(df) == 2

    def test_offseason_empty_cache_is_not_refetched(self, raw_dir, fake_pybaseball):
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        path = raw_dir / "statcast" / "pitches_2024_01.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=statcast_module._KEEP_COLS).to_parquet(path, index=False)

        df = fetch_statcast_month(2024, 1, raw_dir=raw_dir)

        assert fake_pybaseball.count == 0
        assert df.empty

    def test_month_cache_short_circuits_coverage_check(self, raw_dir, fake_pybaseball):
        # Second call within a process must hit _MONTH_CACHE and not re-read
        # the parquet or re-fetch, even after the on-disk cache is deleted.
        _write_schedule(raw_dir, 2024, ["2024-04-01", "2024-04-30"])
        path = _write_pitches(raw_dir, 2024, 4, ["2024-04-01", "2024-04-30"])

        first = fetch_statcast_month(2024, 4, raw_dir=raw_dir)
        path.unlink()
        second = fetch_statcast_month(2024, 4, raw_dir=raw_dir)

        assert fake_pybaseball.count == 0
        pd.testing.assert_frame_equal(first, second)

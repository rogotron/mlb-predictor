"""Tests for the data-provenance manifest builder."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.provenance import build_slate_provenance


def _write_schedule(raw_dir, year, rows):
    path = raw_dir / "schedule" / f"schedule_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_games(processed_dir, year, official_dates):
    path = processed_dir / "games" / f"games_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "game_pk": range(1, len(official_dates) + 1),
        "official_date": pd.to_datetime(official_dates),
    }).to_parquet(path, index=False)


def test_weather_and_umpire_are_not_collected(tmp_path):
    prov = build_slate_provenance(
        date(2026, 7, 4),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "proc",
    )
    by_key = {s["key"]: s for s in prov["sources"]}
    assert by_key["weather"]["status"] == "not_collected"
    assert by_key["umpire"]["status"] == "not_collected"
    # Not-collected sources never count as stale or missing.
    assert by_key["weather"]["stale"] is False


def test_missing_cache_flags_missing_and_stale(tmp_path):
    prov = build_slate_provenance(
        date(2026, 7, 4),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "proc",
    )
    schedule = next(s for s in prov["sources"] if s["key"] == "schedule")
    assert schedule["status"] == "missing"
    assert schedule["stale"] is True
    assert prov["anyMissing"] is True


def test_completed_schedule_caps_future_dated_rows(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    target = date(2026, 7, 4)
    # A fixture-style schedule with Final games pre-populated into the future.
    _write_schedule(raw, 2026, {
        "game_pk": [1, 2, 3],
        "official_date": ["2026-07-03", "2026-07-04", "2026-09-22"],
        "game_date": ["2026-07-03T23:00:00Z", "2026-07-04T23:00:00Z", "2026-09-22T23:00:00Z"],
        "status": ["Final", "Final", "Final"],
    })
    _write_games(proc, 2026, ["2026-07-03", "2026-07-04"])

    prov = build_slate_provenance(target, raw_dir=raw, processed_dir=proc)
    schedule = next(s for s in prov["sources"] if s["key"] == "schedule")
    # Future-dated Final rows are excluded: maxDate is the target, not September.
    assert schedule["maxDate"] == "2026-07-04"
    assert schedule["rows"] == 2
    assert schedule["stale"] is False
    assert prov["dateRange"]["end"] == "2026-07-04"

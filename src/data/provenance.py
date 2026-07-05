"""Data-provenance manifest for the dashboard and the one-command refresh.

Inspects the on-disk caches the refresh pipeline writes and reports, per
source: row count, covered date range, freshness, and status. This is the
single source of truth the dashboard header reads, so displayed data can no
longer silently drift from what is actually cached.

Weather and umpire are reported as ``not_collected`` on purpose — the model
does not use them. See CLAUDE.md and diagnostics/feature_inventory.csv.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.update import SCHEDULE_TIMEZONE
from src.utils.paths import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# A source more than this many days behind the target date is flagged stale.
# One day of lag is normal (Statcast in particular publishes with a delay).
STALE_AFTER_DAYS = 2


def _date_bounds(frame: pd.DataFrame, *columns: str) -> tuple[date | None, date | None]:
    """Return (min, max) date across the first present column, or (None, None)."""
    for col in columns:
        if col in frame.columns:
            parsed = pd.to_datetime(frame[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed.min().date(), parsed.max().date()
    return None, None


def _source_from_parquet(
    key: str,
    label: str,
    paths: list[Path],
    target_date: date,
    *,
    date_columns: tuple[str, ...],
    completed_only: bool = False,
) -> dict[str, Any]:
    """Build a source entry by reading row counts and date bounds from parquet.

    When completed_only is set and a ``status`` column is present, date bounds
    and the row count reflect only completed (Final) games — so a schedule
    cache full of future scheduled games reports its last *played* date, not a
    future one.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return {
            "key": key,
            "label": label,
            "status": "missing",
            "rows": 0,
            "minDate": None,
            "maxDate": None,
            "daysBehind": None,
            "stale": True,
        }

    rows = 0
    min_date: date | None = None
    max_date: date | None = None
    try:
        for path in existing:
            frame = pd.read_parquet(path)
            if completed_only:
                if "status" in frame.columns:
                    frame = frame[frame["status"] == "Final"]
                # Only count results that should exist by the target date, so a
                # season fixture pre-marked Final doesn't report a future date.
                for col in date_columns:
                    if col in frame.columns:
                        as_of = pd.to_datetime(frame[col], errors="coerce").dt.date
                        frame = frame[as_of <= target_date]
                        break
            rows += len(frame)
            lo, hi = _date_bounds(frame, *date_columns)
            if lo is not None and (min_date is None or lo < min_date):
                min_date = lo
            if hi is not None and (max_date is None or hi > max_date):
                max_date = hi
    except Exception as exc:  # pragma: no cover - defensive; provenance must not break builds
        logger.warning("provenance: failed reading %s cache: %s", key, exc)
        return {
            "key": key,
            "label": label,
            "status": "error",
            "rows": rows,
            "minDate": min_date.isoformat() if min_date else None,
            "maxDate": max_date.isoformat() if max_date else None,
            "daysBehind": None,
            "stale": True,
        }

    days_behind = (target_date - max_date).days if max_date else None
    stale = days_behind is None or days_behind >= STALE_AFTER_DAYS
    return {
        "key": key,
        "label": label,
        "status": "ok" if rows > 0 else "missing",
        "rows": int(rows),
        "minDate": min_date.isoformat() if min_date else None,
        "maxDate": max_date.isoformat() if max_date else None,
        "daysBehind": days_behind,
        "stale": bool(stale),
    }


def _statcast_month_paths(raw_dir: Path, year: int) -> list[Path]:
    return sorted((raw_dir / "statcast").glob(f"pitches_{year}_*.parquet"))


def build_slate_provenance(
    target_date: date,
    *,
    game_count: int | None = None,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable provenance manifest for a slate date.

    Reads only what is already cached on disk — it never fetches — so it is
    safe to call at the end of any build without network access.
    """
    year = target_date.year
    now = now or datetime.now(SCHEDULE_TIMEZONE)

    sources = [
        _source_from_parquet(
            "schedule",
            "MLB schedule & scores (final)",
            [raw_dir / "schedule" / f"schedule_{year}.parquet"],
            target_date,
            date_columns=("official_date", "game_date"),
            completed_only=True,
        ),
        _source_from_parquet(
            "processed_games",
            "Processed game logs",
            [processed_dir / "games" / f"games_{year}.parquet"],
            target_date,
            date_columns=("official_date", "game_date"),
        ),
        _source_from_parquet(
            "pitching_gamelogs",
            "Starting-pitcher gamelogs",
            [raw_dir / "pitching_gamelogs" / f"{year}.parquet"],
            target_date,
            date_columns=("game_date",),
        ),
        _source_from_parquet(
            "statcast",
            "Statcast pitches",
            _statcast_month_paths(raw_dir, year),
            target_date,
            date_columns=("game_date",),
        ),
        {
            "key": "weather",
            "label": "Weather",
            "status": "not_collected",
            "rows": 0,
            "minDate": None,
            "maxDate": None,
            "daysBehind": None,
            "stale": False,
        },
        {
            "key": "umpire",
            "label": "Umpire tendencies",
            "status": "not_collected",
            "rows": 0,
            "minDate": None,
            "maxDate": None,
            "daysBehind": None,
            "stale": False,
        },
    ]

    tracked = [s for s in sources if s["status"] not in {"not_collected"}]
    min_dates = [s["minDate"] for s in tracked if s["minDate"]]
    max_dates = [s["maxDate"] for s in tracked if s["maxDate"]]

    return {
        "dataAsOf": now.isoformat(),
        "targetDate": target_date.isoformat(),
        "gameCount": int(game_count) if game_count is not None else None,
        "dateRange": {
            "start": min(min_dates) if min_dates else None,
            "end": max(max_dates) if max_dates else None,
        },
        "anyStale": any(s["stale"] for s in tracked),
        "anyMissing": any(s["status"] in {"missing", "error"} for s in tracked),
        "sources": sources,
    }

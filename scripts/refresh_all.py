"""One-command daily refresh: pull latest data, rebuild features, re-run
predictions, and write dated artifacts with a provenance manifest.

Runs, in order:
  1. Schedule + processed games + SP gamelog caches (refresh_slate_inputs)
  2. Statcast pitches for the season (self-heals truncated months)
  3. Static slate JSON for the date (predictions + provenance + model block)

Then writes data/processed/refresh_manifest.json and public/refresh_manifest.json:
per-source row counts, covered date range, and an ok/partial/failed status per
step. Exits non-zero if any step failed or a tracked source is missing, so a
partial or failed pull is visible rather than silent.

Weather and umpire are intentionally NOT collected (the model does not use
them); they appear in the manifest as ``not_collected``.

Example:
    python scripts/refresh_all.py                 # today
    python scripts/refresh_all.py --date 2026-07-04
    python scripts/refresh_all.py --skip-slate    # data refresh only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.provenance import build_slate_provenance
from src.data.statcast import fetch_statcast_month
from src.data.update import (
    SCHEDULE_TIMEZONE,
    refresh_slate_inputs,
    today_in_schedule_timezone,
)
from src.utils.logging import configure_logging
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)

SEASON_MONTHS = range(3, 11)  # March–October


def _refresh_inputs_step(target: date) -> dict:
    try:
        refresh_slate_inputs(target)
        return {"name": "schedule+ingest+gamelogs", "status": "ok"}
    except Exception as exc:
        logger.exception("input refresh failed")
        return {"name": "schedule+ingest+gamelogs", "status": "failed", "error": str(exc)}


def _refresh_statcast_step(target: date) -> dict:
    year = target.year
    pitches = 0
    months = 0
    errors = 0
    for month in SEASON_MONTHS:
        try:
            df = fetch_statcast_month(year, month, raw_dir=RAW_DIR, force=False)
            if not df.empty:
                pitches += len(df)
                months += 1
        except Exception as exc:
            errors += 1
            logger.warning("statcast %d-%02d failed: %s", year, month, exc)
    status = "ok" if errors == 0 else ("partial" if months else "failed")
    return {
        "name": "statcast",
        "status": status,
        "monthsFetched": months,
        "pitches": pitches,
        "errors": errors,
    }


def _build_slate_step(target: date) -> dict:
    # Imported here so a data-only refresh (--skip-slate) needn't load the
    # model/assembler stack.
    from build_static_slate import build_static_slate

    try:
        result = build_static_slate(target, skip_refresh=True)
        if not result:
            return {"name": "slate", "status": "skipped", "detail": "no games for date"}
        return {"name": "slate", "status": "ok", "games": result["games"]}
    except Exception as exc:
        logger.exception("slate build failed")
        return {"name": "slate", "status": "failed", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument("--skip-slate", action="store_true", help="Refresh data only; skip predictions")
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()
    started = datetime.now(SCHEDULE_TIMEZONE)
    logger.info("=== refresh_all for %s ===", target)

    steps: list[dict] = []
    steps.append(_refresh_inputs_step(target))
    steps.append(_refresh_statcast_step(target))
    if args.skip_slate:
        steps.append({"name": "slate", "status": "skipped", "detail": "--skip-slate"})
    else:
        steps.append(_build_slate_step(target))

    provenance = build_slate_provenance(target, now=datetime.now(SCHEDULE_TIMEZONE))
    finished = datetime.now(SCHEDULE_TIMEZONE)

    hard_failed = any(s["status"] == "failed" for s in steps)
    ok = not hard_failed and not provenance["anyMissing"]

    manifest = {
        "targetDate": target.isoformat(),
        "startedAt": started.isoformat(),
        "finishedAt": finished.isoformat(),
        "ok": ok,
        "steps": steps,
        "dateRange": provenance["dateRange"],
        "provenance": provenance,
    }

    for out in [PROCESSED_DIR / "refresh_manifest.json", Path("public") / "refresh_manifest.json"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)

    # Human-readable summary
    logger.info("--- refresh summary (%s) ---", "OK" if ok else "ATTENTION NEEDED")
    for step in steps:
        logger.info("  %-24s %s", step["name"], step["status"])
    for src in provenance["sources"]:
        logger.info(
            "  source %-22s %-13s rows=%s thru=%s%s",
            src["label"], src["status"], src["rows"], src["maxDate"],
            " STALE" if src["stale"] else "",
        )
    logger.info("date range covered: %s → %s",
                provenance["dateRange"]["start"], provenance["dateRange"]["end"])

    if not ok:
        logger.error("refresh incomplete — see manifest at %s", PROCESSED_DIR / "refresh_manifest.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

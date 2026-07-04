"""Build a static full-dashboard JSON for a date and write it to public/slates/.

Calls the same assembler as the live API so pitcher stats, standings, and
model predictions are all included. Run this once per day after probables
are confirmed; the frontend loads it instantly from the static file.

By default this first refreshes the schedule, processed-games, and SP
gamelog caches (stale processed games silently zero bullpen-workload
features). Pass --skip-refresh to build from the caches as-is.

Example:
    python scripts/build_static_slate.py                  # today
    python scripts/build_static_slate.py --date 2026-05-06
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so both src/ and backend/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.assemble import build_slate_payloads
from src.data.update import refresh_slate_inputs, today_in_schedule_timezone
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf values to null before strict JSON export."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="ISO date (default: today)")
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip refreshing schedule/processed/gamelog caches before building",
    )
    args = parser.parse_args()

    configure_logging()

    target = date.fromisoformat(args.date) if args.date else today_in_schedule_timezone()

    if args.skip_refresh:
        logger.info("skipping cache refresh (--skip-refresh)")
    else:
        logger.info("refreshing slate input caches for %s", target)
        refresh_slate_inputs(target)

    logger.info("building static slate for %s", target)

    games = build_slate_payloads(target)
    if not games:
        logger.warning("no games found for %s", target)
        return

    payload = _json_safe({"date": str(target), "count": len(games), "games": games})
    json_str = json.dumps(payload, indent=2, allow_nan=False)

    for out in [
        Path("public") / "slates" / f"{target}.json",
        Path("dist") / "slates" / f"{target}.json",
    ]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str, encoding="utf-8")
        logger.info("wrote %s (%d games)", out, len(games))

    print(f"wrote {len(games)} games to public/slates/{target}.json")


if __name__ == "__main__":
    main()

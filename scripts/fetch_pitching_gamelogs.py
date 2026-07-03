"""Fetch and cache per-start pitching lines from the MLB Stats API.

Uses the boxscore endpoint (same host as schedules) — reliable and fast.
Incremental: already-cached game_pks are skipped, so re-running is safe.

Example:
    python scripts/fetch_pitching_gamelogs.py --start 2018 --end 2025
    python scripts/fetch_pitching_gamelogs.py --start 2026 --end 2026 --force
"""

from __future__ import annotations

import argparse
import logging

from src.data.pitching_gamelogs import fetch_season_pitching_logs
from src.utils.logging import configure_logging
from src.utils.paths import RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, help="First season (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Last season (inclusive)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    for year in range(args.start, args.end + 1):
        logger.info("=== %d ===", year)
        df = fetch_season_pitching_logs(year, raw_dir=RAW_DIR, force=args.force)
        logger.info("%d: done — %d SP lines in cache", year, len(df))

    logger.info("finished. cache at %s", RAW_DIR / "pitching_gamelogs")


if __name__ == "__main__":
    main()

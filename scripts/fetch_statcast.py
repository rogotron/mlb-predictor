"""Pre-fetch and cache Statcast pitch data by calendar month.

Iterates over the season months (March–October) for each requested year and
stores one parquet per month in data/raw/statcast/. Already-complete months
are skipped unless --force is passed.

Example:
    python scripts/fetch_statcast.py --start 2018 --end 2024
    python scripts/fetch_statcast.py --start 2024 --end 2024 --force
"""

from __future__ import annotations

import argparse
import logging

from src.data.statcast import fetch_statcast_month
from src.utils.logging import configure_logging
from src.utils.paths import RAW_DIR, ensure_dirs

logger = logging.getLogger(__name__)

SEASON_MONTHS = range(3, 11)  # March–October


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True, help="First season (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Last season (inclusive)")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    total = (args.end - args.start + 1) * len(SEASON_MONTHS)
    done = 0
    errors = 0

    for year in range(args.start, args.end + 1):
        for month in SEASON_MONTHS:
            done += 1
            logger.info("[%d/%d] year=%d month=%02d", done, total, year, month)
            try:
                df = fetch_statcast_month(year, month, raw_dir=RAW_DIR, force=args.force)
                if df.empty:
                    logger.info("  no data (off-season or future)")
                else:
                    logger.info("  %d pitches", len(df))
            except Exception as exc:
                logger.warning("  FAILED: %s", exc)
                errors += 1

    logger.info("done. %d months fetched, %d errors. cache at %s", done, errors, RAW_DIR / "statcast")


if __name__ == "__main__":
    main()

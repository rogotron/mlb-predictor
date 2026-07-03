"""Single-call logging setup. Import and call configure_logging() at the top of every script."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | None = None) -> None:
    """Set up root logger with a consistent format.

    Reads LOG_LEVEL from env if level not passed. Safe to call multiple times.
    """
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if root.handlers:
        # already configured
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

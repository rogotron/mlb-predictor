"""Resolve project paths from environment, with sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models")).resolve()


def ensure_dirs() -> None:
    """Create all data/model directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

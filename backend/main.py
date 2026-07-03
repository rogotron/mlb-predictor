"""DiamondEdge FastAPI backend.

Run locally:
    uvicorn backend.main:app --reload --port 8000

The Vite dev server proxies /api → http://localhost:8000, so the React
frontend can hit the API without CORS issues during development.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.model import router as model_router
from backend.routers.predictions import router as predictions_router
from src.models.predict import load_latest_model
from src.utils.logging import configure_logging
from src.utils.paths import MODEL_DIR, ensure_dirs

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm: verify models are loadable before accepting requests."""
    ensure_dirs()
    try:
        load_latest_model(MODEL_DIR, "home_win")
        load_latest_model(MODEL_DIR, "total_runs")
        logger.info("models loaded OK")
    except FileNotFoundError as exc:
        logger.warning("model not found at startup: %s", exc)
    yield


app = FastAPI(
    title="DiamondEdge Baseball Intelligence",
    version="2.4.1",
    description="MLB game prediction API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predictions_router, prefix="/api")
app.include_router(model_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.4.1"}

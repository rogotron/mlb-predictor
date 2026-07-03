"""Prediction endpoints."""
from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from backend.services.assemble import (
    build_model_preview_payloads,
    build_quick_slate_payloads,
    build_slate_payloads,
    invalidate_cache,
)
from src.data.update import fetch_slate_range, today_in_schedule_timezone

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/today")
def get_today(d: date = Query(default=None, description="ISO date (default: today)")):
    """All games for a single date with full dashboard payloads."""
    target = d or today_in_schedule_timezone()
    payloads = build_slate_payloads(target)
    return {"date": str(target), "count": len(payloads), "games": payloads}


@router.get("/quick")
def get_quick(d: date = Query(default=None, description="ISO date (default: today)")):
    """Fast slate payload for calendar navigation."""
    target = d or today_in_schedule_timezone()
    payloads = build_quick_slate_payloads(target)
    return {"date": str(target), "count": len(payloads), "games": payloads}


@router.get("/model-preview")
def get_model_preview(d: date = Query(default=None, description="ISO date (default: today)")):
    """Model predictions without slow rich dashboard detail."""
    target = d or today_in_schedule_timezone()
    payloads = build_model_preview_payloads(target)
    return {"date": str(target), "count": len(payloads), "games": payloads}


@router.get("/range")
def get_range(
    start: date = Query(..., description="Start date (inclusive)"),
    end:   date = Query(..., description="End date (inclusive)"),
):
    """Games across a date range — probables fill in as they're announced."""
    if end < start:
        raise HTTPException(400, "end must be >= start")
    if (end - start).days > 7:
        raise HTTPException(400, "Range cannot exceed 7 days")

    slate = fetch_slate_range(start, end)
    if slate.empty:
        return {"start": str(start), "end": str(end), "count": 0, "games": []}

    # Process each day separately so per-day rolling features are correct
    all_payloads: list[dict] = []
    for day_offset in range((end - start).days + 1):
        day = start + timedelta(days=day_offset)
        day_slate = slate[slate["official_date"] == day]
        if day_slate.empty:
            continue
        payloads = build_slate_payloads(day, slate=day_slate.reset_index(drop=True))
        all_payloads.extend(payloads)

    return {"start": str(start), "end": str(end), "count": len(all_payloads), "games": all_payloads}


@router.post("/cache/invalidate")
def clear_cache(d: date = Query(default=None, description="Specific date to evict (default: all)")):
    """Manually evict prediction cache so next request re-runs the pipeline."""
    invalidate_cache(d)
    return {"cleared": str(d) if d else "all"}

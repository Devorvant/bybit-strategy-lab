from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .chart_service import (
    build_trade_chart_payload,
    build_trade_events_payload,
    build_trade_status_payload,
)

router = APIRouter(prefix="/trade/chart", tags=["trade-chart"])
templates = Jinja2Templates(directory="templates")


@router.get("", response_class=HTMLResponse)
def trade_chart_page(request: Request):
    return templates.TemplateResponse("trade_chart.html", {"request": request})


@router.get("/data")
def trade_chart_data(
    symbol: str = Query("APTUSDT"),
    tf: str = Query("30"),
    strategy: str = Query("my_strategy3.py"),
    opt_id: int | None = Query(None),
    last: int = Query(20, ge=1, le=500),
    limit: int = Query(2000, ge=100, le=50000),
    capital_usd: float = Query(10000.0, ge=0.0, le=1e9),
    signal_mode: str = Query("legacy"),
):
    return build_trade_chart_payload(
        symbol=symbol,
        tf=tf,
        strategy=strategy,
        opt_id=opt_id,
        last=last,
        limit=limit,
        capital_usd=capital_usd,
        signal_mode=signal_mode,
    )


@router.get("/events")
def trade_chart_events(
    symbol: str = Query("APTUSDT"),
    tf: str = Query("30"),
    limit: int = Query(500, ge=10, le=5000),
):
    return build_trade_events_payload(symbol=symbol, tf=tf, limit=limit)


@router.get("/status")
def trade_chart_status(
    symbol: str = Query("APTUSDT"),
    opt_id: int | None = Query(None),
):
    return build_trade_status_payload(symbol=symbol, opt_id=opt_id)

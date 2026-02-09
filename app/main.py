import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from app.config import settings
from app.storage.db import init_db, load_bars, last_signal
from app.data.bybit_ws import ws_collect
from app.data.backfill import backfill_on_startup
from app.reporting.chart import make_chart_html

app = FastAPI()
conn = None

@app.on_event("startup")
async def startup():
    global conn
    conn = init_db()
    # Backfill истории (опционально, по env BACKFILL_ON_START)
    asyncio.create_task(backfill_on_startup(conn))
    # фоновый сбор свечей
    asyncio.create_task(ws_collect(conn))
    # фоновый пересчёт стратегии (опционально, по env ENABLE_STRATEGY)
    if settings.ENABLE_STRATEGY:
        from app.strategy.runner import strategy_loop
        asyncio.create_task(strategy_loop(conn))

@app.get("/health")
def health():
    return {"ok": True, "symbols": settings.SYMBOLS, "tf": settings.TF}

@app.get("/bars")
def bars(symbol: str = Query("APTUSDT"), tf: str = Query(None), limit: int = Query(500, ge=10, le=5000)):
    tf = tf or settings.TF
    rows = load_bars(conn, symbol.upper(), tf, limit=limit)
    return {"symbol": symbol.upper(), "tf": tf, "bars": rows}

@app.get("/signal")
def signal(symbol: str = Query("APTUSDT"), tf: str = Query(None)):
    tf = tf or settings.TF
    s = last_signal(conn, symbol.upper(), tf)
    return {"symbol": symbol.upper(), "tf": tf, "last_signal": s}

@app.get("/chart", response_class=HTMLResponse)
def chart(symbol: str = Query("APTUSDT"), tf: str = Query(None), limit: int = Query(500, ge=10, le=5000)):
    tf = tf or settings.TF
    rows = load_bars(conn, symbol.upper(), tf, limit=limit)
    return make_chart_html(rows)

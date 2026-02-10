import asyncio
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from app.config import settings
from app.storage.db import init_db, load_bars, last_signal
from app.data.bybit_ws import ws_collect
from app.data.backfill import backfill_on_startup
from app.reporting.chart import make_chart_html

app = FastAPI()
conn = None


def _is_postgres() -> bool:
    url = settings.DATABASE_URL or ""
    return url.startswith("postgres")


def _db_distinct_symbols() -> List[str]:
    """Return distinct symbols present in DB (fallback to env list)."""
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
                rows = cur.fetchall()
                syms = [r[0] for r in rows]
        else:
            cur = conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
            syms = [r[0] for r in cur.fetchall()]
        return syms or list(settings.SYMBOLS)
    except Exception:
        return list(settings.SYMBOLS)


def _db_distinct_tfs(symbol: str) -> List[str]:
    """Return distinct timeframes present in DB for symbol (fallback to common presets)."""
    presets = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT tf FROM bars WHERE symbol=%s ORDER BY tf",
                    (symbol,),
                )
                rows = cur.fetchall()
                tfs = [r[0] for r in rows]
        else:
            cur = conn.execute(
                "SELECT DISTINCT tf FROM bars WHERE symbol=? ORDER BY tf",
                (symbol,),
            )
            tfs = [r[0] for r in cur.fetchall()]
        return tfs or presets
    except Exception:
        return presets

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
def bars(symbol: str = Query("APTUSDT"), tf: str = Query(None), limit: int = Query(500, ge=10, le=50000)):
    tf = tf or settings.TF
    rows = load_bars(conn, symbol.upper(), tf, limit=limit)
    return {"symbol": symbol.upper(), "tf": tf, "bars": rows}

@app.get("/signal")
def signal(symbol: str = Query("APTUSDT"), tf: str = Query(None)):
    tf = tf or settings.TF
    s = last_signal(conn, symbol.upper(), tf)
    return {"symbol": symbol.upper(), "tf": tf, "last_signal": s}

@app.get("/chart", response_class=HTMLResponse)
def chart(symbol: str = Query("APTUSDT"), tf: str = Query(None), limit: int = Query(5000, ge=10, le=50000)):
    tf = tf or settings.TF
    symbol = symbol.upper()
    rows = load_bars(conn, symbol, tf, limit=limit)

    # For UI controls
    symbols = _db_distinct_symbols()
    tfs = _db_distinct_tfs(symbol)

    return make_chart_html(rows, symbol=symbol, tf=tf, limit=limit, symbols=symbols, tfs=tfs)

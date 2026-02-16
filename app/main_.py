import asyncio
from typing import List

import json

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from app.config import settings
from app.storage.db import init_db, load_bars, last_signal
from app.data.bybit_ws import ws_collect
from app.data.backfill import backfill_on_startup
from app.reporting.chart import make_chart_html

app = FastAPI()

from app.reporting.tv_debug import router as tv_debug_router
app.include_router(tv_debug_router)

from app.reporting.tv_ingest import router as tv_ingest_router
app.include_router(tv_ingest_router)

from app.reporting.optimize_web import router as optimize_router
app.include_router(optimize_router)

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
        base = [s.strip().upper() for s in settings.SYMBOLS if s.strip()]
        return sorted(set((syms or []) + base))
    except Exception:
        base = [s.strip().upper() for s in settings.SYMBOLS if s.strip()]
        return sorted(set(base))


def _db_distinct_tfs(symbol: str) -> List[str]:
    """Return distinct timeframes present in DB for symbol (fallback to common presets)."""
    presets = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
    base = [str(x) for x in settings.TFS] + [str(settings.TF)]
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
        # Union of DB + configured + presets so UI can switch TF even if DB has only one
        return sorted(set((tfs or []) + base + presets), key=lambda x: (len(x), x))
    except Exception:
        return sorted(set(base + presets), key=lambda x: (len(x), x))

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
def chart(
    symbol: str = Query("APTUSDT"),
    tf: str = Query(None),
    limit: int = Query(5000, ge=10, le=50000),
    strategy: str = Query("my_strategy.py"),
    opt_id: str | None = Query(None),
    opt_last: int = Query(20, ge=1, le=200),
    # Manual overrides for strategy3 (only applied when use_overrides=1)
    use_overrides: int = Query(0, ge=0, le=1),
    p_use_no_trade: bool | None = Query(None),
    p_adx_no_trade_below: float | None = Query(None, ge=0.0, le=100.0),
    p_st_factor: float | None = Query(None, ge=0.0, le=100.0),
    p_rev_cooldown_hrs: int | None = Query(None, ge=0, le=168),
    p_use_flip_limit: bool | None = Query(None),
    p_max_flips_per_day: int | None = Query(None, ge=0, le=100),
    p_atr_mult: float | None = Query(None, ge=0.0, le=100.0),
):
    tf = tf or settings.TF
    symbol = symbol.upper()
    rows = load_bars(conn, symbol, tf, limit=limit)

    # For UI controls
    symbols = _db_distinct_symbols()
    tfs = _db_distinct_tfs(symbol)

    # Optional: load optimized parameters for a strategy (if available)
    opt_results = []
    opt_params = None
    opt_strategy_map = {
        # chart strategy -> optimizer key (stored in opt_results.strategy)
        "my_strategy3.py": "strategy3",
        # Future/optional:
        # "my_strategy2.py": "strategy2",
        # "my_strategy.py": "sma_cross",
        # "my_strategy_tv_like.py": "sma_cross_tv",
        # "my_strategy3_tv_like.py": "strategy3_tv",
    }
    opt_strategy = opt_strategy_map.get(strategy)

    opt_id_int: int | None = None
    if opt_id is not None:
        s = str(opt_id).strip()
        if s.isdigit():
            opt_id_int = int(s)

    def _json_to_dict(x):
        if x is None:
            return None
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return None
        return None

    if opt_strategy:
        try:
            if settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, created_at, best_score, best_metrics FROM opt_results "
                        "WHERE status='done' AND strategy=%s AND symbol=%s AND tf=%s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (opt_strategy, symbol, tf, opt_last),
                    )
                    opt_results = cur.fetchall() or []
                if opt_id_int is not None:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT best_params FROM opt_results "
                            "WHERE id=%s AND status='done' AND strategy=%s AND symbol=%s AND tf=%s",
                            (opt_id_int, opt_strategy, symbol, tf),
                        )
                        row = cur.fetchone()
                    opt_params = _json_to_dict(row[0] if row else None)
            else:
                # SQLite
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, created_at, best_score, best_metrics FROM opt_results "
                    "WHERE status='done' AND strategy=? AND symbol=? AND tf=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (opt_strategy, symbol, tf, opt_last),
                )
                opt_results = cur.fetchall() or []
                if opt_id_int is not None:
                    cur.execute(
                        "SELECT best_params FROM opt_results "
                        "WHERE id=? AND status='done' AND strategy=? AND symbol=? AND tf=?",
                        (opt_id_int, opt_strategy, symbol, tf),
                    )
                    row = cur.fetchone()
                    opt_params = _json_to_dict(row[0] if row else None)
        except Exception:
            # If opt tables are missing or query fails, just ignore and render chart with defaults.
            opt_results = []
            opt_params = None

    # Apply manual coefficient overrides (strategy3 only).
    if strategy == "my_strategy3.py" and int(use_overrides or 0) == 1:
        overrides: dict = {}
        if p_use_no_trade is not None:
            overrides["use_no_trade"] = bool(p_use_no_trade)
        if p_adx_no_trade_below is not None:
            overrides["adx_no_trade_below"] = float(p_adx_no_trade_below)
        if p_st_factor is not None:
            overrides["st_factor"] = float(p_st_factor)
        if p_rev_cooldown_hrs is not None:
            overrides["rev_cooldown_hrs"] = int(p_rev_cooldown_hrs)
        if p_use_flip_limit is not None:
            overrides["use_flip_limit"] = bool(p_use_flip_limit)
        if p_max_flips_per_day is not None:
            overrides["max_flips_per_day"] = int(p_max_flips_per_day)
        if p_atr_mult is not None:
            overrides["atr_mult"] = float(p_atr_mult)

        if overrides:
            if opt_params is None:
                opt_params = {}
            try:
                opt_params = {**opt_params, **overrides}
            except Exception:
                opt_params.update(overrides)

    return make_chart_html(
        rows,
        symbol=symbol,
        tf=tf,
        limit=limit,
        symbols=symbols,
        tfs=tfs,
        strategy=strategy,
        opt_strategy=opt_strategy,
        opt_results=opt_results,
        opt_id=opt_id_int,
        opt_last=opt_last,
        opt_params=opt_params,
        use_overrides=use_overrides,
    )

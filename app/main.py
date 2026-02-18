import asyncio
import time
from typing import List

import json

from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from app.config import settings
from app.storage.db import init_db, load_bars, last_signal
from app.data.bybit_ws import ws_collect
from app.data.backfill import backfill_on_startup
from app.reporting.chart import make_chart_html

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

from app.reporting.tv_debug import router as tv_debug_router
app.include_router(tv_debug_router)

from app.reporting.tv_ingest import router as tv_ingest_router
app.include_router(tv_ingest_router)

from app.reporting.optimize_web import router as optimize_router
app.include_router(optimize_router)

conn = None

# Simple in-memory cache for bars to speed up repeated chart reloads (especially with large limits like 20000)
_BARS_CACHE = {}  # (symbol, tf, limit) -> (ts_mono, rows)
_BARS_CACHE_TTL_SEC = 20.0
_BARS_CACHE_MAX_ITEMS = 32

def _load_bars_cached(symbol: str, tf: str, limit: int):
    key = (symbol, tf, int(limit))
    now = time.monotonic()
    hit = _BARS_CACHE.get(key)
    if hit is not None:
        t0, rows = hit
        if (now - t0) <= _BARS_CACHE_TTL_SEC:
            return rows
    # IMPORTANT: call the real DB loader (not ourselves) — иначе будет рекурсия.
    # (symbol/tf are normalized by the caller; we still upper-case symbol for safety)
    rows = load_bars(conn, str(symbol).upper(), str(tf), limit=int(limit))
    _BARS_CACHE[key] = (now, rows)
    # trim
    if len(_BARS_CACHE) > _BARS_CACHE_MAX_ITEMS:
        # remove oldest
        oldest = sorted(_BARS_CACHE.items(), key=lambda kv: kv[1][0])[: max(1, len(_BARS_CACHE) - _BARS_CACHE_MAX_ITEMS)]
        for k, _ in oldest:
            _BARS_CACHE.pop(k, None)
    return rows


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
    rows = _load_bars_cached(symbol.upper(), tf, limit)
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
    capital_usd: float = Query(10000.0, ge=0.0, le=1e9),
    strategy: str = Query("my_strategy.py"),
    opt_id: str | None = Query(None),
    opt_run_id: str | None = Query(None),
    opt_cand: int | None = Query(None, ge=0, le=200),
    opt_last: int = Query(20, ge=1, le=200),
    # Manual overrides for strategy3 (only applied when use_overrides=1)
    use_overrides: int = Query(0, ge=0, le=1),
    p_position_usd: float | None = Query(None, ge=0.0, le=1e9),
    p_leverage_mult: float | None = Query(None, ge=0.0, le=100.0),
    p_trade_from_current_capital: bool | None = Query(None),
    p_slippage_ticks: int | None = Query(None, ge=0, le=1000),
    p_tick_size: float | None = Query(None, ge=0.0, le=1e6),
    p_fee_percent: float | None = Query(None, ge=0.0, le=100.0),
    p_spread_ticks: float | None = Query(None, ge=0.0, le=1000.0),
    p_funding_8h_percent: float | None = Query(None, ge=-100.0, le=100.0),
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
    rows = _load_bars_cached(symbol, tf, limit)

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



    # Load params from a multi-seed optimizer run candidate (if provided)
    if opt_run_id and opt_cand is not None:
        try:
            from app.reporting.optimize_web import load_candidates  # noqa: WPS433
            cand = load_candidates(str(opt_run_id))
            if cand and isinstance(cand, dict):
                top = cand.get('top_candidates') or []
                if isinstance(top, list) and 0 <= int(opt_cand) < len(top):
                    rec = top[int(opt_cand)] or {}
                    params = rec.get('params') or {}
                    if isinstance(params, dict):
                        opt_params = dict(params)
                        # Optionally apply the same costs used during optimization.
                        if cand.get('use_costs_in_opt') and isinstance(cand.get('fixed_costs'), dict):
                            opt_params.update(cand.get('fixed_costs') or {})
                        # Force strategy3 when loading candidates (they are for strategy3).
                        strategy = 'my_strategy3.py'
        except Exception:
            pass
    # Apply manual coefficient overrides (strategy3 only).
    if strategy == "my_strategy3.py" and int(use_overrides or 0) == 1:
        overrides: dict = {}
        if p_position_usd is not None:
            overrides["position_usd"] = float(p_position_usd)
        if p_leverage_mult is not None:
            overrides["leverage_mult"] = float(p_leverage_mult)
        if p_trade_from_current_capital is not None:
            overrides["trade_from_current_capital"] = bool(p_trade_from_current_capital)
        if p_slippage_ticks is not None:
            overrides["slippage_ticks"] = int(p_slippage_ticks)
        if p_tick_size is not None:
            overrides["tick_size"] = float(p_tick_size)
        if p_fee_percent is not None:
            overrides["fee_percent"] = float(p_fee_percent)
        if p_spread_ticks is not None:
            overrides["spread_ticks"] = float(p_spread_ticks)
        if p_funding_8h_percent is not None:
            overrides["funding_8h_percent"] = float(p_funding_8h_percent)
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
        capital_usd=capital_usd,
    )

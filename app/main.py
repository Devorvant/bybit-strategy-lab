import asyncio
import json
import time
from typing import Dict, List, Tuple

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse
from starlette.middleware.gzip import GZipMiddleware
from app.config import settings
from app.storage.db import init_db, load_bars, last_signal
from app.data.bybit_ws import ws_collect
from app.data.backfill import backfill_on_startup
from app.reporting.chart import build_plotly_payload, make_chart_html

# Reuse optimizer persistence helpers to store manual snapshots in the same table.
from app.reporting import optimize_web as _optw

app = FastAPI()

# Plotly HTML/JSON compresses extremely well; this makes /chart and /api/* much faster.
app.add_middleware(GZipMiddleware, minimum_size=1000)

from app.reporting.tv_debug import router as tv_debug_router
app.include_router(tv_debug_router)

from app.reporting.tv_ingest import router as tv_ingest_router
app.include_router(tv_ingest_router)

from app.reporting.optimize_web import router as optimize_router
app.include_router(optimize_router)

conn = None

# Very small in-memory cache to avoid re-reading the same bars repeatedly when user tweaks params.
_BARS_CACHE: Dict[Tuple[str, str, int], Tuple[float, list]] = {}
_BARS_CACHE_TTL_SEC = 20.0


def _load_bars_cached(symbol: str, tf: str, limit: int) -> list:
    key = (symbol, tf, int(limit))
    now = time.monotonic()
    rec = _BARS_CACHE.get(key)
    if rec is not None:
        ts0, rows0 = rec
        if now - ts0 <= _BARS_CACHE_TTL_SEC:
            return rows0
    rows = load_bars(conn, symbol, tf, limit=limit)
    _BARS_CACHE[key] = (now, rows)
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


@app.get("/api/chart_update")
def api_chart_update(
    symbol: str = Query("APTUSDT"),
    tf: str = Query(None),
    limit: int = Query(5000, ge=10, le=50000),
    capital_usd: float = Query(10000.0, ge=0.0, le=1e9),
    strategy: str = Query("my_strategy.py"),
    opt_id: str | None = Query(None),
    opt_run_id: str | None = Query(None),
    opt_cand: int | None = Query(None, ge=0, le=200),
    opt_last: int = Query(20, ge=1, le=200),
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
    """Return Plotly traces/layout for in-page updates (no full page reload)."""
    tf = tf or settings.TF
    symbol = symbol.upper()

    rows = _load_bars_cached(symbol, tf, limit)

    # Resolve optional optimized params (same logic as /chart)
    opt_results = []
    opt_params = None
    opt_strategy_map = {
        "my_strategy3.py": "strategy3",
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
                cur = conn.cursor()
                if opt_id_int is not None:
                    cur.execute(
                        "SELECT best_params FROM opt_results "
                        "WHERE id=? AND status='done' AND strategy=? AND symbol=? AND tf=?",
                        (opt_id_int, opt_strategy, symbol, tf),
                    )
                    row = cur.fetchone()
                    opt_params = _json_to_dict(row[0] if row else None)
        except Exception:
            opt_params = None

    # Multi-seed candidate
    if opt_run_id and opt_cand is not None:
        try:
            from app.reporting.optimize_web import load_candidates  # noqa: WPS433

            cand = load_candidates(str(opt_run_id))
            if cand and isinstance(cand, dict):
                top = cand.get("top_candidates") or []
                if isinstance(top, list) and 0 <= int(opt_cand) < len(top):
                    rec = top[int(opt_cand)] or {}
                    params = rec.get("params") or {}
                    if isinstance(params, dict):
                        opt_params = dict(params)
                        if cand.get("use_costs_in_opt") and isinstance(cand.get("fixed_costs"), dict):
                            opt_params.update(cand.get("fixed_costs") or {})
                        strategy = "my_strategy3.py"
        except Exception:
            pass

    # Apply manual overrides (strategy3 only)
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

        opt_params = dict(opt_params or {})
        opt_params.update(overrides)

    return build_plotly_payload(rows, strategy=strategy, strategy_params=opt_params, capital_usd=float(capital_usd))


@app.post("/api/save_chart_snapshot")
def api_save_chart_snapshot(body: dict = Body(...)):
    """Save current chart params/metrics into opt_results.

    We store it in the same table as optimizer finals so it appears:
    - in /optimize "Saved results" table
    - in /chart "Optimized" dropdown
    """

    # --- Parse payload (be tolerant) ---
    symbol = str(body.get("symbol") or "APTUSDT").upper()
    tf = str(body.get("tf") or (settings.TF or "30"))
    limit = int(body.get("limit") or 5000)
    limit = max(10, min(50000, limit))
    strategy_file = str(body.get("strategy") or "my_strategy3.py")
    capital_usd = float(body.get("capital_usd") or 10000.0)
    opt_id = body.get("opt_id")
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}

    # Map chart strategy filename -> optimizer key stored in opt_results.strategy
    opt_strategy_map = {
        "my_strategy3.py": "strategy3",
    }
    opt_strategy = opt_strategy_map.get(strategy_file)
    if opt_strategy is None:
        return {"ok": False, "error": f"Unsupported strategy for snapshot: {strategy_file}"}

    # --- Load bars ---
    rows = _load_bars_cached(symbol, tf, limit)
    if not rows:
        return {"ok": False, "error": "No bars loaded"}

    # --- Load base optimized params if opt_id provided ---
    base_params = None
    opt_id_int: int | None = None
    try:
        if opt_id is not None and str(opt_id).strip().isdigit():
            opt_id_int = int(str(opt_id).strip())
    except Exception:
        opt_id_int = None

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

    if opt_id_int is not None:
        try:
            if settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT best_params FROM opt_results WHERE id=%s AND status='done' AND strategy=%s AND symbol=%s AND tf=%s",
                        (opt_id_int, opt_strategy, symbol, tf),
                    )
                    row = cur.fetchone()
                base_params = _json_to_dict(row[0] if row else None)
            else:
                cur = conn.cursor()
                cur.execute(
                    "SELECT best_params FROM opt_results WHERE id=? AND status='done' AND strategy=? AND symbol=? AND tf=?",
                    (opt_id_int, opt_strategy, symbol, tf),
                )
                row = cur.fetchone()
                base_params = _json_to_dict(row[0] if row else None)
        except Exception:
            base_params = None

    # --- Build final params for backtest (defaults -> base_params -> overrides) ---
    # Keep in sync with chart defaults (strategy3 section)
    params: dict = {
        "position_usd": 1000.0,
        "leverage_mult": 1.0,
        "trade_from_current_capital": False,
        "slippage_ticks": 2,
        "tick_size": 0.0001,
        "fee_percent": 0.06,
        "spread_ticks": 1.0,
        "funding_8h_percent": 0.01,
        "use_no_trade": True,
        "adx_len": 14,
        "adx_smooth": 14,
        "adx_no_trade_below": 14.0,
        "st_atr_len": 14,
        "st_factor": 4.0,
        "use_rev_cooldown": True,
        "rev_cooldown_hrs": 8,
        "use_flip_limit": False,
        "max_flips_per_day": 6,
        "use_emergency_sl": True,
        "atr_len": 14,
        "atr_mult": 3.0,
        "close_at_end": False,
    }
    if isinstance(base_params, dict):
        params.update(base_params)
    params.update(overrides)
    params["capital_usd"] = float(capital_usd)

    # --- Run backtest and compute metrics/score ---
    try:
        from app.backtest.strategy3_backtest import backtest_strategy3  # noqa: WPS433

        bt = backtest_strategy3(rows, **params)
    except Exception as e:
        return {"ok": False, "error": f"Backtest failed: {e}"}

    try:
        # Use optimizer metric helpers for consistency
        position_usd = float(params.get("position_usd") or 1000.0)
        ts0 = int(rows[0][0])
        ts1 = int(rows[-1][0])
        m_eq = _optw._equity_metrics(list(bt.equity or []), base=position_usd)
        m_tr = _optw._trade_metrics(list(bt.trades or []), ts0_ms=ts0, ts1_ms=ts1, base=position_usd)
        trades_n = int(len(bt.trades or []))
        metrics = {**m_eq, **m_tr, "trades": trades_n}

        weights = {"w_ret": 2.0, "w_sharpe": 0.5, "w_dd": 1.5, "w_vol": 0.1, "w_time_dd": 4.0, "w_ulcer": 2.0}
        min_trades = 5
        score = float(_optw._score(metrics, trades_n, weights, min_trades))
        metrics["snapshot_score"] = score
    except Exception as e:
        # If metrics fail, still allow saving params
        score = None
        metrics = {"error": f"metrics_failed: {e}"}

    # --- Persist (same table as optimizer) ---
    try:
        _optw._ensure_opt_results_table(conn)
        cfg = {
            "source": "chart_snapshot",
            "base_opt_id": opt_id_int,
            "weights": {"w_ret": 2.0, "w_sharpe": 0.5, "w_dd": 1.5, "w_vol": 0.1, "w_time_dd": 4.0, "w_ulcer": 2.0},
            "min_trades": 5,
        }
        # Store only strategy params (exclude capital_usd which is UI-only)
        save_params = dict(params)
        save_params.pop("capital_usd", None)
        _optw._insert_opt_result(
            conn,
            strategy=opt_strategy,
            symbol=symbol,
            tf=tf,
            config=cfg,
            status="done",
            duration_sec=0.0,
            trials_done=0,
            best_score=score,
            best_params=save_params,
            best_metrics=metrics,
        )

        # Return new id (SQLite: lastrowid; Postgres: we don't have it here) -> reload list client-side.
        # We'll query the last inserted row for this symbol/tf/strategy.
        new_id = None
        try:
            if settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM opt_results WHERE strategy=%s AND symbol=%s AND tf=%s ORDER BY created_at DESC LIMIT 1",
                        (opt_strategy, symbol, tf),
                    )
                    r = cur.fetchone()
                new_id = int(r[0]) if r and r[0] is not None else None
            else:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM opt_results WHERE strategy=? AND symbol=? AND tf=? ORDER BY created_at DESC LIMIT 1",
                    (opt_strategy, symbol, tf),
                )
                r = cur.fetchone()
                new_id = int(r[0]) if r and r[0] is not None else None
        except Exception:
            new_id = None

        return {"ok": True, "id": new_id, "score": score, "metrics": metrics}
    except Exception as e:
        return {"ok": False, "error": f"Save failed: {e}"}

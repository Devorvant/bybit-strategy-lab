from __future__ import annotations

import asyncio
import collections
import html
import json
import time
import uuid
import os
import datetime as dt
from pathlib import Path
import threading
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings
from app.storage.db import load_bars

from app.backtest.strategy3_backtest import backtest_strategy3


router = APIRouter()


# ------------------------------
# In-memory run state (online progress)
# ------------------------------


@dataclass
class RunState:
    run_id: str
    status: str = "queued"  # queued|running|done|error
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    cfg: Dict[str, Any] = field(default_factory=dict)
    trials_done: int = 0
    best_score: float = float("-inf")
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_trial: int = 0

    logs: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=400))
    error: Optional[str] = None


RUNS: Dict[str, RunState] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Cross-worker persistence (Railway часто запускает несколько worker-процессов).
# Храним состояние оптимизации в файловой системе (/tmp), чтобы страница
# /optimize/run/<id> могла читать прогресс независимо от того, какой worker
# обслуживает HTTP-запрос.
# ──────────────────────────────────────────────────────────────────────────────
RUN_DIR = Path(os.environ.get("OPTIMIZER_RUN_DIR", "/tmp/optimizer_runs"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

_FILE_LOCKS: Dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()

def _lock_for(run_id: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        lk = _FILE_LOCKS.get(run_id)
        if lk is None:
            lk = threading.Lock()
            _FILE_LOCKS[run_id] = lk
        return lk

def _run_dir(run_id: str) -> Path:
    return RUN_DIR / run_id

def _log_path(run_id: str) -> Path:
    return _run_dir(run_id) / "log.txt"

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def _tail_lines(path: Path, n: int = 200, max_bytes: int = 200_000) -> List[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    read_from = max(0, size - max_bytes)
    with path.open("rb") as f:
        f.seek(read_from)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    ls = text.splitlines()
    return ls[-n:]

def persist_state(st: RunState) -> None:
    # Минимальное состояние (без массивов баров и т.п.)
    trials_total = int(st.cfg.get("trials", 0)) if st.cfg else 0
    now = time.time()
    elapsed = (st.finished_at or now) - st.started_at if st.started_at else 0.0
    d: Dict[str, Any] = {
        "run_id": st.run_id,
        "status": st.status,
        # UI expects these field names:
        "trial": int(st.trials_done),
        "trials": trials_total,
        "elapsed_s": float(max(0.0, elapsed)),
        # legacy/backward compatible fields:
        "trials_done": int(st.trials_done),
        "best_score": None if st.best_score == float("-inf") else st.best_score,
        "best_trial": int(st.best_trial),
        "best_params": st.best_params,
        "best_metrics": st.best_metrics,
        "cfg": st.cfg,
        "error": st.error,
        "started_at": float(st.started_at),
        "updated_at": now,
    }
    with _lock_for(st.run_id):
        _atomic_write_json(_run_dir(st.run_id) / "progress.json", d)

def persist_log(run_id: str, msg: str) -> None:
    with _lock_for(run_id):
        _append_line(_log_path(run_id), msg)

def _add_log(st: RunState, msg: str) -> None:
    """Append to in-memory log and persist to /tmp.

    This is critical on Railway where HTTP requests and the optimizer may run in
    different worker processes.
    """
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"{ts} {msg}"
    st.logs.append(line)
    try:
        persist_log(st.run_id, line)
    except Exception:
        # Never fail the optimizer because of logging.
        pass

_DB_STATE_LOCK = threading.Lock()


def _persist_throttled(st: RunState, *, force: bool = False, conn=None) -> None:
    """Persist progress snapshots.

    - Always writes a small JSON snapshot to /tmp (fast, per-replica)
    - When DB connection is provided, also upserts into opt_run_state so that
      progress is visible across Railway replicas.
    """

    now = time.time()
    last = getattr(st, "_last_persist", 0.0)
    if not (force or (now - last) >= 1.0 or (st.trials_done % 10 == 0)):
        return

    try:
        persist_state(st)
        setattr(st, "_last_persist", now)
    except Exception:
        pass

    if conn is None:
        return

    # /tmp is not shared between replicas; keep DB snapshot updated.
    try:
        payload = json.loads((_run_dir(st.run_id) / "progress.json").read_text(encoding="utf-8"))
    except Exception:
        payload = {
            "run_id": st.run_id,
            "status": st.status,
            "trial": int(st.trials_done),
            "trials": int(st.cfg.get("trials", 0)) if st.cfg else 0,
            "elapsed_s": float(max(0.0, (st.finished_at or now) - st.started_at)),
            "best_score": None if st.best_score == float("-inf") else st.best_score,
            "best_trial": int(st.best_trial),
            "best_params": st.best_params,
            "best_metrics": st.best_metrics,
            "cfg": st.cfg,
            "error": st.error,
            "started_at": float(st.started_at),
            "updated_at": now,
        }

    try:
        with _DB_STATE_LOCK:
            _ensure_opt_run_state_table(conn)
            _upsert_opt_run_state(conn, st.run_id, payload, list(st.logs))
    except Exception:
        # Never fail optimizer due to persistence.
        pass
RUN_LOCK = asyncio.Lock()


# ------------------------------
# DB helpers (persist only final result)
# ------------------------------


def _is_postgres() -> bool:
    url = settings.DATABASE_URL or ""
    return url.startswith("postgres")


def _ensure_opt_results_table(conn) -> None:
    """Create opt_results table if missing.

    We persist only final results here (no per-iteration writes).
    """

    ddl_pg = """
    CREATE TABLE IF NOT EXISTS opt_results (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      strategy TEXT NOT NULL,
      symbol TEXT NOT NULL,
      tf TEXT NOT NULL,
      config JSONB,
      status TEXT NOT NULL,
      duration_sec DOUBLE PRECISION,
      trials_done INT,
      best_score DOUBLE PRECISION,
      best_params JSONB,
      best_metrics JSONB
    );
    CREATE INDEX IF NOT EXISTS opt_results_created_at_idx ON opt_results(created_at DESC);
    """

    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS opt_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      strategy TEXT NOT NULL,
      symbol TEXT NOT NULL,
      tf TEXT NOT NULL,
      config TEXT,
      status TEXT NOT NULL,
      duration_sec REAL,
      trials_done INTEGER,
      best_score REAL,
      best_params TEXT,
      best_metrics TEXT
    );
    """

    if _is_postgres():
        try:
            with conn.cursor() as cur:
                for stmt in [s.strip() for s in ddl_pg.split(";") if s.strip()]:
                    cur.execute(stmt + ";")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn.executescript(ddl_sqlite)
        conn.commit()


def _insert_opt_result(
    conn,
    *,
    strategy: str,
    symbol: str,
    tf: str,
    config: Dict[str, Any],
    status: str,
    duration_sec: float,
    trials_done: int,
    best_score: Optional[float],
    best_params: Optional[Dict[str, Any]],
    best_metrics: Optional[Dict[str, Any]],
) -> None:
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO opt_results(strategy, symbol, tf, config, status, duration_sec, trials_done, best_score, best_params, best_metrics)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        strategy,
                        symbol,
                        tf,
                        json.dumps(config),
                        status,
                        float(duration_sec),
                        int(trials_done),
                        float(best_score) if best_score is not None else None,
                        json.dumps(best_params) if best_params else None,
                        json.dumps(best_metrics) if best_metrics else None,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn.execute(
            """
            INSERT INTO opt_results(strategy, symbol, tf, config, status, duration_sec, trials_done, best_score, best_params, best_metrics)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                strategy,
                symbol,
                tf,
                json.dumps(config),
                status,
                float(duration_sec),
                int(trials_done),
                float(best_score) if best_score is not None else None,
                json.dumps(best_params) if best_params else None,
                json.dumps(best_metrics) if best_metrics else None,
            ),
        )
        conn.commit()


def _load_last_results(conn, limit: int = 30) -> List[Dict[str, Any]]:
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, strategy, symbol, tf, status, duration_sec, trials_done, best_score, best_params, best_metrics
                FROM opt_results
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    else:
        cur = conn.execute(
            """
            SELECT id, created_at, strategy, symbol, tf, status, duration_sec, trials_done, best_score, best_params, best_metrics
            FROM opt_results
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "created_at": str(r[1]),
                "strategy": r[2],
                "symbol": r[3],
                "tf": r[4],
                "status": r[5],
                "duration_sec": float(r[6]) if r[6] is not None else None,
                "trials_done": int(r[7]) if r[7] is not None else None,
                "best_score": float(r[8]) if r[8] is not None else None,
                "best_params": r[9],
                "best_metrics": r[10],
            }
        )
    return out


# ------------------------------
# DB helpers (cross-replica progress)
# ------------------------------


def _ensure_opt_run_state_table(conn) -> None:
    """Create opt_run_state table if missing.

    /tmp is NOT shared between Railway replicas. We keep a latest snapshot in DB
    (one row per run_id) so that /optimize/run/<id> can always show progress.
    """

    ddl_pg = """
    CREATE TABLE IF NOT EXISTS opt_run_state (
      run_id TEXT PRIMARY KEY,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      payload JSONB,
      logs TEXT
    );
    CREATE INDEX IF NOT EXISTS opt_run_state_updated_at_idx ON opt_run_state(updated_at DESC);
    """

    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS opt_run_state (
      run_id TEXT PRIMARY KEY,
      updated_at TEXT NOT NULL,
      payload TEXT,
      logs TEXT
    );
    """

    if _is_postgres():
        try:
            with conn.cursor() as cur:
                for stmt in [s.strip() for s in ddl_pg.split(";") if s.strip()]:
                    cur.execute(stmt + ";")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn.executescript(ddl_sqlite)
        conn.commit()


def _upsert_opt_run_state(conn, run_id: str, payload: Dict[str, Any], logs: List[str]) -> None:
    logs_txt = "\n".join(logs[-400:])
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO opt_run_state(run_id, updated_at, payload, logs)
                    VALUES(%s, now(), %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                      SET updated_at = now(),
                          payload = EXCLUDED.payload,
                          logs = EXCLUDED.logs
                    """,
                    (run_id, json.dumps(payload), logs_txt),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        conn.execute(
            """
            INSERT OR REPLACE INTO opt_run_state(run_id, updated_at, payload, logs)
            VALUES(?, datetime('now'), ?, ?)
            """,
            (run_id, json.dumps(payload), logs_txt),
        )
        conn.commit()


def _load_opt_run_state(conn, run_id: str) -> Optional[Dict[str, Any]]:
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute("SELECT payload, logs, updated_at FROM opt_run_state WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
    else:
        cur = conn.execute("SELECT payload, logs, updated_at FROM opt_run_state WHERE run_id=?", (run_id,))
        row = cur.fetchone()

    if not row:
        return None

    payload_raw = row[0]
    logs_txt = row[1] or ""
    try:
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
    except Exception:
        payload = {"run_id": run_id, "status": "unknown"}

    payload["logs"] = [ln for ln in str(logs_txt).splitlines() if ln.strip()]
    payload["updated_at"] = str(row[2])
    return payload


# ------------------------------
# Optimizer core
# ------------------------------


def _equity_metrics(pnl_curve: List[float], base: float) -> Dict[str, float]:
    """Compute metrics from cumulative PnL curve.

    Strategy3 backtest returns equity as cumulative PnL in USD (starts near 0).
    For meaningful returns/DD we convert to *total equity* = base + pnl.

    Returns:
      - ret: total return over the period (fraction, e.g. 0.25 = +25%)
      - dd: max drawdown (fraction, positive, e.g. 0.2 = 20%)
      - vol: volatility of log returns per bar
      - sharpe: sharpe of log returns per bar (scaled by sqrt(N))
      - time_in_dd: fraction of bars spent below prior peak (0..1)
      - ulcer: ulcer index on drawdowns (RMS drawdown fraction, 0..)
    """
    if base <= 0:
        base = 1.0

    if not pnl_curve or len(pnl_curve) < 2:
        return {"ret": 0.0, "dd": 0.0, "vol": 0.0, "sharpe": 0.0, "time_in_dd": 0.0, "ulcer": 0.0}

    equity = [base + float(x) for x in pnl_curve]
    ret = (equity[-1] / base) - 1.0

    peak = equity[0]
    max_dd = 0.0  # negative
    dd_sq_sum = 0.0
    dd_n = 0
    bars_in_dd = 0

    for v in equity:
        if v > peak:
            peak = v
        if peak <= 0:
            continue
        dd = (v / peak) - 1.0  # 0 at peak, negative in drawdown
        if dd < 0:
            bars_in_dd += 1
            dd_frac = -dd  # positive drawdown fraction
            dd_sq_sum += dd_frac * dd_frac
            dd_n += 1
        if dd < max_dd:
            max_dd = dd

    dd_abs = abs(max_dd)
    time_in_dd = bars_in_dd / max(1, len(equity))
    ulcer = math.sqrt(dd_sq_sum / dd_n) if dd_n else 0.0

    # returns series
    rs: List[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        cur = equity[i]
        if prev > 0 and cur > 0:
            rs.append(cur / prev)

    if len(rs) < 2:
        return {
            "ret": float(ret),
            "dd": float(dd_abs),
            "vol": 0.0,
            "sharpe": 0.0,
            "time_in_dd": float(time_in_dd),
            "ulcer": float(ulcer),
        }

    log_r = [math.log(x) for x in rs if x > 0]
    if len(log_r) < 2:
        return {
            "ret": float(ret),
            "dd": float(dd_abs),
            "vol": 0.0,
            "sharpe": 0.0,
            "time_in_dd": float(time_in_dd),
            "ulcer": float(ulcer),
        }

    mean_r = sum(log_r) / len(log_r)
    var = sum((x - mean_r) ** 2 for x in log_r) / (len(log_r) - 1)
    vol = math.sqrt(var) if var > 0 else 0.0
    sharpe = 0.0 if vol == 0 else (mean_r / vol) * math.sqrt(len(log_r))

    return {
        "ret": float(ret),
        "dd": float(dd_abs),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "time_in_dd": float(time_in_dd),
        "ulcer": float(ulcer),
    }

def _score(m: Dict[str, float], trades: int, weights: Dict[str, float], min_trades: int) -> float:
    """Composite objective to maximize.

    Goal: smooth, consistently rising equity — not just a positive tail.
    We therefore penalize drawdown *and* time spent in drawdown, plus ulcer index.
    """
    s = 0.0
    s += weights.get("w_ret", 2.0) * m.get("ret", 0.0)
    s += weights.get("w_sharpe", 0.5) * m.get("sharpe", 0.0)

    s -= weights.get("w_dd", 1.5) * m.get("dd", 0.0)
    s -= weights.get("w_vol", 0.1) * m.get("vol", 0.0)

    # Penalize "tail wins": long time below peak and persistent drawdowns
    s -= weights.get("w_time_dd", 1.0) * m.get("time_in_dd", 0.0)
    s -= weights.get("w_ulcer", 0.5) * m.get("ulcer", 0.0)

    # Keep existing minimum-trades penalty (user requested not to tune trade count yet)
    if trades < min_trades:
        s -= 0.2 * (min_trades - trades) / max(1, min_trades)

    return float(s)

def _train_val_split(bars: List[Tuple[int, float, float, float, float, float]], train_frac: float):
    n = len(bars)
    k = max(50, int(n * train_frac))
    k = min(k, n - 20)
    return bars[:k], bars[k:]


def _sample_params(rng, position_usd: float) -> Dict[str, Any]:
    return {
        "position_usd": float(position_usd),
        "use_no_trade": True,
        "adx_len": 14,
        "adx_smooth": 14,
        "adx_no_trade_below": rng.uniform(10.0, 25.0),
        "st_atr_len": 14,
        "st_factor": rng.uniform(2.0, 6.0),
        "use_rev_cooldown": True,
        "rev_cooldown_hrs": rng.choice([0, 2, 4, 6, 8, 12, 16]),
        "use_flip_limit": False,
        "max_flips_per_day": 6,
        "use_emergency_sl": True,
        "atr_len": 14,
        "atr_mult": rng.uniform(1.5, 5.0),
        "close_at_end": False,
    }


def _run_optimizer_sync(run: RunState, bars: List[Tuple[int, float, float, float, float, float]], conn) -> None:
    import random

    cfg = run.cfg
    symbol = cfg["symbol"]
    tf = cfg["tf"]
    trials = int(cfg["trials"])
    max_seconds = int(cfg["max_seconds"])
    patience = int(cfg["patience"])
    train_frac = float(cfg.get("train_frac", 0.7))
    seed = int(cfg.get("seed", 42))
    position_usd = float(cfg.get("position_usd", 1000.0))
    min_trades = int(cfg.get("min_trades", 5))
    weights = {
        "w_ret": float(cfg.get("w_ret", 2.0)),
        "w_dd": float(cfg.get("w_dd", 1.5)),
        "w_vol": float(cfg.get("w_vol", 0.1)),
        "w_sharpe": float(cfg.get("w_sharpe", 0.5)),
        "w_time_dd": float(cfg.get("w_time_dd", 1.0)),
        "w_ulcer": float(cfg.get("w_ulcer", 0.5)),
    }

    rng = random.Random(seed)
    t0 = time.time()
    no_improve = 0

    run.status = "running"
    _add_log(run, f"RUN_START run_id={run.run_id} symbol={symbol} tf={tf} bars={len(bars)} trials={trials}")
    # Make sure the run page can show status even if it is served by another worker.
    _persist_throttled(run, force=True, conn=conn)

    try:
        for trial in range(1, trials + 1):
            if max_seconds and (time.time() - t0) >= max_seconds:
                _add_log(run, f"STOP max_seconds reached at trial={trial-1}")
                break

            params = _sample_params(rng, position_usd)
            _, va_bars = _train_val_split(bars, train_frac)
            bt = backtest_strategy3(va_bars, **params)
            m = _equity_metrics(bt.equity, base=position_usd)
            trades = len(bt.trades)
            s = _score(m, trades, weights, min_trades)

            run.trials_done = trial
            _persist_throttled(run, conn=conn)

            if s > run.best_score:
                run.best_score = s
                run.best_params = params
                run.best_trial = trial
                run.best_metrics = {**m, "trades": trades, "val_score": s}
                no_improve = 0
                _add_log(
                    run,
                    f"[{trial}/{trials}] best={run.best_score:.6f} (trial={run.best_trial}) "
                    f"ret={m['ret']:.4f} dd={m['dd']:.4f} sharpe={m['sharpe']:.3f} trades={trades} "
                    f"st_factor={params['st_factor']:.2f} adx_nt<{params['adx_no_trade_below']:.1f} atr_mult={params['atr_mult']:.2f} cooldown={params['rev_cooldown_hrs']}h"
                )
            else:
                no_improve += 1

            # occasional progress line
            if trial % max(1, trials // 20) == 0:
                _add_log(run, f"PROGRESS trial={trial} best={run.best_score:.6f} best_trial={run.best_trial}")

            if patience and no_improve >= patience:
                _add_log(run, f"EARLY_STOP patience={patience} reached at trial={trial} best_trial={run.best_trial}")
                break

        run.finished_at = time.time()
        run.status = "done"
        _persist_throttled(run, force=True, conn=conn)

        dur = run.finished_at - run.started_at
        _add_log(
            run,
            f"DONE run_id={run.run_id} trials_done={run.trials_done} best_score={run.best_score:.6f} best_trial={run.best_trial} duration_sec={dur:.1f}"
        )
        if run.best_params:
            _add_log(run, "BEST_PARAMS " + json.dumps(run.best_params, ensure_ascii=False))
        if run.best_metrics:
            _add_log(run, "BEST_METRICS " + json.dumps(run.best_metrics, ensure_ascii=False))

        _persist_throttled(run, force=True, conn=conn)

        # Persist ONLY final result
        _ensure_opt_results_table(conn)
        _insert_opt_result(
            conn,
            strategy="strategy3",
            symbol=symbol,
            tf=str(tf),
            config=cfg,
            status="done",
            duration_sec=float(dur),
            trials_done=int(run.trials_done),
            best_score=float(run.best_score) if run.best_score != float("-inf") else None,
            best_params=run.best_params,
            best_metrics=run.best_metrics,
        )

    except Exception as e:  # pragma: no cover
        run.finished_at = time.time()
        run.status = "error"
        run.error = repr(e)
        _add_log(run, "ERROR " + run.error)
        _persist_throttled(run, force=True, conn=conn)
        # Persist error summary
        try:
            _ensure_opt_results_table(conn)
            _insert_opt_result(
                conn,
                strategy="strategy3",
                symbol=symbol,
                tf=str(tf),
                config=cfg,
                status="error",
                duration_sec=float(run.finished_at - run.started_at),
                trials_done=int(run.trials_done),
                best_score=float(run.best_score) if run.best_score != float("-inf") else None,
                best_params=run.best_params,
                best_metrics={"error": run.error, **(run.best_metrics or {})},
            )
        except Exception:
            pass


# ------------------------------
# HTML
# ------------------------------


def _page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang='ru'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background: #0b1020; color: #e8eaf2; }}
    a {{ color: #9ad1ff; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 18px; }}
    .card {{ background: #111a33; border: 1px solid #1f2a4d; border-radius: 14px; padding: 14px; margin: 12px 0; }}
    .row {{ display:flex; gap: 10px; flex-wrap: wrap; align-items: end; }}
    label {{ font-size: 12px; opacity: .85; display:block; margin-bottom: 4px; }}
    input, select {{ background: #0d1430; border: 1px solid #22315c; color: #e8eaf2; border-radius: 10px; padding: 8px 10px; }}
    button {{ background: #2b78ff; border: none; color: white; border-radius: 10px; padding: 9px 12px; cursor:pointer; font-weight:600; }}
    button:disabled {{ opacity:.5; cursor:not-allowed; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0d1430; border: 1px solid #22315c; border-radius: 10px; padding: 10px; max-height: 420px; overflow:auto; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ text-align:left; padding: 8px; border-bottom: 1px solid #1f2a4d; font-size: 13px; }}
    .muted {{ opacity: .75; }}
    .pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #22315c; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
      <div>
        <div style='font-size:20px; font-weight:700;'>{html.escape(title)}</div>
        <div class='muted' style='margin-top:4px;'>Оптимизация Strategy 3 (2h). Итерации показываем в вебе, в Postgres сохраняем только итог.</div>
      </div>
      <div>
        <a href='/chart'>Chart</a>
      </div>
    </div>
    {body}
  </div>
</body>
</html>"""


def _optimize_index_html(conn) -> str:
    # Distinct symbols/tfs from DB (reuse logic from main.py, but local to avoid circular import)
    symbols = [s.strip().upper() for s in settings.SYMBOLS if s.strip()]
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
                symbols = sorted(set(symbols + [r[0] for r in cur.fetchall()]))
        else:
            cur = conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
            symbols = sorted(set(symbols + [r[0] for r in cur.fetchall()]))
    except Exception:
        symbols = sorted(set(symbols))

    default_symbol = symbols[0] if symbols else "APTUSDT"

    # last saved results
    try:
        _ensure_opt_results_table(conn)
        last = _load_last_results(conn, limit=25)
    except Exception:
        last = []

    rows_html = ""
    for r in last:
        bp = r.get("best_params")
        bm = r.get("best_metrics")
        bs = r.get("best_score")
        bs_str = "" if bs is None else f"{float(bs):.6f}"
        rows_html += "<tr>" + "".join(
            [
                f"<td>{html.escape(str(r.get('created_at')))}</td>",
                f"<td>{html.escape(str(r.get('symbol')))}</td>",
                f"<td>{html.escape(str(r.get('tf')))}</td>",
                f"<td><span class='pill'>{html.escape(str(r.get('status')))}</span></td>",
                f"<td>{'' if r.get('trials_done') is None else int(r.get('trials_done'))}</td>",
                f"<td>{bs_str}</td>",
                f"<td class='muted'>{html.escape(str(bp)[:140]) if bp else ''}</td>",
                f"<td class='muted'>{html.escape(str(bm)[:140]) if bm else ''}</td>",
            ]
        ) + "</tr>"

    body = f"""
    <div class='card'>
      <form id='startForm'>
        <div class='row'>
          <div>
            <label>symbol</label>
            <select name='symbol'>
              {''.join([f"<option value='{html.escape(s)}' {'selected' if s==default_symbol else ''}>{html.escape(s)}</option>" for s in symbols])}
            </select>
          </div>
          <div>
            <label>tf</label>
            <input name='tf' value='120' style='width:90px'/>
          </div>
          <div>
            <label>limit-bars</label>
            <input name='limit_bars' value='5000' style='width:110px'/>
          </div>
          <div>
            <label>trials</label>
            <input name='trials' value='2000' style='width:110px'/>
          </div>
          <div>
            <label>max-seconds</label>
            <input name='max_seconds' value='900' style='width:110px'/>
          </div>
          <div>
            <label>patience</label>
            <input name='patience' value='300' style='width:110px'/>
          </div>
          <div>
            <label>position_usd</label>
            <input name='position_usd' value='1000' style='width:110px'/>
          </div>
          <div>
            <button type='submit'>Start optimize</button>
          </div>
        </div>
        <div class='muted' style='margin-top:10px;'>В Postgres сохраняем только финал. Прогресс смотри на странице run.</div>
      </form>
    </div>

    <div class='card'>
      <div style='display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px;'>
        <div style='font-weight:700;'>Saved results (final only)</div>
        <div class='row' style='gap:8px;'>
          <button type='button' id='btnClearErrors' style='background:#22315c;'>Clear errors</button>
          <button type='button' id='btnClearAll' style='background:#5c2b2b;'>Clear all</button>
        </div>
      </div>
      <table>
        <thead><tr>
          <th>created</th><th>symbol</th><th>tf</th><th>status</th><th>trials</th><th>best_score</th><th>best_params</th><th>best_metrics</th>
        </tr></thead>
        <tbody>
          {rows_html if rows_html else "<tr><td colspan='8' class='muted'>No results yet.</td></tr>"}
        </tbody>
      </table>
    </div>

    <script>
      const form = document.getElementById('startForm');
      form.addEventListener('submit', async (e) => {{
        e.preventDefault();
        const fd = new FormData(form);
        const payload = Object.fromEntries(fd.entries());
        const res = await fetch('/api/optimize/start', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const j = await res.json();
        if (!res.ok) {{
          alert(j.detail || 'Failed');
          return;
        }}
        window.location.href = `/optimize/run/${{j.run_id}}`;
      }});

      async function clearResults(mode) {{
        const ok = (mode === 'all')
          ? confirm('Очистить ВСЕ результаты оптимизации?')
          : confirm('Очистить только error результаты?');
        if (!ok) return;
        const res = await fetch('/api/optimize/clear', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ mode }})
        }});
        const j = await res.json();
        if (!res.ok) {{
          alert(j.detail || 'Failed');
          return;
        }}
        window.location.reload();
      }}

      const btnErr = document.getElementById('btnClearErrors');
      const btnAll = document.getElementById('btnClearAll');
      if (btnErr) btnErr.addEventListener('click', () => clearResults('errors'));
      if (btnAll) btnAll.addEventListener('click', () => clearResults('all'));
    </script>
    """

    return _page_shell("Optimizer", body)


def _optimize_run_html(run_id: str) -> str:
    run_id_safe = html.escape(run_id)
    body_tpl = """
    <div class="header">
      <div>
        <h1>Optimizer run</h1>
        <div class="muted">Оптимизация Strategy 3 (2h). Итерации показываем в вебе, в Postgres сохраняем только итог.</div>
      </div>
      <div class="row">
        <a class="btn" href="/chart">Chart</a>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div>
          <div class="muted">run_id:</div>
          <div><span class="pill">__RUN_ID__</span></div>
        </div>
        <div><a class="btn" href="/optimize">← back</a></div>
      </div>
      <div style="height:8px"></div>

      <div class="muted" id="status">loading...</div>
    </div>

    <div class="card">
      <h2>Best</h2>
      <pre id="best">—</pre>
    </div>

    <div class="card">
      <h2>Progress log</h2>
      <pre id="log">loading...</pre>
    </div>

    <script>
      const runId = "__RUN_ID__";
      const elStatus = document.getElementById('status');
      const elBest = document.getElementById('best');
      const elLog = document.getElementById('log');

      async function tick() {
        try {
          const r = await fetch(`/api/optimize/status?run_id=${encodeURIComponent(runId)}`);
          const j = await r.json();
          elStatus.textContent = `status=${j.status}  trial=${j.trial || 0}/${j.trials || 0}  elapsed=${j.elapsed_s || 0}s`;
          if (j.best_score !== null && j.best_score !== undefined) {
            elBest.textContent = JSON.stringify({best_score: j.best_score, best_params: j.best_params, best_metrics: j.best_metrics}, null, 2);
          }
          elLog.textContent = (j.logs || []).join('\n');
          if (j.status === 'done' || j.status === 'error') {
            clearInterval(timer);
          }
        } catch (e) {
          elStatus.textContent = 'error: ' + (e && e.message ? e.message : e);
        }
      }
      tick();
      const timer = setInterval(tick, 1000);
    </script>
    """
    body = body_tpl.replace("__RUN_ID__", run_id_safe)
    return _page_shell("Optimizer run", body)


# ------------------------------
# Routes
# ------------------------------


@router.get("/optimize", response_class=HTMLResponse)
def optimize_index():
    from app.main import conn  # noqa: WPS433

    return _optimize_index_html(conn)


@router.get("/optimize/run/{run_id}", response_class=HTMLResponse)
def optimize_run(run_id: str):
    return _optimize_run_html(run_id)


@router.post("/api/optimize/start")
async def optimize_start(payload: Dict[str, Any]):
    """Start optimizer.

    payload fields (strings from form):
      symbol, tf, limit_bars, trials, max_seconds, patience, position_usd
    """

    from app.main import conn  # noqa: WPS433

    symbol = str(payload.get("symbol", "APTUSDT")).upper()
    tf = str(payload.get("tf", "120"))
    limit_bars = int(payload.get("limit_bars", 5000))
    trials = int(payload.get("trials", 2000))
    max_seconds = int(payload.get("max_seconds", 900))
    patience = int(payload.get("patience", 300))
    position_usd = float(payload.get("position_usd", 1000))

    # Load bars first (fast fail)
    bars = load_bars(conn, symbol, tf, limit=limit_bars)
    if len(bars) < 300:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Not enough bars: {len(bars)}. Need >= 300. Check symbol/tf data."},
        )

    run_id = uuid.uuid4().hex[:10]
    cfg = {
        "symbol": symbol,
        "tf": tf,
        "limit_bars": limit_bars,
        "trials": trials,
        "max_seconds": max_seconds,
        "patience": patience,
        "position_usd": position_usd,
        "train_frac": 0.7,
        "seed": 42,
        "min_trades": 5,
        # weights (tune later)
        "w_ret": 2.0,
        "w_dd": 1.5,
        "w_vol": 0.1,
        "w_sharpe": 0.5,
        "w_time_dd": 1.0,
        "w_ulcer": 0.5,
    }

    state = RunState(run_id=run_id, cfg=cfg)
    persist_state(state)
    # Ensure DB table for cross-replica progress is ready and persist
    # an initial snapshot so that the run page works across Railway replicas.
    try:
        _ensure_opt_run_state_table(conn)
    except Exception:
        # Don't fail start if table init fails; status endpoint will fall back.
        pass
    _persist_throttled(state, force=True, conn=conn)
    async with RUN_LOCK:
        RUNS[run_id] = state

    # Run in background thread to avoid blocking FastAPI event loop
    async def _bg():
        # IMPORTANT: don't share the global DB connection across threads.
        # psycopg2 connections are not thread-safe; sharing can cause hangs that
        # look like the run page being stuck on "loading...".
        try:
            from app.storage.db import get_conn  # noqa: WPS433

            thread_conn = get_conn()
        except Exception:
            thread_conn = conn

        try:
            await asyncio.to_thread(_run_optimizer_sync, state, bars, thread_conn)
        finally:
            try:
                if thread_conn is not conn:
                    thread_conn.close()
            except Exception:
                pass

    asyncio.create_task(_bg())

    return {"ok": True, "run_id": run_id}


@router.get("/api/optimize/status")
async def optimize_status(run_id: str = Query(...)):
    from app.main import conn  # noqa: WPS433

    # 1) if current worker has it in memory, persist (also updates DB)
    async with RUN_LOCK:
        st = RUNS.get(run_id)
    if st is not None:
        _persist_throttled(st, force=True, conn=conn)

    # 2) try filesystem snapshot (works within the same replica)
    prog_path = _run_dir(run_id) / "progress.json"
    if prog_path.exists():
        try:
            data = json.loads(prog_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"run_id": run_id, "status": "unknown"}
        data["logs"] = _tail_lines(_log_path(run_id), n=200)
        return data

    # 3) try DB snapshot (works across replicas)
    # NOTE: do not take _DB_STATE_LOCK here. If another thread is holding the
    # lock while talking to the DB and the connection is slow, the status
    # endpoint may appear to "hang" and the UI will stay on "loading...".
    try:
        _ensure_opt_run_state_table(conn)
        d = _load_opt_run_state(conn, run_id)
        if d is not None:
            return d
    except Exception:
        pass

    # 4) fallback to memory if present
    if st is None:
        return JSONResponse(status_code=404, content={"detail": "run_id not found"})
    return {
        "run_id": st.run_id,
        "status": st.status,
        "trial": int(st.trials_done),
        "trials": int(st.cfg.get("trials", 0)) if st.cfg else 0,
        "elapsed_s": float(max(0.0, (st.finished_at or time.time()) - st.started_at)),
        "trials_done": int(st.trials_done),
        "best_score": None if st.best_score == float("-inf") else st.best_score,
        "best_trial": int(st.best_trial),
        "best_params": st.best_params,
        "best_metrics": st.best_metrics,
        "cfg": st.cfg,
        "error": st.error,
        "logs": list(st.logs),
    }


@router.post("/api/optimize/clear")
async def optimize_clear(payload: Dict[str, Any]):
    """Clear saved opt_results.

    payload: {"mode": "errors"|"all"}
    """

    from app.main import conn  # noqa: WPS433

    mode = str(payload.get("mode", "errors")).lower().strip()
    _ensure_opt_results_table(conn)

    if _is_postgres():
        try:
            with conn.cursor() as cur:
                if mode == "all":
                    cur.execute("TRUNCATE TABLE opt_results;")
                else:
                    cur.execute("DELETE FROM opt_results WHERE status='error';")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        if mode == "all":
            conn.execute("DELETE FROM opt_results;")
        else:
            conn.execute("DELETE FROM opt_results WHERE status='error';")
        conn.commit()

    return {"ok": True, "mode": mode}
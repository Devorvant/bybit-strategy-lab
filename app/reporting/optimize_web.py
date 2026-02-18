from __future__ import annotations

import asyncio
import collections
import html
import json
import math
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
# Small HTML shell helper
# ------------------------------

def _page_shell(title: str, body_html: str) -> str:
    """Wrap page body into a minimal responsive HTML document."""

    t = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{t}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0b1220; color: #e6e6e6; }}
    a {{ color: inherit; }}
    .topbar {{ padding: 10px 14px; display: flex; gap: 10px; align-items: center; border-bottom: 1px solid #1b2940; position: sticky; top: 0; background: #0b1220; z-index: 10; flex-wrap: wrap; }}
    .navlink {{ display:inline-block; text-decoration:none; background:#0e1830; border:1px solid #1b2940; color:#e6e6e6; padding: 7px 10px; border-radius: 999px; font-weight: 700; }}
    .navlink.active {{ background:#1b5cff33; border-color:#2b6dff; }}
    .wrap {{ padding: 10px 14px 20px 14px; }}
	    /* Grid-like rows used by the optimizer form */
	    .row {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; }}
	    .row > div {{ display:flex; flex-direction:column; gap:6px; }}
	    .row label {{ font-size: 12px; opacity: 0.9; }}
    select, input, button {{ background: #0e1830; color: #e6e6e6; border: 1px solid #1b2940; border-radius: 8px; padding: 7px 10px; outline: none; }}
    button {{ cursor: pointer; }}
    .apply {{ background: #2b6dff; border: 0; color: #fff; font-weight: 700; }}
    .danger {{ background: #fb7185; border: 0; color: #0b1220; font-weight: 800; }}
    .muted {{ opacity: 0.85; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid #15243a; font-size: 12px; }}
    thead th {{ position: sticky; top: 58px; background: #0b1220; z-index: 5; text-align: left; opacity: 0.9; }}
    @media (max-width: 760px) {{
      .topbar {{ gap: 8px; }}
	      .row {{ flex-direction: column; align-items: stretch; }}
      select, input {{ flex: 1 1 140px; }}
      .apply {{ width: 100%; }}
    }}
  
    /* Tooltips */
    .lbl {{ display:flex; align-items:center; gap:6px; }}
    .tip {{ display:inline-block; width:16px; height:16px; border-radius:999px;
           background:#22315c; color:#d7e3ff; font-size:11px; line-height:16px;
           text-align:center; cursor:help; position:relative; flex:0 0 auto; }}
    .tip:hover::after {{
      content: attr(data-tip);
      position:absolute;
      left: 50%;
      transform: translateX(-50%);
      bottom: 140%;
      background: #0e1830;
      border: 1px solid #1b2940;
      color: #e6e6e6;
      padding: 8px 10px;
      border-radius: 10px;
      white-space: pre-line;
      width: min(360px, 80vw);
      z-index: 50;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .tip:hover::before {{
      content: "";
      position:absolute;
      left:50%;
      transform: translateX(-50%);
      bottom: 125%;
      border: 7px solid transparent;
      border-top-color:#1b2940;
      z-index: 51;
    }}

  </style>
</head>
<body>
  <div class=\"topbar\">
    <a class=\"navlink\" href=\"/chart\">📈 Chart</a>
    <a class=\"navlink active\" href=\"/optimize\">🧪 Optimizer</a>
  </div>
  <div class=\"wrap\">{body_html}</div>
</body>
</html>"""


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


# ------------------------------
# Candidate persistence (multi-seed)
# ------------------------------

def _candidates_path(run_id: str) -> Path:
    return _run_dir(run_id) / "candidates.json"


def save_candidates(run_id: str, payload: Dict[str, Any]) -> None:
    # Persist top candidates for the run (stored in /tmp/optimizer_runs/<id>/candidates.json).
    try:
        with _lock_for(run_id):
            _atomic_write_json(_candidates_path(run_id), payload)
    except Exception:
        pass


def load_candidates(run_id: str) -> Dict[str, Any] | None:
    pth = _candidates_path(run_id)
    if not pth.exists():
        return None
    try:
        return json.loads(pth.read_text(encoding='utf-8'))
    except Exception:
        return None

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



def _delete_opt_result(conn, opt_id: int) -> int:
    """Delete a single saved optimization result by id.

    Returns number of deleted rows (0/1).
    """
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM opt_results WHERE id=%s", (int(opt_id),))
                n = cur.rowcount or 0
            conn.commit()
            return int(n)
        except Exception:
            conn.rollback()
            raise
    else:
        cur = conn.execute("DELETE FROM opt_results WHERE id=?", (int(opt_id),))
        conn.commit()
        return int(cur.rowcount or 0)

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



def _trade_metrics(trades: list, ts0_ms: int, ts1_ms: int, base: float) -> Dict[str, float]:
    """Extra metrics based on closed trades.

    pf: profit factor = sum(profits) / abs(sum(losses))
    avg_trade_pct: avg pnl per trade, normalized by base (position_usd) to make comparable
    tpd: trades per day over the evaluated window
    """
    n = len(trades) if trades else 0
    if n <= 0:
        return {"pf": 0.0, "avg_trade_pct": 0.0, "tpd": 0.0}

    prof = 0.0
    loss = 0.0
    pnl_sum = 0.0
    for t in trades:
        p = float(getattr(t, "pnl", 0.0))
        pnl_sum += p
        if p >= 0:
            prof += p
        else:
            loss += -p

    pf = (prof / loss) if loss > 0 else (999.0 if prof > 0 else 0.0)
    avg_trade = pnl_sum / n
    avg_trade_pct = (avg_trade / base) if base else 0.0

    dur_days = max(1e-9, (float(ts1_ms) - float(ts0_ms)) / 86400000.0)
    tpd = float(n) / dur_days

    return {"pf": float(pf), "avg_trade_pct": float(avg_trade_pct), "tpd": float(tpd)}

def _score(m: Dict[str, float], trades: int, weights: Dict[str, float], min_trades: int) -> float:
    # Hard constraint: ignore configs with too few trades
    if trades < min_trades:
        return float("-inf")

    # Classic weighted score (kept backward compatible)
    s = 0.0
    s += weights.get("w_ret", 2.0) * m.get("ret", 0.0)
    s += weights.get("w_sharpe", 0.5) * m.get("sharpe", 0.0)
    s -= weights.get("w_dd", 1.5) * m.get("dd", 0.0)
    s -= weights.get("w_vol", 0.1) * m.get("vol", 0.0)
    s -= weights.get("w_time_dd", 4.0) * m.get("time_in_dd", 0.0)
    s -= weights.get("w_ulcer", 2.0) * m.get("ulcer", 0.0)

    # Optional extras (work only if metrics were provided)
    w_pf = weights.get("w_pf", 0.0)
    if w_pf:
        pf = max(1e-6, m.get("pf", 0.0))
        s += float(w_pf) * math.log(pf)

    w_avg = weights.get("w_avg_trade", 0.0)
    if w_avg:
        s += float(w_avg) * m.get("avg_trade_pct", 0.0)

    w_tpd = weights.get("w_tpd", 0.0)
    if w_tpd:
        target = float(weights.get("tpd_target", 1.0))
        s -= float(w_tpd) * abs(m.get("tpd", 0.0) - target)

    return float(s)


def _score_breakdown(m: Dict[str, float], trades: int, weights: Dict[str, float], min_trades: int) -> Dict[str, Any]:
    """Explain score as a sum of weighted components.

    This helps verify that weights really affect the optimizer.

    Output example:
      {
        "ok": True,
        "total": 1.23,
        "components": {
          "ret": {"w": 2.0, "x": 0.12, "c": 0.24},
          "dd": {"w": 1.5, "x": 0.10, "c": -0.15},
          ...
        }
      }
    """

    out: Dict[str, Any] = {"ok": True, "total": float("-inf"), "components": {}, "trades": int(trades), "min_trades": int(min_trades)}
    if trades < min_trades:
        out["ok"] = False
        return out

    def _comp(name: str, w: float, x: float, sign: float = 1.0) -> None:
        # sign = +1 means + w*x ; sign = -1 means - w*x
        c = float(sign) * float(w) * float(x)
        out["components"][name] = {"w": float(w), "x": float(x), "c": float(c)}

    _comp("ret", weights.get("w_ret", 2.0), m.get("ret", 0.0), +1.0)
    _comp("sharpe", weights.get("w_sharpe", 0.5), m.get("sharpe", 0.0), +1.0)
    _comp("dd", weights.get("w_dd", 1.5), m.get("dd", 0.0), -1.0)
    _comp("vol", weights.get("w_vol", 0.1), m.get("vol", 0.0), -1.0)
    _comp("time_in_dd", weights.get("w_time_dd", 4.0), m.get("time_in_dd", 0.0), -1.0)
    _comp("ulcer", weights.get("w_ulcer", 2.0), m.get("ulcer", 0.0), -1.0)

    # Optional extras
    w_pf = float(weights.get("w_pf", 0.0) or 0.0)
    if w_pf:
        pf = max(1e-6, float(m.get("pf", 0.0)))
        # in _score we do + w_pf * log(pf)
        out["components"]["pf_log"] = {"w": float(w_pf), "x": float(math.log(pf)), "c": float(w_pf * math.log(pf))}

    w_avg = float(weights.get("w_avg_trade", 0.0) or 0.0)
    if w_avg:
        x = float(m.get("avg_trade_pct", 0.0))
        out["components"]["avg_trade_pct"] = {"w": float(w_avg), "x": float(x), "c": float(w_avg * x)}

    w_tpd = float(weights.get("w_tpd", 0.0) or 0.0)
    if w_tpd:
        target = float(weights.get("tpd_target", 1.0))
        x = abs(float(m.get("tpd", 0.0)) - target)
        out["components"]["tpd_penalty"] = {"w": float(w_tpd), "x": float(x), "c": float(-w_tpd * x)}

    out["total"] = float(sum(float(v.get("c", 0.0)) for v in out["components"].values()))
    return out

def _train_val_split(bars: List[Tuple[int, float, float, float, float, float]], train_frac: float):
    n = len(bars)
    k = max(50, int(n * train_frac))
    k = min(k, n - 20)
    return bars[:k], bars[k:]


def _sample_params(rng, position_usd: float) -> Dict[str, Any]:
    """Sample a candidate parameter set.

    We keep core indicator lengths fixed for stability, but allow a few boolean
    regime switches to vary. This increases expressiveness without exploding
    the search space too much.
    """
    use_flip_limit = rng.choice([False, True])
    return {
        "position_usd": float(position_usd),
        # was fixed True before; now we let optimizer choose
        "use_no_trade": rng.choice([True, False]),
        "adx_len": 14,
        "adx_smooth": 14,
        "adx_no_trade_below": rng.uniform(10.0, 25.0),
        "st_atr_len": 14,
        "st_factor": rng.uniform(2.0, 6.0),
        "use_rev_cooldown": True,
        "rev_cooldown_hrs": rng.choice([0, 2, 4, 6, 8, 12, 16]),
        # new: allow limiting flips/day
        "use_flip_limit": bool(use_flip_limit),
        "max_flips_per_day": rng.choice([2, 3, 4, 5, 6, 8, 10, 12]) if use_flip_limit else 6,
        "use_emergency_sl": True,
        "atr_len": 14,
        "atr_mult": rng.uniform(1.5, 5.0),
        "close_at_end": False,
    }



def _run_optimizer_sync(run: RunState, bars: List[Tuple[int, float, float, float, float, float]], conn) -> None:
    import random

    cfg = run.cfg
    cfg.setdefault("select_on", "train")
    symbol = cfg["symbol"]
    tf = cfg["tf"]
    trials = int(cfg["trials"])
    max_seconds = int(cfg["max_seconds"])
    patience = int(cfg["patience"])
    train_frac = float(cfg.get("train_frac", 0.7))
    seed = int(cfg.get("seed", 42))
    position_usd = float(cfg.get("position_usd", 1000.0))
    min_trades = int(cfg.get("min_trades", 5))
    optimizer = str(cfg.get("optimizer", "random")).lower().strip() or "random"

    # Optional: include execution costs inside the optimization loop.
    # If disabled, optimizer sees a "frictionless" backtest and costs are only
    # applied later in /chart.
    use_costs_in_opt = bool(cfg.get("use_costs_in_opt", False))
    fixed_costs: Dict[str, Any] = {
        "fee_percent": float(cfg.get("fee_percent", 0.0) or 0.0),
        "spread_ticks": float(cfg.get("spread_ticks", 0.0) or 0.0),
        "slippage_ticks": int(float(cfg.get("slippage_ticks", 0.0) or 0.0)),
        "tick_size": float(cfg.get("tick_size", 0.0001) or 0.0001),
        "funding_8h_percent": float(cfg.get("funding_8h_percent", 0.0) or 0.0),
    }

    weights = {
        "w_ret": float(cfg.get("w_ret", 2.0)),
        "w_dd": float(cfg.get("w_dd", 1.5)),
        "w_vol": float(cfg.get("w_vol", 0.1)),
        "w_sharpe": float(cfg.get("w_sharpe", 0.5)),
        "w_time_dd": float(cfg.get("w_time_dd", 4.0)),
        "w_ulcer": float(cfg.get("w_ulcer", 2.0)),
        # extended / optional weights
        "w_pf": float(cfg.get("w_pf", 0.0)),
        "w_avg_trade": float(cfg.get("w_avg_trade", 0.0)),
        "w_tpd": float(cfg.get("w_tpd", 0.0)),
        "tpd_target": float(cfg.get("tpd_target", 1.0)),
    }

    rng = random.Random(seed)

    train_bars, va_bars = _train_val_split(bars, train_frac)
    best_train_score = float("-inf")
    best_sel_score = float("-inf")
    t0 = time.time()
    no_improve = 0

    run.status = "running"
    _add_log(
        run,
        f"RUN_START run_id={run.run_id} optimizer={optimizer} symbol={symbol} tf={tf} bars={len(bars)} trials={trials}",
    )
    _add_log(
        run,
        "RUN_CFG "
        f"seed={seed} select_on={cfg.get('select_on','train')} optimizer={optimizer} "
        f"use_costs_in_opt={use_costs_in_opt} fixed_costs={json.dumps(fixed_costs, ensure_ascii=True)} "
        f"weights={json.dumps(weights, ensure_ascii=True)} min_trades={min_trades}",
    )
    _persist_throttled(run, force=True, conn=conn)

    def _maybe_update_best(params: Dict[str, Any], trial_no: int) -> float:
        nonlocal best_train_score, best_sel_score, no_improve

        params_eval = dict(params)
        if use_costs_in_opt:
            params_eval.update(fixed_costs)

        bt_tr = backtest_strategy3(train_bars, **params_eval)
        m_tr = _equity_metrics(bt_tr.equity, base=position_usd)
        m_tr.update(_trade_metrics(bt_tr.trades, train_bars[0][0], train_bars[-1][0], base=position_usd))
        trades_tr = len(bt_tr.trades)
        s_tr = _score(m_tr, trades_tr, weights, min_trades)
        bd_tr = _score_breakdown(m_tr, trades_tr, weights, min_trades)

        # Always compute validation so we can select_on='val' and also report it.
        bt_va = backtest_strategy3(va_bars, **params_eval)
        m_va = _equity_metrics(bt_va.equity, base=position_usd)
        m_va.update(_trade_metrics(bt_va.trades, va_bars[0][0], va_bars[-1][0], base=position_usd))
        trades_va = len(bt_va.trades)
        s_va = _score(m_va, trades_va, weights, min_trades)
        bd_va = _score_breakdown(m_va, trades_va, weights, min_trades)

        select_on = str(cfg.get("select_on", "train")).lower().strip() or "train"
        if select_on not in ("train", "val"):
            select_on = "train"
        sel_score = s_tr if select_on == "train" else s_va

        run.trials_done = int(trial_no)
        _persist_throttled(run, conn=conn)

        improved = sel_score > best_sel_score
        if improved:
            best_sel_score = sel_score
            best_train_score = max(best_train_score, s_tr)
            no_improve = 0

            # Evaluate on full dataset for reporting
            bt_full = backtest_strategy3(bars, **params_eval)
            m_full = _equity_metrics(bt_full.equity, base=position_usd)
            m_full.update(_trade_metrics(bt_full.trades, bars[0][0], bars[-1][0], base=position_usd))

            run.best_score = float(sel_score)
            run.best_trial = int(trial_no)
            # IMPORTANT: save original strategy params (without costs) so that
            # /chart can re-run with its own cost settings. Costs are stored in
            # run.cfg and also reported in logs.
            run.best_params = params
            run.best_metrics = {
                # for convenience keep a "val-like" view on top-level too
                **m_va,
                "trades": int(trades_va),
                "val_score": float(s_va),
                "train_score": float(s_tr),
                "select_on": select_on,
                "use_costs_in_opt": bool(use_costs_in_opt),
                "fixed_costs": fixed_costs if use_costs_in_opt else {},
                "train_score_breakdown": bd_tr,
                "val_score_breakdown": bd_va,
                "sel_score_breakdown": bd_tr if select_on == "train" else bd_va,
                "train_metrics": {**m_tr, "trades": int(trades_tr)},
                "full_metrics": {**m_full, "trades": int(len(bt_full.trades))},
            }

            # Add a clear score decomposition log line so you can see weights
            # affecting the optimizer in real time.
            sel_bd = bd_tr if select_on == "train" else bd_va
            try:
                comps = sel_bd.get("components", {}) if isinstance(sel_bd, dict) else {}
                # Keep log short but informative.
                parts_str = ", ".join(
                    [
                        f"{k}={float(v.get('c',0.0)):.4g}"
                        for k, v in list(comps.items())[:8]
                    ]
                )
                _add_log(run, f"SCORE_BREAKDOWN select_on={select_on} total={sel_score:.6g} parts: {parts_str}")
            except Exception:
                pass

            _add_log(
                run,
                f"[{trial_no}/{trials}] best_sel={sel_score:.6g} (select_on={select_on}) "
                f"VAL ret={m_va.get('ret',0):.4g} dd={m_va.get('dd',0):.4g} sharpe={m_va.get('sharpe',0):.4g} trades={trades_va} "
                f"TRAIN ret={m_tr.get('ret',0):.4g} dd={m_tr.get('dd',0):.4g} sharpe={m_tr.get('sharpe',0):.4g} trades={trades_tr}",
            )
            _persist_throttled(run, force=True, conn=conn)
        else:
            no_improve += 1

        return float(s_tr)

    try:
        if optimizer == "optuna_tpe":
            try:
                import optuna  # type: ignore
            except Exception as e:
                raise RuntimeError("Optuna is not installed. Add optuna to requirements.txt") from e

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )

            def objective(trial):
                use_flip_limit = trial.suggest_categorical("use_flip_limit", [False, True])
                params = {
                    "position_usd": float(position_usd),
                    "use_no_trade": trial.suggest_categorical("use_no_trade", [True, False]),
                    "adx_len": 14,
                    "adx_smooth": 14,
                    "adx_no_trade_below": float(trial.suggest_float("adx_no_trade_below", 10.0, 25.0)),
                    "st_atr_len": 14,
                    "st_factor": float(trial.suggest_float("st_factor", 2.0, 6.0)),
                    "use_rev_cooldown": True,
                    "rev_cooldown_hrs": int(trial.suggest_categorical("rev_cooldown_hrs", [0, 2, 4, 6, 8, 12, 16])),
                    "use_flip_limit": bool(use_flip_limit),
                    "max_flips_per_day": int(trial.suggest_categorical("max_flips_per_day", [2, 3, 4, 5, 6, 8, 10, 12])) if use_flip_limit else 6,
                    "use_emergency_sl": True,
                    "atr_len": 14,
                    "atr_mult": float(trial.suggest_float("atr_mult", 1.5, 5.0)),
                    "close_at_end": False,
                }

                trial_no = int(trial.number) + 1

                if max_seconds and (time.time() - t0) >= max_seconds:
                    study.stop()
                    return float("-inf")

                s_tr = _maybe_update_best(params, trial_no)

                if trial_no % max(1, trials // 20) == 0:
                    _add_log(run, f"PROGRESS trial={trial_no} best={run.best_score:.6f} best_trial={run.best_trial}")

                if patience and no_improve >= patience:
                    _add_log(run, f"EARLY_STOP patience={patience} reached at trial={trial_no} best_trial={run.best_trial}")
                    study.stop()

                return float(s_tr)

            study.optimize(objective, n_trials=trials, timeout=float(max_seconds) if max_seconds else None)

        elif optimizer == "random_multiseed":
            # Multi-seed random: run independent random searches for seeds seed_from..seed_to.
            seed_from = int(cfg.get("seed_from", seed))
            seed_to = int(cfg.get("seed_to", seed))
            trials_per_seed = int(cfg.get("trials_per_seed", trials) or trials)
            top_n = int(cfg.get("top_n", 20) or 20)

            if seed_to < seed_from:
                seed_to = seed_from
            trials_per_seed = max(1, int(trials_per_seed))
            top_n = max(1, min(200, int(top_n)))

            total_trials = int(cfg.get("trials", 0) or 0)
            if total_trials <= 0:
                total_trials = (seed_to - seed_from + 1) * trials_per_seed

            global_trial = 0
            best_by_seed = []  # list of best result per seed
            stop_all = False

            # selection mode (train/val)
            select_on = str(cfg.get("select_on", "train")).lower().strip() or "train"
            if select_on not in ("train", "val"):
                select_on = "train"

            for si, seed_i in enumerate(range(seed_from, seed_to + 1), start=1):
                rng_i = random.Random(int(seed_i))
                best_sel_seed = float("-inf")
                best_seed_rec = None
                best_trial_seed = 0
                best_trial_global = 0
                no_improve_seed = 0

                _add_log(run, f"SEED_START seed={seed_i} ({si}/{seed_to-seed_from+1}) trials={trials_per_seed}")

                for t in range(1, trials_per_seed + 1):
                    global_trial += 1
                    run.trials_done = global_trial

                    if max_seconds and (time.time() - t0) >= max_seconds:
                        _add_log(run, f"STOP max_seconds reached at global_trial={global_trial-1}")
                        stop_all = True
                        break

                    params = _sample_params(rng_i, position_usd)
                    params_eval = dict(params)
                    if use_costs_in_opt:
                        params_eval.update(fixed_costs)

                    bt_tr = backtest_strategy3(train_bars, **params_eval)
                    m_tr = _equity_metrics(bt_tr.equity, base=position_usd)
                    m_tr.update(_trade_metrics(bt_tr.trades, train_bars[0][0], train_bars[-1][0], base=position_usd))
                    trades_tr = len(bt_tr.trades)
                    s_tr = _score(m_tr, trades_tr, weights, min_trades)
                    bd_tr = _score_breakdown(m_tr, trades_tr, weights, min_trades)

                    bt_va = backtest_strategy3(va_bars, **params_eval)
                    m_va = _equity_metrics(bt_va.equity, base=position_usd)
                    m_va.update(_trade_metrics(bt_va.trades, va_bars[0][0], va_bars[-1][0], base=position_usd))
                    trades_va = len(bt_va.trades)
                    s_va = _score(m_va, trades_va, weights, min_trades)
                    bd_va = _score_breakdown(m_va, trades_va, weights, min_trades)

                    # Selection score for this trial
                    sel_score = float(s_tr if select_on == "train" else s_va)
                    sel_bd = bd_tr if select_on == "train" else bd_va

                    # Seed-best
                    if sel_score > best_sel_seed:
                        best_sel_seed = sel_score
                        best_trial_seed = t
                        best_trial_global = global_trial
                        best_seed_rec = {
                            "seed": int(seed_i),
                            "trial_in_seed": int(best_trial_seed),
                            "trial_global": int(best_trial_global),
                            "sel_score": float(sel_score),
                            "train_score": float(s_tr),
                            "val_score": float(s_va),
                            "params": dict(params),
                            "train_metrics": dict(m_tr),
                            "val_metrics": dict(m_va),
                            "train_score_breakdown": bd_tr,
                            "val_score_breakdown": bd_va,
                            "sel_score_breakdown": sel_bd,
                        }
                        no_improve_seed = 0
                    else:
                        no_improve_seed += 1

                    # Global-best (across seeds)
                    if sel_score > run.best_score:
                        run.best_score = float(sel_score)
                        run.best_trial = int(global_trial)
                        run.best_params = dict(params)
                        run.best_metrics = {
                            **dict(m_va),
                            "trades": int(trades_va),
                            "val_score": float(s_va),
                            "train_score": float(s_tr),
                            "select_on": str(select_on),
                            "use_costs_in_opt": bool(use_costs_in_opt),
                            "fixed_costs": dict(fixed_costs if use_costs_in_opt else {}),
                            "train_score_breakdown": bd_tr,
                            "val_score_breakdown": bd_va,
                            "sel_score_breakdown": sel_bd,
                            "train_metrics": {**dict(m_tr), "trades": int(trades_tr)},
                        }

                        # Log breakdown + summary
                        try:
                            comps = sel_bd.get("components", {}) if isinstance(sel_bd, dict) else {}
                            parts_str = ", ".join([f"{k}={float(v.get('c',0.0)):.4g}" for k, v in list(comps.items())[:8]])
                            _add_log(run, f"SCORE_BREAKDOWN select_on={select_on} total={sel_score:.6g} parts: {parts_str}")
                        except Exception:
                            pass

                        _add_log(
                            run,
                            f"[{global_trial}/{total_trials}] NEW_GLOBAL_BEST seed={seed_i} t={t}/{trials_per_seed} best_sel={sel_score:.6g} (select_on={select_on}) "
                            f"VAL ret={m_va.get('ret',0):.4g} dd={m_va.get('dd',0):.4g} sharpe={m_va.get('sharpe',0):.4g} trades={trades_va} "
                            f"TRAIN ret={m_tr.get('ret',0):.4g} dd={m_tr.get('dd',0):.4g} sharpe={m_tr.get('sharpe',0):.4g} trades={trades_tr}",
                        )
                        _persist_throttled(run, force=True, conn=conn)

                    # Periodic progress log
                    if global_trial % max(1, total_trials // 20) == 0:
                        _add_log(run, f"PROGRESS global_trial={global_trial} best={run.best_score:.6f} best_trial={run.best_trial}")

                    # Seed early-stop
                    if patience and no_improve_seed >= patience:
                        _add_log(run, f"SEED_EARLY_STOP seed={seed_i} patience={patience} at t={t} best_t={best_trial_seed} best_sel={best_sel_seed:.6g}")
                        break

                if best_seed_rec is not None:
                    best_by_seed.append(best_seed_rec)
                    _add_log(run, f"SEED_DONE seed={seed_i} best_sel={best_sel_seed:.6g} best_trial_seed={best_trial_seed} best_trial_global={best_trial_global}")

                if stop_all:
                    break

            # Save best-by-seed + top-N overall
            best_by_seed_sorted = sorted(best_by_seed, key=lambda r: float(r.get("sel_score", float("-inf"))), reverse=True)
            top_candidates = best_by_seed_sorted[:top_n]
            save_candidates(
                run.run_id,
                {
                    "run_id": run.run_id,
                    "optimizer": str(optimizer),
                    "symbol": str(symbol),
                    "tf": str(tf),
                    "limit_bars": int(cfg.get("limit_bars", 0) or 0),
                    "select_on": str(select_on),
                    "seed_from": int(seed_from),
                    "seed_to": int(seed_to),
                    "trials_per_seed": int(trials_per_seed),
                    "top_n": int(top_n),
                    "use_costs_in_opt": bool(use_costs_in_opt),
                    "fixed_costs": dict(fixed_costs if use_costs_in_opt else {}),
                    "weights": dict(weights),
                    "min_trades": int(min_trades),
                    "best_by_seed": best_by_seed_sorted,
                    "top_candidates": top_candidates,
                },
            )
            _add_log(run, f"CANDIDATES_SAVED seeds={len(best_by_seed_sorted)} top={len(top_candidates)}")

        else:
            # classic random (default)
            for trial in range(1, trials + 1):
                if max_seconds and (time.time() - t0) >= max_seconds:
                    _add_log(run, f"STOP max_seconds reached at trial={trial-1}")
                    break

                params = _sample_params(rng, position_usd)
                _maybe_update_best(params, trial)

                if trial % max(1, trials // 20) == 0:
                    _add_log(run, f"PROGRESS trial={trial} best={run.best_score:.6f} best_trial={run.best_trial}")

                if patience and no_improve >= patience:
                    _add_log(run, f"EARLY_STOP patience={patience} reached at trial={trial} best_trial={run.best_trial}")
                    break

        # FULL metrics for chart comparison
        if run.best_params:
            params_eval = dict(run.best_params)
            if use_costs_in_opt:
                params_eval.update(fixed_costs)
            bt_full = backtest_strategy3(bars, **params_eval)
            m_full = _equity_metrics(bt_full.equity, base=position_usd)
            trades_full = len(bt_full.trades)
            if run.best_metrics is None:
                run.best_metrics = {}
            run.best_metrics["full_metrics"] = {**m_full, "trades": trades_full}

        run.finished_at = time.time()
        run.status = "done"
        _persist_throttled(run, force=True, conn=conn)

        dur = run.finished_at - run.started_at
        _add_log(
            run,
            f"DONE run_id={run.run_id} trials_done={run.trials_done} best_score={run.best_score:.6f} best_trial={run.best_trial} duration_sec={dur:.1f}",
        )
        if run.best_params:
            _add_log(run, "BEST_PARAMS " + json.dumps(run.best_params, ensure_ascii=False))
        if run.best_metrics:
            _add_log(run, "BEST_METRICS " + json.dumps(run.best_metrics, ensure_ascii=False))

        _persist_throttled(run, force=True, conn=conn)

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
            best_score=None if run.best_score == float("-inf") else float(run.best_score),
            best_params=run.best_params,
            best_metrics=run.best_metrics,
        )

    except Exception as e:
        run.finished_at = time.time()
        run.status = "error"
        run.error = str(e)
        _persist_throttled(run, force=True, conn=conn)
        _add_log(run, f"ERROR {type(e).__name__}: {e}")

        dur = (run.finished_at or time.time()) - run.started_at
        try:
            _ensure_opt_results_table(conn)
            _insert_opt_result(
                conn,
                strategy="strategy3",
                symbol=symbol,
                tf=str(tf),
                config=cfg,
                status="error",
                duration_sec=float(dur),
                trials_done=int(run.trials_done),
                best_score=None if run.best_score == float("-inf") else float(run.best_score),
                best_params=run.best_params,
                best_metrics=run.best_metrics,
            )
        except Exception:
            pass


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
                f"<td><button type='button' class='btn ghost' onclick=\"deleteResult({int(r.get('id') or 0)})\" title='Delete'>🗑️</button></td>",
            ]
        ) + "</tr>"


    def _to_dict(x) -> Dict[str, Any]:
        if x is None:
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return {}

    cards_html = ""
    for r in last:
        bm = _to_dict(r.get("best_metrics"))
        bs = r.get("best_score")
        bs_str = "" if bs is None else f"{float(bs):.4f}"
        ret = bm.get("ret")
        dd = bm.get("dd")
        trades = bm.get("trades")
        created = str(r.get("created_at"))
        symbol = str(r.get("symbol"))
        tf = str(r.get("tf"))
        status = str(r.get("status"))
        rid = int(r.get("id")) if r.get("id") is not None else 0

        parts = []
        if bs is not None:
            parts.append(f"score={float(bs):.2f}")
        if isinstance(ret, (int, float)):
            parts.append(f"ret={ret*100:+.1f}%")
        if isinstance(dd, (int, float)):
            parts.append(f"dd={dd*100:.1f}%")
        if isinstance(trades, (int, float)):
            parts.append(f"trades={int(trades)}")

        chart_href = f"/chart?symbol={html.escape(symbol)}&tf={html.escape(tf)}&strategy=my_strategy3.py&opt_id={rid}"
        cards_html += f"""
        <div class='card result-card'>
          <div style='display:flex; justify-content:space-between; gap:10px; align-items:flex-start;'>
            <div>
              <div style='font-weight:800;'>#{rid} {html.escape(created)}</div>
              <div class='muted' style='margin-top:4px;'>{html.escape(symbol)} tf={html.escape(tf)} · <span class='pill'>{html.escape(status)}</span></div>
              <div class='muted' style='margin-top:6px;'>{html.escape(" ".join(parts))}</div>
            </div>
            <div style='display:flex; flex-direction:column; gap:8px; align-items:flex-end;'>
              <a class='btn ghost' href='{chart_href}'>Open chart</a>
              <button type='button' class='btn ghost' onclick="deleteResult({rid})" title='Delete'>🗑️</button>
            </div>
          </div>
        </div>
        """

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
            <label>seed</label>
            <input name='seed' value='42' style='width:90px'/>
          </div>
          <div>
            <label>seed_from</label>
            <input name='seed_from' value='1' style='width:90px'/>
          </div>
          <div>
            <label>seed_to</label>
            <input name='seed_to' value='100' style='width:90px'/>
          </div>
          <div>
            <label>trials_per_seed</label>
            <input name='trials_per_seed' value='500' style='width:110px'/>
          </div>
          <div>
            <label>top_n</label>
            <input name='top_n' value='20' style='width:90px'/>
          </div>
          <div>
            <label><span class='lbl'>select_on<span class='tip' data-tip='Критерий отбора лучшего trial.\ntrain — выбираем лучшее по train (val только для отчёта).\nval — выбираем лучшее по val (меньше переобучение).'>?</span></span></label>
            <select name='select_on' style='width:110px'>
              <option value='train' selected>train</option>
              <option value='val'>val</option>
            </select>
          </div>
          <div>
            <label>train_frac</label>
            <input name='train_frac' value='0.7' style='width:90px'/>
          </div>
          <div>
            <label>position_usd</label>
            <input name='position_usd' value='1000' style='width:110px'/>
          </div>
          <div>
            <button type='submit'>Start optimize</button>
          </div>
          <div>
            <button type='button' id='btnMultiSeed' style='background:#1f5c3b;'>Start multi-seed (Random)</button>
          </div>
          <div>
            <label>optimizer</label>

            <select name='optimizer' style='width:170px'>
              <option value='random' selected>Random (classic)</option>
              <option value='optuna_tpe'>Optuna TPE</option>
            </select>
          </div>

        </div>
        
        <hr style='border:0; border-top:1px solid rgba(255,255,255,0.08); margin:14px 0;'/>
        <div style='font-weight:700; margin-bottom:8px;'>Формула score и веса (classic)</div>
        <div class='muted' style='margin-bottom:10px; line-height:1.35;'>
          score = w_ret·ret + w_sharpe·sharpe − w_dd·dd − w_vol·vol − w_time_dd·time_in_dd − w_ulcer·ulcer
        </div>

        <div class='row' style='flex-wrap:wrap; gap:14px;'>
          <div>
            <label><span class='lbl'>w_ret (доходность)<span class='tip' data-tip='Вес доходности (ret).&#10;ret = (капитал_конец / капитал_старт) − 1.&#10;Больше w_ret → оптимизатор сильнее гонится за прибылью.'>?</span></span></label>
            <input name='w_ret' value='2.0' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_sharpe (Sharpe)<span class='tip' data-tip='Вес «ровности» кривой (Sharpe).&#10;Больше w_sharpe → меньше «американских горок», больше стабильности.&#10;Часто снижает агрессивность.'>?</span></span></label>
            <input name='w_sharpe' value='0.5' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_dd (просадка)<span class='tip' data-tip='Штраф за максимальную просадку (max drawdown).&#10;dd = max падение от пика до дна (в долях).&#10;Больше w_dd → избегает глубоких провалов.'>?</span></span></label>
            <input name='w_dd' value='1.5' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_vol (волат.)<span class='tip' data-tip='Штраф за «дребезг» equity (волатильность кривой капитала).&#10;Больше w_vol → кривая более гладкая.&#10;Слишком большой w_vol может душить трендовые стратегии.'>?</span></span></label>
            <input name='w_vol' value='0.1' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_time_dd (время в DD)<span class='tip' data-tip='Штраф за долю времени в просадке.&#10;time_in_dd = доля баров, когда equity ниже своего исторического максимума.&#10;Больше w_time_dd → меньше «застреваний» ниже пиков.'>?</span></span></label>
            <input name='w_time_dd' value='4.0' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_ulcer (Ulcer)<span class='tip' data-tip='Ulcer Index — учитывает и глубину, и длительность просадок.&#10;Наказывает затяжные откаты и «некрасивую» кривую.&#10;Больше w_ulcer → сильнее предпочтение планомерному росту.'>?</span></span></label>
            <input name='w_ulcer' value='2.0' style='width:110px'/>
          </div>

          <div>
            <label><span class='lbl'>min_trades<span class='tip' data-tip='Минимальное число сделок для допуска результата.&#10;Если сделок меньше — score штрафуется/режется.&#10;Нужно, чтобы оптимизатор не выбирал «случайно удачный» прогон на 1–2 сделках.&#10;5 = хотим хотя бы 5 сделок за тестовый период.'>?</span></span></label>
            <input name='min_trades' value='5' style='width:110px'/>
          </div>
        </div>

        <div style='margin-top:10px; font-weight:700;'>Новые веса (опционально)</div>
        <div class='row' style='flex-wrap:wrap; gap:14px; margin-top:8px;'>
          <div>
            <label><span class='lbl'>w_pf<span class='tip' data-tip='Вес Profit Factor (пока 0 — не влияет).&#10;Profit Factor = gross_profit / gross_loss.&#10;Если включить (w_pf>0), будет поощрять качество сделок, а не только итоговую доходность.'>?</span></span></label>
            <input name='w_pf' value='0' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_avg_trade<span class='tip' data-tip='Вес средней сделки (пока 0 — не влияет).&#10;Полезно, чтобы отсеивать стратегии с тонким edge, который съедят комиссии.'>?</span></span></label>
            <input name='w_avg_trade' value='0' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>w_tpd<span class='tip' data-tip='Контроль сделок в день (пока 0 — не влияет).&#10;Если включить (w_tpd>0), можно штрафовать слишком частую торговлю&#10;или отклонение от целевой частоты (tpd_target).'>?</span></span></label>
            <input name='w_tpd' value='0' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>tpd_target<span class='tip' data-tip='Целевая частота trades/day.&#10;Используется только если w_tpd>0.&#10;1.0 = хотим примерно 1 сделку в день (± отклонение будет штрафоваться).'>?</span></span></label>
            <input name='tpd_target' value='1.0' style='width:110px'/>
          </div>
        </div>

        <div style='margin-top:14px; font-weight:700;'>Комиссии в оптимизации (опционально)</div>
        <div class='muted' style='margin-bottom:8px; line-height:1.35;'>
          Если включить галочку, то эти значения будут прокинуты в backtest_strategy3() на каждом trial.
          Это важно, чтобы «лучшие параметры» выбирались уже с учётом трения (fee/spread/slippage/funding).
        </div>
        <div class='row' style='flex-wrap:wrap; gap:14px;'>
          <div style='flex-direction:row; align-items:center; gap:10px; padding-top:20px;'>
            <input type='checkbox' name='use_costs_in_opt' id='use_costs_in_opt'/>
            <label for='use_costs_in_opt' style='margin:0; font-size:13px;'>use_costs_in_opt</label>
          </div>
          <div>
            <label>fee_percent</label>
            <input name='fee_percent' value='0.0' style='width:110px'/>
          </div>
          <div>
            <label>spread_ticks</label>
            <input name='spread_ticks' value='0.0' style='width:110px'/>
          </div>
          <div>
            <label>slippage_ticks</label>
            <input name='slippage_ticks' value='0' style='width:110px'/>
          </div>
          <div>
            <label>tick_size</label>
            <input name='tick_size' value='0.0001' style='width:110px'/>
          </div>
          <div>
            <label>funding_8h_percent</label>
            <input name='funding_8h_percent' value='0.0' style='width:130px'/>
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
          <th>created</th><th>symbol</th><th>tf</th><th>status</th><th>trials</th><th>best_score</th><th>best_params</th><th>best_metrics</th><th></th>
        </tr></thead>
        <tbody>
          {rows_html if rows_html else "<tr><td colspan='8' class='muted'>No results yet.</td></tr>"}
        </tbody>
      </table>
    </div>

    <div class='results-cards'>
      {cards_html if cards_html else "<div class='card'><div class='muted'>No results yet.</div></div>"}
    </div>

    <script>
      const form = document.getElementById('startForm');

      async function startRun(optimizerOverride=null) {{
        const fd = new FormData(form);
        const payload = Object.fromEntries(fd.entries());
        if (optimizerOverride) {{
          payload.optimizer = optimizerOverride;
          if (optimizerOverride === 'random_multiseed' && payload.trials_per_seed) {{
            payload.trials = payload.trials_per_seed; // compatibility
          }}
        }}
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
      }}

      form.addEventListener('submit', async (e) => {{
        e.preventDefault();
        await startRun(null);
      }});

      const btnMulti = document.getElementById('btnMultiSeed');
      if (btnMulti) {{
        btnMulti.addEventListener('click', async () => {{
          await startRun('random_multiseed');
        }});
      }}

            async function deleteResult(id) {{
        if (!id) return;
        const ok = confirm(`Удалить результат #${{id}}?`);
        if (!ok) return;
        const res = await fetch('/api/optimize/delete', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ id }})
        }});
        const j = await res.json();
        if (!res.ok) {{
          alert(j.detail || 'Failed');
          return;
        }}
        window.location.reload();
      }}

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
    """Render run page without JS.

    Railway/hosting environments may set a strict CSP that blocks inline
    scripts, which would leave the page stuck on "loading...". We therefore
    render the latest snapshot server-side and use meta-refresh while running.
    """

    from app.main import conn  # noqa: WPS433

    run_id_safe = html.escape(run_id)

    # Try /tmp snapshot first (fast on same replica), then DB snapshot.
    prog_path = _run_dir(run_id) / "progress.json"
    data = None
    logs = []
    if prog_path.exists():
        try:
            data = json.loads(prog_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"run_id": run_id, "status": "unknown"}
        logs = _tail_lines(_log_path(run_id), n=200)
    else:
        try:
            _ensure_opt_run_state_table(conn)
            data = _load_opt_run_state(conn, run_id)
            if data is not None:
                logs = data.get("logs", []) or []
        except Exception:
            data = None

    if data is None:
        # Not found: show a minimal page with a hint.
        body = f"""
        <div class="header">
          <div>
            <h1>Optimizer run</h1>
            <div class="muted">Оптимизация Strategy 3 (2h). Итерации показываем в вебе, в Postgres сохраняем только итог.</div>
          </div>
          <div class="row">
            <a class="btn ghost" href="/chart">📈 Chart</a>
          </div>
        </div>

        <div class="card">
          <div class="row" style="justify-content:space-between;">
            <div>
              <div class="muted">run_id:</div>
              <div><span class="pill">{run_id_safe}</span></div>
            </div>
            <div><a class="btn" href="/optimize">← Optimizer</a></div>
          </div>
          <div style="height:8px"></div>
          <div class="muted">run_id не найден. Возможно, оптимизация уже завершилась, а запись о прогрессе была очищена. Проверь список результатов на странице /optimize.</div>
        </div>
        """
        return _page_shell("Optimizer run", body)

    status = str(data.get("status", "unknown"))
    trial = int(data.get("trial") or data.get("trials_done") or 0)
    trials = int(data.get("trials") or 0)
    elapsed = float(data.get("elapsed_s") or 0.0)
    best_score = data.get("best_score")
    best_trial = data.get("best_trial")
    best_params = data.get("best_params")
    best_metrics = data.get("best_metrics")
    err = data.get("error")

    refresh = ""
    if status not in ("done", "error"):
        refresh = '<meta http-equiv="refresh" content="2">'

    best_obj = {
        "best_score": best_score,
        "best_trial": best_trial,
        "best_params": best_params,
        "best_metrics": best_metrics,
    }

    best_pre = html.escape(json.dumps(best_obj, ensure_ascii=False, indent=2))


    # Optional: show saved top candidates (multi-seed random)
    cand = load_candidates(run_id)
    cand_card = ""
    if cand and isinstance(cand, dict):
        top = cand.get('top_candidates') or []
        if isinstance(top, list) and top:
            rows = []
            for i, r in enumerate(top):
                try:
                    seed_i = int(r.get('seed')) if r.get('seed') is not None else ''
                except Exception:
                    seed_i = ''
                sel_score = r.get('sel_score')
                tr_sc = r.get('train_score')
                va_sc = r.get('val_score')
                vm = r.get('val_metrics') or {}
                try:
                    ret = float(vm.get('ret', 0.0))
                    dd = float(vm.get('dd', 0.0))
                    sh = float(vm.get('sharpe', 0.0))
                except Exception:
                    ret, dd, sh = 0.0, 0.0, 0.0
                trades = vm.get('trades', r.get('trades'))

                # Chart link: load candidate params into chart.
                link = f"/chart?symbol={html.escape(str(cand.get('symbol','')))}&tf={html.escape(str(cand.get('tf','')))}&limit={int(cand.get('limit_bars', 5000) or 5000)}&strategy=my_strategy3.py&opt_run_id={html.escape(run_id)}&opt_cand={i}"
                rows.append(
                    "<tr>"
                    f"<td>{i+1}</td>"
                    f"<td>{seed_i}</td>"
                    f"<td>{html.escape(str(sel_score))}</td>"
                    f"<td>{html.escape(str(tr_sc))}</td>"
                    f"<td>{html.escape(str(va_sc))}</td>"
                    f"<td>{ret:.4g}</td>"
                    f"<td>{dd:.4g}</td>"
                    f"<td>{sh:.4g}</td>"
                    f"<td>{html.escape(str(trades))}</td>"
                    f"<td><a class='btn ghost' href='{link}'>📈 Chart</a></td>"
                    "</tr>"
                )

            table = "".join(rows)
            cand_card = f"""
    <div class='card'>
      <h2>Top candidates</h2>
      <div class='muted' style='margin-bottom:8px;'>Multi-seed Random: seed {int(cand.get('seed_from',0))}..{int(cand.get('seed_to',0))}, trials_per_seed={int(cand.get('trials_per_seed',0))}, select_on={html.escape(str(cand.get('select_on','')))}</div>
      <div style='overflow:auto'>
        <table class='table' style='min-width:980px'>
          <thead><tr>
            <th>#</th><th>seed</th><th>sel_score</th><th>train_score</th><th>val_score</th>
            <th>val_ret</th><th>val_dd</th><th>val_sharpe</th><th>val_trades</th><th></th>
          </tr></thead>
          <tbody>{table}</tbody>
        </table>
      </div>
    </div>
            """

    logs_pre = html.escape("\n".join(logs))
    err_html = (f'<div class="muted" style="margin-top:8px;color:#ffb4b4">{html.escape(str(err))}</div>' if err else "")

    body = f"""
    {refresh}
    <div class="header">
      <div>
        <h1>Optimizer run</h1>
        <div class="muted">Оптимизация Strategy 3 (2h). Итерации показываем в вебе, в Postgres сохраняем только итог.</div>
      </div>
      <div class="row">
        <a class="btn ghost" href="/chart">📈 Chart</a>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div>
          <div class="muted">run_id:</div>
          <div><span class="pill">{run_id_safe}</span></div>
        </div>
        <div><a class="btn" href="/optimize">← Optimizer</a></div>
      </div>
      <div style="height:8px"></div>
      <div class="muted">status={html.escape(status)} trial={trial}/{trials} elapsed={elapsed:.1f}s</div>
      {err_html}
    </div>

    <div class="card">
      <h2>Best</h2>
      <pre>{best_pre}</pre>
    </div>

    {cand_card}

    <div class="card">
      <h2>Progress log</h2>
      <pre>{logs_pre}</pre>
    </div>
    """

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
      symbol, tf, limit_bars, trials, max_seconds, patience, position_usd, optimizer
    """

    from app.main import conn  # noqa: WPS433

    symbol = str(payload.get("symbol", "APTUSDT")).upper()
    tf = str(payload.get("tf", "120"))
    limit_bars = int(payload.get("limit_bars", 5000))
    trials = int(payload.get("trials", 2000))
    max_seconds = int(payload.get("max_seconds", 900))
    patience = int(payload.get("patience", 300))
    position_usd = float(payload.get("position_usd", 1000))

    optimizer = str(payload.get("optimizer", "random")).strip() or "random"

    # extra controls (optional)
    try:
        seed = int(payload.get("seed", 42) or 42)
    except Exception:
        seed = 42


    # Multi-seed random (optional)
    def _i(key: str, default: int) -> int:
        try:
            v = payload.get(key, default)
            if v is None or v == "":
                return int(default)
            return int(float(v))
        except Exception:
            return int(default)

    seed_from = _i("seed_from", seed)
    seed_to = _i("seed_to", seed)
    trials_per_seed = _i("trials_per_seed", trials)
    top_n = _i("top_n", 20)

    # normalize
    if seed_to < seed_from:
        seed_to = seed_from
    trials_per_seed = max(1, min(200000, int(trials_per_seed)))
    top_n = max(1, min(200, int(top_n)))

    # If multi-seed is selected, total trials shown in UI = seeds * trials_per_seed
    if str(payload.get("optimizer", "")).strip().lower() == "random_multiseed":
        seeds_count = (seed_to - seed_from + 1)
        # safety cap to avoid accidental huge jobs via web
        seeds_count = max(1, min(500, seeds_count))
        seed_to = seed_from + seeds_count - 1
        trials = seeds_count * trials_per_seed

    select_on = str(payload.get("select_on", "train")).lower().strip() or "train"
    if select_on not in ("train", "val"):
        select_on = "train"

    try:
        train_frac = float(payload.get("train_frac", 0.7) or 0.7)
    except Exception:
        train_frac = 0.7
    train_frac = min(0.9, max(0.5, train_frac))

    def _f(key: str, default: float) -> float:
        try:
            v = payload.get(key, default)
            if v is None or v == "":
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def _b(key: str) -> bool:
        v = payload.get(key)
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "on", "y")


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
        "optimizer": optimizer,
        "train_frac": train_frac,
        "seed": seed,
        "seed_from": seed_from,
        "seed_to": seed_to,
        "trials_per_seed": trials_per_seed,
        "top_n": top_n,
        "select_on": select_on,
        # costs inside optimization (optional)
        "use_costs_in_opt": _b("use_costs_in_opt"),
        "fee_percent": _f("fee_percent", 0.0),
        "spread_ticks": _f("spread_ticks", 0.0),
        "slippage_ticks": _f("slippage_ticks", 0.0),
        "tick_size": _f("tick_size", 0.0001),
        "funding_8h_percent": _f("funding_8h_percent", 0.0),
                "min_trades": int(payload.get("min_trades", 5) or 5),
                # weights (classic)
                "w_ret": _f("w_ret", 2.0),
                "w_dd": _f("w_dd", 1.5),
                "w_vol": _f("w_vol", 0.1),
                "w_sharpe": _f("w_sharpe", 0.5),
                "w_time_dd": _f("w_time_dd", 4.0),
                "w_ulcer": _f("w_ulcer", 2.0),
                # new weights (currently unused in score; keep 0)
                "w_pf": _f("w_pf", 0.0),
                "w_avg_trade": _f("w_avg_trade", 0.0),
                "w_tpd": _f("w_tpd", 0.0),
                "tpd_target": _f("tpd_target", 1.0),
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


@router.post("/api/optimize/delete")
async def optimize_delete(payload: Dict[str, Any]):
    """Delete a single saved optimization result.

    payload: {"id": <int>}
    """

    from app.main import conn  # noqa: WPS433

    opt_id = payload.get("id")
    try:
        opt_id_int = int(opt_id)
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid id"})

    _ensure_opt_results_table(conn)
    n = _delete_opt_result(conn, opt_id_int)
    return {"ok": True, "deleted": n, "id": opt_id_int}

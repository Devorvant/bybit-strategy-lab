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
from app.storage.db import load_bars, load_bars_before, bars_min_max_ts
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
    /* Cards / block spacing */
    .card {{ border: 1px solid #1b2940; background: #0e1830; border-radius: 14px; padding: 14px; margin: 16px 0; }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 15px; }}
    .result-card {{ padding: 10px 12px; }}
    .section-title {{ font-weight: 700; margin: 0 0 8px 0; }}
    .divider {{ height: 1px; background: #1b2940; margin: 12px 0; }}
    .startbtn {{ background: #1f9d6a; border: 1px solid #1f9d6a; color: #04110c; font-weight: 700; padding: 10px 14px; border-radius: 12px; cursor: pointer; }}
    .startbtn:hover {{ filter: brightness(1.05); }}
    .startbtn.secondary {{ background: transparent; color: #cfe7ff; border: 1px solid #2a3d66; }}
    .startbtn.secondary:hover {{ background: #0b1220; }}


  
      .shiftBtns {{ display:flex; gap:6px; margin-top:6px; flex-wrap:wrap; }}
      .btnMini {{ padding:4px 8px; border-radius:999px; border:1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.9); cursor:pointer; font-size:12px; }}
      .btnMini:hover {{ background: rgba(255,255,255,0.10); }}
    
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

def _json_sanitize(obj: Any) -> Any:
    """Make obj JSON-safe: replace NaN/±Inf with None recursively."""
    if obj is None:
        return None
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    # Fallback: stringify unknown objects (e.g., numpy types)
    try:
        if isinstance(obj, (np.floating, np.integer)):  # type: ignore[name-defined]
            v = float(obj)
            return None if not math.isfinite(v) else v
    except Exception:
        pass
    return str(obj)

def _json_dumps_safe(obj: Any, **kwargs: Any) -> str:
    """json.dumps with sanitization and strict JSON compliance."""
    return json.dumps(_json_sanitize(obj), allow_nan=False, **kwargs)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json_dumps_safe(data, ensure_ascii=False), encoding="utf-8")
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
                        _json_dumps_safe(config),
                        status,
                        float(duration_sec),
                        int(trials_done),
                        float(best_score) if best_score is not None else None,
                        _json_dumps_safe(best_params) if best_params else None,
                        _json_dumps_safe(best_metrics) if best_metrics else None,
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
                _json_dumps_safe(config),
                status,
                float(duration_sec),
                int(trials_done),
                float(best_score) if best_score is not None else None,
                _json_dumps_safe(best_params) if best_params else None,
                _json_dumps_safe(best_metrics) if best_metrics else None,
            ),
        )
        conn.commit()


def _load_last_results(conn, limit: int = 30) -> List[Dict[str, Any]]:
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, strategy, symbol, tf, status, duration_sec, trials_done, best_score, best_params, best_metrics, config
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
            SELECT id, created_at, strategy, symbol, tf, status, duration_sec, trials_done, best_score, best_params, best_metrics, config
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
                "config": r[11],
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
                    (run_id, _json_dumps_safe(payload), logs_txt),
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
            (run_id, _json_dumps_safe(payload), logs_txt),
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


def _tf_seconds(tf: str) -> int:
    """Convert tf (minutes as string) to seconds per bar."""
    try:
        m = int(float(str(tf).strip()))
        return max(60, m * 60)
    except Exception:
        return 3600


def _robust_evaluate(
    *,
    bars: List[Tuple[int, float, float, float, float, float]],
    params: Dict[str, Any],
    tf: str,
    position_usd: float,
    train_frac: float,
    weights: Dict[str, float],
    min_trades: int,
    use_costs_in_opt: bool,
    fixed_costs: Dict[str, Any],
    windows: int,
    window_bars: int,
    shift_hours: float,
    first_end_hours_ago: float,
) -> Dict[str, Any]:
    """Walk-forward robustness evaluation.

    We create N windows ending at: now, now-shift, now-2*shift, ...
    Each window has length `window_bars` (bars). For each window we run a
    train/val split and compute scores (typically we care about val_score).
    """

    out: Dict[str, Any] = {
        "cfg": {
            "windows": int(windows),
            "window_bars": int(window_bars),
            "shift_hours": float(shift_hours),
            "first_end_hours_ago": float(first_end_hours_ago),
            "train_frac": float(train_frac),
            "score_on": "val",
        },
        "summary": {},
        "windows": [],
    }

    n = len(bars)
    windows = max(1, int(windows))
    window_bars = max(200, int(window_bars))

    tf_sec = _tf_seconds(tf)
    # Determine shift in bars.
    if first_end_hours_ago and windows > 1:
        total_shift_hours = max(0.0, float(first_end_hours_ago))
        step_hours = total_shift_hours / float(windows - 1)
    else:
        step_hours = max(0.0, float(shift_hours))

    shift_bars = int(round((step_hours * 3600.0) / float(tf_sec))) if step_hours > 0 else 0
    shift_bars = max(1, shift_bars) if windows > 1 else 0

    # Evaluate
    scores_val: List[float] = []
    ok_windows = 0

    params_eval = dict(params)

    params_eval["confirm_on_close"] = bool(params.get("confirm_on_close", False))
    if use_costs_in_opt:
        params_eval.update(fixed_costs)

    for i in range(windows):
        end_idx = n - i * shift_bars if shift_bars else n
        start_idx = end_idx - window_bars
        if start_idx < 0:
            break  # not enough history for further windows
        wbars = bars[start_idx:end_idx]
        if len(wbars) < 220:
            break

        tr_bars, va_bars = _train_val_split(wbars, train_frac)

        bt_tr = backtest_strategy3(tr_bars, **params_eval)
        m_tr = _equity_metrics(bt_tr.equity, base=position_usd)
        m_tr.update(_trade_metrics(bt_tr.trades, tr_bars[0][0], tr_bars[-1][0], base=position_usd))
        trades_tr = len(bt_tr.trades)
        s_tr = _score(m_tr, trades_tr, weights, min_trades)

        bt_va = backtest_strategy3(va_bars, **params_eval)
        m_va = _equity_metrics(bt_va.equity, base=position_usd)
        m_va.update(_trade_metrics(bt_va.trades, va_bars[0][0], va_bars[-1][0], base=position_usd))
        trades_va = len(bt_va.trades)
        s_va = _score(m_va, trades_va, weights, min_trades)

        win = {
            "i": int(i),
            "end_shift_hours": float(i * step_hours),
            "ts_min": int(wbars[0][0]),
            "ts_max": int(wbars[-1][0]),
            "bars": int(len(wbars)),
            "train_score": float(s_tr),
            "val_score": float(s_va),
            "train_metrics": m_tr,
            "val_metrics": m_va,
            "train_trades": int(trades_tr),
            "val_trades": int(trades_va),
        }
        out["windows"].append(win)

        # For summary use val_score only if window is valid (enough trades)
        if trades_va >= min_trades:
            ok_windows += 1
            scores_val.append(float(s_va))

    if scores_val:
        scores_sorted = sorted(scores_val)
        mean = sum(scores_val) / float(len(scores_val))
        median = scores_sorted[len(scores_sorted) // 2]
        out["summary"] = {
            "windows_done": int(len(out["windows"])),
            "ok_windows": int(ok_windows),
            "mean_val_score": float(mean),
            "median_val_score": float(median),
            "min_val_score": float(min(scores_val)),
            "max_val_score": float(max(scores_val)),
        }
    else:
        out["summary"] = {"windows_done": int(len(out["windows"])), "ok_windows": int(ok_windows)}

    return out

def _sample_params(
    rng,
    position_usd: float,
    fixed_params: Dict[str, Any],
    opt_flags: Dict[str, bool],
) -> Dict[str, Any]:
    """Sample a candidate parameter set with per-parameter opt toggles."""

    def _opt(key: str) -> bool:
        return bool(opt_flags.get(key, False))

    def _fx(key: str, default: Any) -> Any:
        return fixed_params.get(key, default)

    # switches
    use_no_trade = rng.choice([True, False]) if _opt("use_no_trade") else bool(_fx("use_no_trade", True))
    use_rev_cooldown = rng.choice([True, False]) if _opt("use_rev_cooldown") else bool(_fx("use_rev_cooldown", True))
    use_emergency_sl = rng.choice([True, False]) if _opt("use_emergency_sl") else bool(_fx("use_emergency_sl", True))
    use_flip_limit = rng.choice([False, True]) if _opt("use_flip_limit") else bool(_fx("use_flip_limit", True))

    # lengths
    adx_len = int(rng.randint(7, 40)) if _opt("adx_len") else int(_fx("adx_len", 14))
    adx_smooth = int(rng.randint(7, 40)) if _opt("adx_smooth") else int(_fx("adx_smooth", 14))
    st_atr_len = int(rng.randint(7, 80)) if _opt("st_atr_len") else int(_fx("st_atr_len", 14))
    atr_len = int(rng.randint(7, 120)) if _opt("atr_len") else int(_fx("atr_len", 14))

    # floats (wider bounds to allow ADX~30 and atr_mult>5)
    adx_no_trade_below = float(rng.uniform(10.0, 25.0)) if _opt("adx_no_trade_below") else float(_fx("adx_no_trade_below", 24.0))
    st_factor = float(rng.uniform(2.0, 6.0)) if _opt("st_factor") else float(_fx("st_factor", 4.0))
    atr_mult = float(rng.uniform(1.5, 5.0)) if _opt("atr_mult") else float(_fx("atr_mult", 4.0))

    # discrete ints
    rev_cooldown_hrs = int(rng.choice([0, 2, 4, 6, 8, 12, 16])) if _opt("rev_cooldown_hrs") else int(_fx("rev_cooldown_hrs", 12))

    if use_flip_limit:
        max_flips_per_day = int(rng.choice([2, 3, 4, 5, 6, 8, 10, 12])) if _opt("max_flips_per_day") else int(_fx("max_flips_per_day", 6))
    else:
        max_flips_per_day = int(_fx("max_flips_per_day", 6))

    close_at_end = rng.choice([False, True]) if _opt("close_at_end") else bool(_fx("close_at_end", False))

    return {
        "position_usd": float(position_usd),
        "use_no_trade": bool(use_no_trade),
        "adx_len": int(adx_len),
        "adx_smooth": int(adx_smooth),
        "adx_no_trade_below": float(adx_no_trade_below),
        "st_atr_len": int(st_atr_len),
        "st_factor": float(st_factor),
        "use_rev_cooldown": bool(use_rev_cooldown),
        "rev_cooldown_hrs": int(rev_cooldown_hrs),
        "use_flip_limit": bool(use_flip_limit),
        "max_flips_per_day": int(max_flips_per_day),
        "use_emergency_sl": bool(use_emergency_sl),
        "atr_len": int(atr_len),
        "atr_mult": float(atr_mult),
        "close_at_end": bool(close_at_end),
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
    # train/val split ratio
    try:
        train_frac = float((cfg.get("train_frac") if cfg else None) or 0.7)
    except Exception:
        train_frac = 0.7
    seed = int(cfg.get("seed", 42))
    try:
        position_usd = float((cfg.get("position_usd") if cfg else None) or 1000.0)
    except Exception:
        position_usd = 1000.0
    fixed_params = cfg.get("fixed_params") or {}
    opt_flags = cfg.get("opt_flags") or {}
    min_trades = int(cfg.get("min_trades", 5))
    optimizer = str(cfg.get("optimizer", "random")).lower().strip() or "random"

    # Realism option: signal confirmed on close (t-1) -> execute on next open (t)
    confirm_on_close = bool(cfg.get("confirm_on_close", False))

    # Optional: include execution costs inside the optimization loop.
    # If disabled, optimizer sees a "frictionless" backtest and costs are only
    # applied later in /chart.
    use_costs_in_opt = bool(cfg.get("use_costs_in_opt", False))
    fixed_costs: Dict[str, Any] = {
        "fee_percent": float(cfg.get("fee_percent", 0.06) or 0.06),
        "spread_ticks": float(cfg.get("spread_ticks", 1.0) or 1.0),
        "slippage_ticks": int(float(cfg.get("slippage_ticks", 2) or 2)),
        "tick_size": float(cfg.get("tick_size", 0.0001) or 0.0001),
        "funding_8h_percent": float(cfg.get("funding_8h_percent", 0.01) or 0.01),
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
        f"RUN_START run_id={run.run_id} optimizer={optimizer} symbol={symbol} tf={tf} bars={len(bars)} trials={trials} end_shift_hours={cfg.get('end_shift_hours',0)} end_ts={cfg.get('end_ts')}",
    )
    _add_log(
        run,
        "RUN_CFG "
        f"seed={seed} select_on={cfg.get('select_on','train')} optimizer={optimizer} end_shift_hours={cfg.get('end_shift_hours',0)} "
        f"use_costs_in_opt={use_costs_in_opt} fixed_costs={json.dumps(fixed_costs, ensure_ascii=True)} "
        f"weights={json.dumps(weights, ensure_ascii=True)} min_trades={min_trades}",
    )
    _persist_throttled(run, force=True, conn=conn)

    def _maybe_update_best(params: Dict[str, Any], trial_no: int) -> float:
        nonlocal best_train_score, best_sel_score, no_improve

        params_eval = dict(params)

        params_eval["confirm_on_close"] = bool(params.get("confirm_on_close", False))
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
        if select_on not in ("train", "val", "mix"):
            select_on = "train"
        if select_on == "train":
            sel_score = s_tr
        elif select_on == "val":
            sel_score = s_va
        else:
            # mix: blended val/train
            alpha = (cfg.get("mix_alpha", 0.7) if cfg else 0.7) or 0.7
            try:
                alpha = float(alpha)
            except Exception:
                alpha = 0.7
            alpha = min(1.0, max(0.0, alpha))
            sel_score = alpha * s_va + (1.0 - alpha) * s_tr

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
            # Save strategy params (without costs). Include confirm_on_close so /chart re-runs identically.
            run.best_params = dict(params)
            run.best_params["confirm_on_close"] = confirm_on_close
            run.best_metrics = {
                # for convenience keep a "val-like" view on top-level too
                **m_va,
                "trades": int(trades_va),
                "val_score": float(s_va),
                "train_score": float(s_tr),
                "select_on": select_on,
                # Execution mode (signal on close -> fill on next open)
                "confirm_on_close": bool(confirm_on_close),
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

            def _params_from_trial(trial, position_usd: float, fixed_params: Dict[str, Any], opt_flags: Dict[str, bool]) -> Dict[str, Any]:
                def _opt(key: str) -> bool:
                    return bool(opt_flags.get(key, False))

                def _fx(key: str, default: Any) -> Any:
                    return fixed_params.get(key, default)

                use_no_trade = trial.suggest_categorical("use_no_trade", [True, False]) if _opt("use_no_trade") else bool(_fx("use_no_trade", True))
                use_rev_cooldown = trial.suggest_categorical("use_rev_cooldown", [True, False]) if _opt("use_rev_cooldown") else bool(_fx("use_rev_cooldown", True))
                use_emergency_sl = trial.suggest_categorical("use_emergency_sl", [True, False]) if _opt("use_emergency_sl") else bool(_fx("use_emergency_sl", True))

                use_flip_limit = trial.suggest_categorical("use_flip_limit", [False, True]) if _opt("use_flip_limit") else bool(_fx("use_flip_limit", True))

                adx_len = int(trial.suggest_int("adx_len", 7, 40)) if _opt("adx_len") else int(_fx("adx_len", 14))
                adx_smooth = int(trial.suggest_int("adx_smooth", 7, 40)) if _opt("adx_smooth") else int(_fx("adx_smooth", 14))
                st_atr_len = int(trial.suggest_int("st_atr_len", 7, 80)) if _opt("st_atr_len") else int(_fx("st_atr_len", 14))
                atr_len = int(trial.suggest_int("atr_len", 7, 120)) if _opt("atr_len") else int(_fx("atr_len", 14))

                adx_no_trade_below = float(trial.suggest_float("adx_no_trade_below", 10.0, 25.0)) if _opt("adx_no_trade_below") else float(_fx("adx_no_trade_below", 24.0))
                st_factor = float(trial.suggest_float("st_factor", 2.0, 6.0)) if _opt("st_factor") else float(_fx("st_factor", 4.0))
                atr_mult = float(trial.suggest_float("atr_mult", 1.5, 5.0)) if _opt("atr_mult") else float(_fx("atr_mult", 4.0))

                rev_cooldown_hrs = int(trial.suggest_categorical("rev_cooldown_hrs", [0, 2, 4, 6, 8, 12, 16])) if _opt("rev_cooldown_hrs") else int(_fx("rev_cooldown_hrs", 12))

                if use_flip_limit:
                    max_flips_per_day = int(trial.suggest_categorical("max_flips_per_day", [2, 3, 4, 5, 6, 8, 10, 12])) if _opt("max_flips_per_day") else int(_fx("max_flips_per_day", 6))
                else:
                    max_flips_per_day = int(_fx("max_flips_per_day", 6))

                close_at_end = trial.suggest_categorical("close_at_end", [False, True]) if _opt("close_at_end") else bool(_fx("close_at_end", False))

                return {
                    "position_usd": float(position_usd),
                    "use_no_trade": bool(use_no_trade),
                    "adx_len": int(adx_len),
                    "adx_smooth": int(adx_smooth),
                    "adx_no_trade_below": float(adx_no_trade_below),
                    "st_atr_len": int(st_atr_len),
                    "st_factor": float(st_factor),
                    "use_rev_cooldown": bool(use_rev_cooldown),
                    "rev_cooldown_hrs": int(rev_cooldown_hrs),
                    "use_flip_limit": bool(use_flip_limit),
                    "max_flips_per_day": int(max_flips_per_day),
                    "use_emergency_sl": bool(use_emergency_sl),
                    "atr_len": int(atr_len),
                    "atr_mult": float(atr_mult),
                    "close_at_end": bool(close_at_end),
                }


            def objective(trial):
                params = _params_from_trial(trial, position_usd, fixed_params, opt_flags)

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
            if select_on not in ("train", "val", "mix"):
                select_on = "train"

            mode = str(cfg.get("multiseed_mode", "seeds") or "seeds").strip().lower()
            if mode not in ("seeds", "windows"):
                mode = "seeds"

            if mode == "windows":
                # Multi-window random: run independent random searches on multiple shifted windows of data.
                windows_count = int(cfg.get("robust_windows", 10) or 10)
                windows_count = max(1, min(500, windows_count))
                robust_shift_hours = float(cfg.get("robust_shift_hours", 10.0) or 10.0)
                first_end_hours_ago = float(cfg.get("robust_first_end_hours_ago", 0.0) or 0.0)

                # Bars per window.
                # We reuse the walk-forward section as a "window config" for the multi-window search.
                # robust_check may be OFF — we only take these numbers.
                # If robust_bars is set (>0), it overrides limit_bars for the window length.
                limit_bars = int(cfg.get("robust_bars", 0) or 0) or int(cfg.get("limit_bars", 5000) or 5000)

                total_trials = windows_count * trials_per_seed
                global_trial = 0
                stop_all = False

                base_seed = int(cfg.get("seed_from", seed) or seed)

                for wi in range(windows_count):
                    window_shift_h = first_end_hours_ago + float(wi) * robust_shift_hours
                    end_ts_w = int(int(cfg.get("end_ts", 0) or 0) - window_shift_h * 3600.0 * 1000.0)

                    bars_w = load_bars_before(conn, symbol, tf, end_ts=end_ts_w, limit=limit_bars)
                    if len(bars_w) < 300:
                        _add_log(run, f"WINDOW_SKIP idx={wi+1}/{windows_count} shift_h={window_shift_h:.4g} not_enough_bars={len(bars_w)}")
                        continue

                    split_w = int(len(bars_w) * train_frac)
                    split_w = min(len(bars_w) - 20, max(50, split_w))
                    train_bars_w = bars_w[:split_w]
                    va_bars_w = bars_w[split_w:]

                    rng_i = random.Random(int(base_seed + wi))
                    best_sel_w = float("-inf")
                    best_rec = None
                    best_trial_w = 0
                    best_trial_global = 0
                    no_improve_w = 0

                    _add_log(run, f"WINDOW_START idx={wi+1}/{windows_count} shift_h={window_shift_h:.4g} end_ts={end_ts_w} trials={trials_per_seed}")

                    for t in range(1, trials_per_seed + 1):
                        global_trial += 1
                        run.trials_done = global_trial

                        if max_seconds and (time.time() - t0) >= max_seconds:
                            _add_log(run, f"STOP max_seconds reached at global_trial={global_trial-1}")
                            stop_all = True
                            break

                        params = _sample_params(rng_i, position_usd, fixed_params, opt_flags)
                        params_eval = dict(params)
                        params_eval["confirm_on_close"] = bool(params.get("confirm_on_close", False))
                        if use_costs_in_opt:
                            params_eval.update(fixed_costs)

                        bt_tr = backtest_strategy3(train_bars_w, **params_eval)
                        m_tr = _equity_metrics(bt_tr.equity, base=position_usd)
                        m_tr.update(_trade_metrics(bt_tr.trades, train_bars_w[0][0], train_bars_w[-1][0], base=position_usd))
                        trades_tr = len(bt_tr.trades)
                        s_tr = _score(m_tr, trades_tr, weights, min_trades)
                        bd_tr = _score_breakdown(m_tr, trades_tr, weights, min_trades)

                        bt_va = backtest_strategy3(va_bars_w, **params_eval)
                        m_va = _equity_metrics(bt_va.equity, base=position_usd)
                        m_va.update(_trade_metrics(bt_va.trades, va_bars_w[0][0], va_bars_w[-1][0], base=position_usd))
                        trades_va = len(bt_va.trades)
                        s_va = _score(m_va, trades_va, weights, min_trades)
                        bd_va = _score_breakdown(m_va, trades_va, weights, min_trades)

                        sel_score = float(s_tr if select_on == "train" else s_va)
                        sel_bd = bd_tr if select_on == "train" else bd_va

                        if sel_score > best_sel_w:
                            best_sel_w = sel_score
                            best_trial_w = t
                            best_trial_global = global_trial
                            best_rec = {
                                "seed": int(base_seed + wi),  # compatibility field
                                "window_idx": int(wi + 1),
                                "shift_hours": float(window_shift_h),
                                "end_ts": int(end_ts_w),
                                "trial_in_window": int(best_trial_w),
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
                            no_improve_w = 0
                        else:
                            no_improve_w += 1

                        if sel_score > run.best_score:
                            run.best_score = float(sel_score)
                            run.best_trial = int(global_trial)
                            run.best_params = dict(params)
                            run.best_params["confirm_on_close"] = bool(cfg.get("confirm_on_close", False))
                            run.best_metrics = {
                                **dict(m_va),
                                "trades": int(trades_va),
                                "val_score": float(s_va),
                                "train_score": float(s_tr),
                                "select_on": str(select_on),
                                "use_costs_in_opt": bool(use_costs_in_opt),
                                "fixed_costs": dict(fixed_costs if use_costs_in_opt else {}),
                                "confirm_on_close": bool(cfg.get("confirm_on_close", False)),
                                "train_score_breakdown": bd_tr,
                                "val_score_breakdown": bd_va,
                                "sel_score_breakdown": sel_bd,
                                "train_metrics": {**dict(m_tr), "trades": int(trades_tr)},
                                "window_idx": int(wi + 1),
                                "shift_hours": float(window_shift_h),
                            }
                            _add_log(run, f"[{global_trial}/{total_trials}] NEW_GLOBAL_BEST window={wi+1}/{windows_count} t={t}/{trials_per_seed} best_sel={sel_score:.6g} (select_on={select_on}) "
                                          f"VAL ret={m_va.get('ret',0):.4g} dd={m_va.get('dd',0):.4g} sharpe={m_va.get('sharpe',0):.4g} trades={trades_va}")
                            _persist_throttled(run, force=True, conn=conn)

                        if global_trial % max(1, total_trials // 20) == 0:
                            _add_log(run, f"PROGRESS global_trial={global_trial} best={run.best_score:.6f} best_trial={run.best_trial}")

                        if patience and no_improve_w >= patience:
                            _add_log(run, f"WINDOW_EARLY_STOP window={wi+1}/{windows_count} patience={patience} at t={t} best_t={best_trial_w} best_sel={best_sel_w:.6g}")
                            break

                    if best_rec is not None:
                        best_by_seed.append(best_rec)
                        _add_log(run, f"WINDOW_DONE idx={wi+1}/{windows_count} best_sel={best_sel_w:.6g} best_trial={best_trial_w} best_trial_global={best_trial_global}")

                    if stop_all:
                        break

            else:
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

                        params = _sample_params(rng_i, position_usd, fixed_params, opt_flags)
                        params_eval = dict(params)
                        params_eval["confirm_on_close"] = bool(params.get("confirm_on_close", False))
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
                            run.best_params["confirm_on_close"] = confirm_on_close
                            run.best_metrics = {
                                **dict(m_va),
                                "trades": int(trades_va),
                                "val_score": float(s_va),
                                "train_score": float(s_tr),
                                "select_on": str(select_on),
                                "use_costs_in_opt": bool(use_costs_in_opt),
                                "fixed_costs": dict(fixed_costs if use_costs_in_opt else {}),
                                "confirm_on_close": bool(confirm_on_close),
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
                    "multiseed_mode": str(cfg.get("multiseed_mode","seeds") or "seeds"),
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

                params = _sample_params(rng, position_usd, fixed_params, opt_flags)
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

        # Robustness check (walk-forward) — works for any optimizer.
        if bool(cfg.get("robust_check")) and run.best_params:
            try:
                rw = int(cfg.get("robust_windows", 10) or 10)
                rbars = int(cfg.get("robust_bars", int(cfg.get("limit_bars", 5000)) or 5000) or 5000)
                rshift = float(cfg.get("robust_shift_hours", 10.0) or 10.0)
                rfirst = float(cfg.get("robust_first_end_hours_ago", 0.0) or 0.0)
                rtopk = int(cfg.get("robust_top_k", 1) or 1)
                rtopk = max(1, min(50, rtopk))

                _add_log(run, f"ROBUST_START windows={rw} window_bars={rbars} shift_hours={rshift} first_end_hours_ago={rfirst} top_k={rtopk}")

                # Ensure we have enough history to build all robustness windows.
                # IMPORTANT: this must NOT change the optimization window (bars used to find best).
                tf_sec = _tf_seconds(tf)
                if rfirst and rw > 1:
                    step_hours = max(0.0, float(rfirst)) / float(rw - 1)
                else:
                    step_hours = max(0.0, float(rshift))
                shift_bars = int(round((step_hours * 3600.0) / float(tf_sec))) if step_hours > 0 else 0
                shift_bars = max(1, shift_bars) if rw > 1 else 0
                required = int(rbars + shift_bars * max(0, rw - 1))
                required = max(required, rbars)

                end_ts = int(cfg.get('end_ts')) if cfg.get('end_ts') is not None else None
                bars_robust = bars
                if end_ts is not None and required > len(bars):
                    bars_robust = load_bars_before(conn, symbol, tf, end_ts=end_ts, limit=required)
                _add_log(run, f"ROBUST_LOAD opt_bars={len(bars)} robust_bars={len(bars_robust)} required={required} shift_bars={shift_bars} base_end_shift_hours={cfg.get('end_shift_hours',0)}")

                # Build candidate list: always include best, and also include saved top candidates (if any).
                cand_payload = load_candidates(run.run_id) if rtopk > 1 else None
                cand_list: List[Dict[str, Any]] = [{"name": "best", "params": dict(run.best_params)}]

                if cand_payload and isinstance(cand_payload, dict):
                    top = cand_payload.get("top_candidates") or []
                    if isinstance(top, list):
                        for r in top:
                            if len(cand_list) >= rtopk:
                                break
                            p = r.get("params")
                            if isinstance(p, dict):
                                cand_list.append({"name": f"cand_{len(cand_list)}", "seed": r.get("seed"), "params": dict(p)})

                best_robust = None
                for c in cand_list:
                    ev = _robust_evaluate(
                        bars=bars_robust,
                        params=c["params"],
                        tf=tf,
                        position_usd=position_usd,
                        train_frac=train_frac,
                        weights=weights,
                        min_trades=min_trades,
                        use_costs_in_opt=use_costs_in_opt,
                        fixed_costs=fixed_costs,
                        windows=rw,
                        window_bars=rbars,
                        shift_hours=rshift,
                        first_end_hours_ago=rfirst,
                    )
                    c["robust_eval"] = ev
                    try:
                        ev_cfg = ev.get("cfg")
                        if isinstance(ev_cfg, dict):
                            ev_cfg["base_end_shift_hours"] = float(cfg.get("end_shift_hours", 0.0) or 0.0)
                    except Exception:
                        pass
                    summ = (ev.get("summary") or {})
                    _add_log(run, f"ROBUST_CAND name={c.get('name')} mean_val={summ.get('mean_val_score')} min_val={summ.get('min_val_score')} ok={summ.get('ok_windows')}")

                    if c.get("name") == "best":
                        best_robust = ev

                if run.best_metrics is None:
                    run.best_metrics = {}
                if best_robust is not None:
                    run.best_metrics["robust_eval"] = best_robust

                # If we have candidates.json, enrich it with robust_eval (for top candidates) so UI can show it.
                if cand_payload and isinstance(cand_payload, dict):
                    top = cand_payload.get("top_candidates") or []
                    if isinstance(top, list):
                        # Map by params JSON for stability
                        def _pkey(p: Dict[str, Any]) -> str:
                            try:
                                return json.dumps(p, sort_keys=True)
                            except Exception:
                                return str(p)

                        mrob = {_pkey(c["params"]): c.get("robust_eval") for c in cand_list if isinstance(c.get("params"), dict)}
                        for r in top:
                            p = r.get("params")
                            if isinstance(p, dict):
                                k = _pkey(p)
                                if k in mrob:
                                    r["robust_eval"] = mrob[k]
                        cand_payload["top_candidates"] = top
                        save_candidates(run.run_id, cand_payload)

                _add_log(run, "ROBUST_DONE")
            except Exception as e:
                _add_log(run, f"ROBUST_ERROR {type(e).__name__}: {e}")
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

        cfg_json = r.get('config')

        try:

            cfgd = json.loads(cfg_json) if cfg_json else {}

        except Exception:

            cfgd = {}

        end_shift_h = cfgd.get('end_shift_hours', 0) or 0

        limit_h = int(cfgd.get('limit_bars', 5000) or 5000)

        chart_href = f"/chart?symbol={html.escape(symbol)}&tf={html.escape(tf)}&strategy=my_strategy3.py&limit={limit_h}&end_shift_hours={end_shift_h}&opt_id={rid}"
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
          <label><span class='lbl'>end_shift_hours<span class='tip' data-tip='Сдвиг конца базового окна назад (в часах).
0 = до последней свечи, 10 = конец окна 10 часов назад.
Это влияет и на оптимизацию, и на robust-check (нулевое окно).'>?</span></span></label>
          <input name='end_shift_hours' value='0' style='width:130px'/>
          <div class='shiftBtns'>
            <button type='button' class='btnMini' onclick="setShift(0)">0h</button>
            <button type='button' class='btnMini' onclick="setShift(6)">6h</button>
            <button type='button' class='btnMini' onclick="setShift(12)">12h</button>
            <button type='button' class='btnMini' onclick="setShift(24)">24h</button>
            <button type='button' class='btnMini' onclick="setShift(48)">48h</button>
            <button type='button' class='btnMini' onclick="setShift(72)">72h</button>
            <button type='button' class='btnMini' onclick="setShift(168)">7d</button>
          </div>
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
              <option value='mix'>mix</option>
            </select>
            <input name='mix_alpha' value='0.7' step='0.05' min='0' max='1' type='number' style='width:88px' title='mix_alpha: доля val в mix (0..1)' />
          </div>
          <div>
            <label>train_frac</label>
            <input name='train_frac' value='0.7' style='width:90px'/>
          </div>
          <div>
            <label class='lbl'>confirm_on_close <span class='tip' data-tip='Если включено: сигнал подтверждаем закрытием свечи (t-1) и исполняем на открытии следующей (t).
Если выключено: используем текущую свечу (может перерисовываться на live).'>?</span></label>
            <input type='checkbox' name='confirm_on_close' checked />

          </div>

          <div class='card' style='grid-column:1/-1; margin-top:6px; margin-bottom:16px;'>
            <div style='font-weight:700; margin-bottom:6px;'>Параметры стратегии (opt / fixed)</div>
            <div class='muted' style='margin-bottom:10px; line-height:1.35;'>
              Галочка <b>opt</b> — параметр участвует в оптимизации. Если <b>opt</b> выключено — берётся значение <b>fx</b> ниже.
            </div>

            <div style='display:grid; grid-template-columns: 200px 120px 70px 1fr; gap:8px 10px; align-items:center;'>
              <div class='muted' style='grid-column:1/-1; font-weight:700; margin-top:2px;'>Основные</div>

              <div>use_no_trade</div>
              <div>
                <select name='fx_use_no_trade' style='width:110px'>
                  <option value='true' selected>Да</option>
                  <option value='false'>Нет</option>
                </select>
              </div>
              <div><label class='lbl'><input type='checkbox' name='opt_use_no_trade' /> opt</label></div>
              <div class='muted'>Фильтр: не торговать при слабом тренде</div>

              <div>adx_no_trade_below</div>
              <div><input name='fx_adx_no_trade_below' value='24' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_adx_no_trade_below' checked /> opt</label></div>
              <div class='muted'>Порог ADX ниже — NO TRADE</div>

              <div>st_factor</div>
              <div><input name='fx_st_factor' value='4.0' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_st_factor' checked /> opt</label></div>
              <div class='muted'>Коэф Supertrend (множитель ATR)</div>

              <div>use_rev_cooldown</div>
              <div>
                <select name='fx_use_rev_cooldown' style='width:110px'>
                  <option value='true' selected>Да</option>
                  <option value='false'>Нет</option>
                </select>
              </div>
              <div><label class='lbl'><input type='checkbox' name='opt_use_rev_cooldown' /> opt</label></div>
              <div class='muted'>Пауза после разворота</div>

              <div>rev_cooldown_hrs</div>
              <div><input name='fx_rev_cooldown_hrs' value='12' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_rev_cooldown_hrs' /> opt</label></div>
              <div class='muted'>Длина паузы (часы)</div>

              <div>use_flip_limit</div>
              <div>
                <select name='fx_use_flip_limit' style='width:110px'>
                  <option value='true' selected>Да</option>
                  <option value='false'>Нет</option>
                </select>
              </div>
              <div><label class='lbl'><input type='checkbox' name='opt_use_flip_limit' /> opt</label></div>
              <div class='muted'>Лимит частых переворотов</div>

              <div>max_flips_per_day</div>
              <div><input name='fx_max_flips_per_day' value='6' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_max_flips_per_day' /> opt</label></div>
              <div class='muted'>Макс переворотов в сутки (если use_flip_limit=Да)</div>

              <div>atr_mult</div>
              <div><input name='fx_atr_mult' value='4.0' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_atr_mult' checked /> opt</label></div>
              <div class='muted'>Множитель ATR для SL</div>

              <div style='grid-column:1/-1; border-top:1px solid rgba(255,255,255,0.08); margin:6px 0;'></div>
              <div class='muted' style='grid-column:1/-1; font-weight:700;'>Прочее</div>

              <div>adx_len</div>
              <div><input name='fx_adx_len' value='14' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_adx_len' /> opt</label></div>
              <div class='muted'>Период ADX</div>

              <div>adx_smooth</div>
              <div><input name='fx_adx_smooth' value='14' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_adx_smooth' /> opt</label></div>
              <div class='muted'>Сглаживание ADX</div>

              <div>st_atr_len</div>
              <div><input name='fx_st_atr_len' value='14' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_st_atr_len' /> opt</label></div>
              <div class='muted'>ATR период для Supertrend</div>

              <div>atr_len</div>
              <div><input name='fx_atr_len' value='14' style='width:110px'/></div>
              <div><label class='lbl'><input type='checkbox' name='opt_atr_len' /> opt</label></div>
              <div class='muted'>ATR период для авар. SL</div>

              <div>use_emergency_sl</div>
              <div>
                <select name='fx_use_emergency_sl' style='width:110px'>
                  <option value='true' selected>Да</option>
                  <option value='false'>Нет</option>
                </select>
              </div>
              <div><label class='lbl'><input type='checkbox' name='opt_use_emergency_sl' /> opt</label></div>
              <div class='muted'>Аварийный стоп-лосс</div>

              <div>close_at_end</div>
              <div>
                <select name='fx_close_at_end' style='width:110px'>
                  <option value='false' selected>Нет</option>
                  <option value='true'>Да</option>
                </select>
              </div>
              <div><label class='lbl'><input type='checkbox' name='opt_close_at_end' /> opt</label></div>
              <div class='muted'>Закрыть позицию в конце окна</div>

            </div>
          </div>
          </div>
          <div>
            <label>position_usd</label>
            <input name='position_usd' value='1000' style='width:110px'/>
          </div>
          <div>
            <button type='submit' class='startbtn'>Start optimize</button>
          </div>
          <div>
            <button type='button' id='btnMultiSeed' style='background:#1f5c3b;' class='startbtn'>Start multi-seed (Random)</button>
          </div>
          <div>
            <label>multi mode</label>
            <select name='multiseed_mode' style='width:170px'>
              <option value='seeds' selected>Seeds (different RNG)</option>
              <option value='windows'>Windows (shift data)</option>
            </select>
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
            <input name='fee_percent' value='0.06' style='width:110px'/>
          </div>
          <div>
            <label>spread_ticks</label>
            <input name='spread_ticks' value='1.0' style='width:110px'/>
          </div>
          <div>
            <label>slippage_ticks</label>
            <input name='slippage_ticks' value='2' style='width:110px'/>
          </div>
          <div>
            <label>tick_size</label>
            <input name='tick_size' value='0.0001' style='width:110px'/>
          </div>
          <div>
            <label>funding_8h_percent</label>
            <input name='funding_8h_percent' value='0.01' style='width:130px'/>
          </div>
        </div>
    

        <div style='margin-top:14px; font-weight:700;'>Проверка устойчивости (walk-forward)</div>
        <div class='muted' style='margin-bottom:8px; line-height:1.35;'>
          После завершения оптимизации прогоняем best (и top-K кандидатов, если есть) на нескольких сдвинутых окнах данных.
          Это помогает понять, насколько параметры стабильны при смещении времени.
        </div>
        <div class='row' style='flex-wrap:wrap; gap:14px;'>
          <div style='flex-direction:row; align-items:center; gap:10px; padding-top:20px;'>
            <input type='checkbox' name='robust_check' id='robust_check'/>
            <label for='robust_check' style='margin:0; font-size:13px;'>robust_check</label>
          </div>
          <div>
            <label><span class='lbl'>robust_windows<span class='tip' data-tip='Сколько окон проверить (например 10).'>?</span></span></label>
            <input name='robust_windows' value='10' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>robust_bars<span class='tip' data-tip='Длина каждого окна в барах.&#10;Например 20000 для «длинной» проверки.'>?</span></span></label>
            <input name='robust_bars' value='20000' style='width:110px'/>
          </div>
          <div>
            <label><span class='lbl'>first_end_hours_ago<span class='tip' data-tip='Если >0: самое старое окно будет заканчиваться примерно N часов назад, а шаг сдвига вычислим автоматически.&#10;Последнее окно всегда заканчивается «сейчас».&#10;Если 0: используется robust_shift_hours.'>?</span></span></label>
            <input name='robust_first_end_hours_ago' value='0' style='width:130px'/>
          </div>
          <div>
            <label><span class='lbl'>robust_shift_hours<span class='tip' data-tip='Шаг сдвига конца окна назад (в часах), если first_end_hours_ago=0.&#10;Например 10 часов (как у тебя) или 24 часа.'>?</span></span></label>
            <input name='robust_shift_hours' value='10' style='width:130px'/>
          </div>
          <div>
            <label><span class='lbl'>robust_top_k<span class='tip' data-tip='Сколько кандидатов проверять.&#10;Для обычного random/optuna обычно достаточно 1 (best).&#10;Для multi-seed можно 5–20.'>?</span></span></label>
            <input name='robust_top_k' value='5' style='width:110px'/>
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
    
    function setShift(h) {{
      const el = document.querySelector("[name='end_shift_hours']");
      if (el) {{ el.value = String(h); }}
    }}
    
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

    cfg = (data or {}).get("cfg") or {}
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


    # Pretty tables for Best params and Robustness results
    params_table = ""
    if isinstance(best_params, dict) and best_params:
        rows = []
        for k in sorted(best_params.keys()):
            v = best_params.get(k)
            rows.append(f"<tr><td style='white-space:nowrap'>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>")
        params_table = (
            "<div class='card'>"
            "<h2>Best params</h2>"
            "<div style='overflow:auto'><table class='table' style='min-width:520px'>"
            "<thead><tr><th>param</th><th>value</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "</div>"
        )

    robust_table = ""
    robust = None
    if isinstance(best_metrics, dict):
        robust = best_metrics.get("robust_eval")
    if isinstance(robust, dict) and robust.get("windows"):
        summ = robust.get("summary") or {}
        cfg_r = robust.get("cfg") or {}
        # Summary line
        sum_line = (
            f"<div class='muted' style='margin-bottom:8px;'>"
            f"windows_done={html.escape(str(summ.get('windows_done')))} "
            f"ok_windows={html.escape(str(summ.get('ok_windows')))} "
            f"mean_val={html.escape(str(summ.get('mean_val_score')))} "
            f"median_val={html.escape(str(summ.get('median_val_score')))} "
            f"min_val={html.escape(str(summ.get('min_val_score')))} "
            f"max_val={html.escape(str(summ.get('max_val_score')))}"
            f"</div>"
        )

        rows = []
        for w in robust.get("windows") or []:
            tm = w.get("train_metrics") or {}
            vm = w.get("val_metrics") or {}
            def _f(d, k):
                try:
                    return float(d.get(k, 0.0))
                except Exception:
                    return 0.0
            tr_ret, tr_dd, tr_sh = _f(tm, "ret"), _f(tm, "dd"), _f(tm, "sharpe")
            va_ret, va_dd, va_sh = _f(vm, "ret"), _f(vm, "dd"), _f(vm, "sharpe")
            tr_trades = w.get("train_trades", tm.get("trades", ""))
            va_trades = w.get("val_trades", vm.get("trades", ""))

            # Link to chart for this shifted window (open in new tab).
            try:
                _sym = str((cfg or {}).get("symbol") or "")
                _tf = str((cfg or {}).get("tf") or "")
                _strategy = str((cfg or {}).get("strategy") or "my_strategy3.py")
                _limit = int(cfg_r.get("window_bars") or 5000)
                _shift_h = float(w.get("end_shift_hours") or 0)
                win_link = (
                    f"/chart?symbol={html.escape(_sym)}"
                    f"&tf={html.escape(_tf)}"
                    f"&limit={_limit}"
                    f"&strategy={html.escape(_strategy)}"
                    f"&end_shift_hours={_shift_h}"
                )
            except Exception:
                win_link = ""
            rows.append(
                "<tr>"
                f"<td>{int(w.get('i', 0))}</td>"
                f"<td>{html.escape(str(w.get('end_shift_hours')))}</td>"
                f"<td>{html.escape(str(w.get('bars')))}</td>"
                f"<td>{html.escape(str(w.get('train_score')))}</td>"
                f"<td>{html.escape(str(w.get('val_score')))}</td>"
                f"<td>{tr_ret:.4g}</td>"
                f"<td>{tr_dd:.4g}</td>"
                f"<td>{tr_sh:.4g}</td>"
                f"<td>{html.escape(str(tr_trades))}</td>"
                f"<td>{va_ret:.4g}</td>"
                f"<td>{va_dd:.4g}</td>"
                f"<td>{va_sh:.4g}</td>"
                f"<td>{html.escape(str(va_trades))}</td>"
                f"<td>{(f'<a class=\\'btn ghost\\' target=\\'_blank\\' href=\\'{win_link}\\'>📈</a>' if win_link else '')}</td>"
                "</tr>"
            )

        robust_table = (
            "<div class='card'>"
            "<h2>Robust check</h2>"
            f"<div class='muted' style='margin-bottom:6px;'>"
            f"window_bars={html.escape(str(cfg_r.get('window_bars')))} "
            f"shift_hours={html.escape(str(cfg_r.get('shift_hours')))} "
            f"first_end_hours_ago={html.escape(str(cfg_r.get('first_end_hours_ago')))} "
            f"score_on={html.escape(str(cfg_r.get('score_on','val')))}"
            f"</div>"
            + sum_line +
            "<div style='overflow:auto'>"
            "<table class='table' style='min-width:1280px'>"
            "<thead><tr>"
            "<th>i</th><th>shift_h</th><th>bars</th><th>train_score</th><th>val_score</th>"
            "<th>train_ret</th><th>train_dd</th><th>train_sharpe</th><th>train_trades</th>"
            "<th>val_ret</th><th>val_dd</th><th>val_sharpe</th><th>val_trades</th><th></th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table></div>"
            "</div>"
        )
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
                # Params preview (main knobs)
                p = r.get('params') or {}
                def _pv(key, fmt='{:.4g}'):
                    try:
                        v = p.get(key)
                        if v is None:
                            return ''
                        if isinstance(v, (int, bool)):
                            return str(v)
                        return fmt.format(float(v))
                    except Exception:
                        return str(p.get(key,''))

                params_preview = " ".join([
                    f"adx={_pv('adx_no_trade_below')}",
                    f"st={_pv('st_factor')}",
                    f"cool={_pv('rev_cooldown_hrs','{:.0f}')}",
                    f"atr={_pv('atr_mult')}",
                    f"flips={_pv('max_flips_per_day','{:.0f}')}",
                ]).strip()

                # Robust summary (if available)
                rrob = r.get('robust_eval') or {}
                rsum = (rrob.get('summary') or {}) if isinstance(rrob, dict) else {}
                r_mean = rsum.get('mean_val_score')
                r_min = rsum.get('min_val_score')
                # Chart link: load candidate params into chart.
                cand_shift_h = r.get('end_shift_hours', r.get('shift_hours', (cfg or {}).get('end_shift_hours', 0) or 0) )
                link = f"/chart?symbol={html.escape(str(cand.get('symbol','')))}&tf={html.escape(str(cand.get('tf','')))}&limit={int(cand.get('limit_bars', 5000) or 5000)}&strategy=my_strategy3.py&end_shift_hours={float(cand_shift_h or 0)}&opt_run_id={html.escape(run_id)}&opt_cand={i}"
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
                    f"<td>{html.escape(str(params_preview))}</td>"
                    f"<td>{html.escape(str(r_mean))}</td>"
                    f"<td>{html.escape(str(r_min))}</td>"
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
            <th>val_ret</th><th>val_dd</th><th>val_sharpe</th><th>val_trades</th><th>params</th><th>robust_mean</th><th>robust_min</th><th></th>
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

    {params_table}
    {robust_table}

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
    if select_on not in ("train", "val", "mix"):
        select_on = "train"

    # For select_on='mix': blended score between val and train.
    try:
        mix_alpha = float(payload.get("mix_alpha", 0.7) or 0.7)
    except Exception:
        mix_alpha = 0.7
    mix_alpha = min(1.0, max(0.0, mix_alpha))

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

    def _safe_float(v: Any, default: float) -> float:
        try:
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

    def _bdef(key: str, default: bool) -> bool:
        if key not in payload:
            return bool(default)
        return _b(key)

    # Load bars first (fast fail).
    # We anchor the window by end_ts so you can shift the end back in time (end_shift_hours).

    end_shift_hours = max(0.0, _f('end_shift_hours', 0.0))

    min_ts, max_ts = bars_min_max_ts(conn, symbol, tf)
    if max_ts is None:
        return JSONResponse(status_code=400, content={"detail": f"No bars in DB for {symbol} tf={tf}."})

    end_ts = int(max_ts - end_shift_hours * 3600.0 * 1000.0)
    if min_ts is not None and end_ts < int(min_ts):
        return JSONResponse(status_code=400, content={"detail": f"end_shift_hours={end_shift_hours} is too large; end_ts is before DB min_ts."})
    # Load the last N bars with ts <= end_ts
    # (optimization window is always anchored to this end_ts)
    bars = load_bars_before(conn, symbol, tf, end_ts=end_ts, limit=limit_bars)
    if len(bars) < 300:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Not enough bars: {len(bars)}. Need >= 300. Check symbol/tf data."},
        )

    run_id = uuid.uuid4().hex[:10]
    # Reset top candidates storage for this run (avoid mixing with previous criteria).
    save_candidates(run_id, {'top_n': top_n, 'top_candidates': [], 'ts': int(time.time())})
    # Strategy params: can be fixed (fx_*) and optionally included into optimization (opt_*).
    fixed_params = {
        # main / core
        "use_no_trade": _b("fx_use_no_trade") if "fx_use_no_trade" in payload else True,
        "adx_no_trade_below": _f("fx_adx_no_trade_below", 24.0),
        "st_factor": _f("fx_st_factor", 4.0),
        "use_rev_cooldown": _b("fx_use_rev_cooldown") if "fx_use_rev_cooldown" in payload else True,
        "rev_cooldown_hrs": _i("fx_rev_cooldown_hrs", 12),
        "use_flip_limit": _b("fx_use_flip_limit") if "fx_use_flip_limit" in payload else True,
        "max_flips_per_day": _i("fx_max_flips_per_day", 6),
        "atr_mult": _f("fx_atr_mult", 4.0),

        # other / lengths & switches
        "adx_len": _i("fx_adx_len", 14),
        "adx_smooth": _i("fx_adx_smooth", 14),
        "st_atr_len": _i("fx_st_atr_len", 14),
        "atr_len": _i("fx_atr_len", 14),
        "use_emergency_sl": _b("fx_use_emergency_sl") if "fx_use_emergency_sl" in payload else True,
        "close_at_end": _b("fx_close_at_end"),
    }

    opt_flags = {
        # main defaults: keep same behavior as older optimizer:
        #   - float knobs are optimized by default
        #   - discrete switches/hrs/limits are fixed unless user enables 'opt'
        "use_no_trade": _bdef("opt_use_no_trade", False),
        "adx_no_trade_below": _bdef("opt_adx_no_trade_below", True),
        "st_factor": _bdef("opt_st_factor", True),
        "use_rev_cooldown": _bdef("opt_use_rev_cooldown", False),
        "rev_cooldown_hrs": _bdef("opt_rev_cooldown_hrs", False),
        "use_flip_limit": _bdef("opt_use_flip_limit", False),
        "max_flips_per_day": _bdef("opt_max_flips_per_day", False),
        "atr_mult": _bdef("opt_atr_mult", True),

        # other defaults: False
        "adx_len": _bdef("opt_adx_len", False),
        "adx_smooth": _bdef("opt_adx_smooth", False),
        "st_atr_len": _bdef("opt_st_atr_len", False),
        "atr_len": _bdef("opt_atr_len", False),
        "use_emergency_sl": _bdef("opt_use_emergency_sl", False),
        "close_at_end": _bdef("opt_close_at_end", False),
    }

    cfg = {
        "symbol": symbol,
        "tf": tf,
        "end_shift_hours": float(end_shift_hours),
        "end_ts": int(end_ts),
        "end_ts_iso": dt.datetime.utcfromtimestamp(end_ts/1000.0).isoformat() + "Z",
        "limit_bars": limit_bars,
        "trials": trials,
        "max_seconds": max_seconds,
        "patience": patience,
        "position_usd": position_usd,
        "optimizer": optimizer,
        "train_frac": train_frac,
        "mix_alpha": _safe_float(payload.get("mix_alpha"), 0.7),
        # Execution mode: confirm signal on candle close (t-1) and execute on next candle open (t)
        "confirm_on_close": _b("confirm_on_close"),
        "fixed_params": fixed_params,
        "opt_flags": opt_flags,
        "seed": seed,
        "seed_from": seed_from,
        "seed_to": seed_to,
        "multiseed_mode": str(payload.get("multiseed_mode", "seeds") or "seeds").strip().lower(),
        "trials_per_seed": trials_per_seed,
        "top_n": top_n,
        "select_on": select_on,
        # costs inside optimization (optional)
        "use_costs_in_opt": _b("use_costs_in_opt"),
        "fee_percent": _f("fee_percent", 0.06),
        "spread_ticks": _f("spread_ticks", 1.0),
        "slippage_ticks": _f("slippage_ticks", 2.0),
        "tick_size": _f("tick_size", 0.0001),
        "funding_8h_percent": _f("funding_8h_percent", 0.01),
# robustness check (walk-forward) (optional)
"robust_check": _b("robust_check"),
"robust_windows": _i("robust_windows", 10),
"robust_bars": _i("robust_bars", limit_bars),
"robust_shift_hours": _f("robust_shift_hours", 10.0),
"robust_first_end_hours_ago": _f("robust_first_end_hours_ago", 0.0),
"robust_top_k": _i("robust_top_k", 10),
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

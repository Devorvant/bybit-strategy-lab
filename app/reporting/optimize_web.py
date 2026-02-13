from __future__ import annotations

import asyncio
import collections
import html
import json
import time
import uuid
import os
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
    d: Dict[str, Any] = {
        "run_id": st.run_id,
        "status": st.status,
        "trials_done": st.trials_done,
        "best_score": None if st.best_score == float("-inf") else st.best_score,
        "best_trial": st.best_trial,
        "best_params": st.best_params,
        "best_metrics": st.best_metrics,
        "cfg": st.cfg,
        "error": st.error,
        "started_at": st.started_at,
        "updated_at": time.time(),
    }
    with _lock_for(st.run_id):
        _atomic_write_json(_run_dir(st.run_id) / "progress.json", d)

def persist_log(run_id: str, msg: str) -> None:
    with _lock_for(run_id):
        _append_line(_run_dir(run_id) / "log.txt", msg)

def _add_log(st: RunState, msg: str) -> None:
    """Append a human-readable log line for UI + persist to disk."""
    st.logs.append(msg)
    if len(st.logs) > 300:
        del st.logs[:-300]
    try:
        persist_log(st.run_id, msg)
    except Exception:
        # Logging must never break the run
        pass
    st.updated_at = datetime.now(timezone.utc)


def _persist_throttled(st: RunState, *, force: bool = False) -> None:
    # Не пишем в файл слишком часто
    now = time.time()
    last = getattr(st, "_last_persist", 0.0)
    if force or (now - last) >= 1.0 or (st.trials_done % 10 == 0):
        try:
            persist_state(st)
            setattr(st, "_last_persist", now)
        except Exception:
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
# Optimizer core
# ------------------------------


def _equity_metrics(equity: List[float]) -> Dict[str, float]:
    if not equity or len(equity) < 2:
        return {"ret": 0.0, "dd": 0.0, "vol": 0.0, "sharpe": 0.0}
    ret = (equity[-1] / equity[0]) - 1.0 if equity[0] != 0 else 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0
        if dd < max_dd:
            max_dd = dd
    dd_abs = abs(max_dd)
    # log returns
    rs: List[float] = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0 and equity[i] > 0:
            rs.append((equity[i] / equity[i - 1]))
    if len(rs) < 2:
        return {"ret": ret, "dd": dd_abs, "vol": 0.0, "sharpe": 0.0}
    import math

    log_r = [math.log(x) for x in rs]
    mean_r = sum(log_r) / len(log_r)
    var = sum((x - mean_r) ** 2 for x in log_r) / (len(log_r) - 1)
    vol = math.sqrt(var)
    sharpe = 0.0 if vol == 0 else (mean_r / vol) * math.sqrt(len(log_r))
    return {"ret": ret, "dd": dd_abs, "vol": vol, "sharpe": sharpe}


def _score(m: Dict[str, float], trades: int, weights: Dict[str, float], min_trades: int) -> float:
    s = 0.0
    s += weights.get("w_ret", 2.0) * m["ret"]
    s += weights.get("w_sharpe", 0.5) * m["sharpe"]
    s -= weights.get("w_dd", 1.5) * m["dd"]
    s -= weights.get("w_vol", 0.1) * m["vol"]
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
    }

    rng = random.Random(seed)
    t0 = time.time()
    no_improve = 0

    run.status = "running"
    run.logs.append(f"RUN_START run_id={run.run_id} symbol={symbol} tf={tf} bars={len(bars)} trials={trials}")

    try:
        for trial in range(1, trials + 1):
            if max_seconds and (time.time() - t0) >= max_seconds:
                run.logs.append(f"STOP max_seconds reached at trial={trial-1}")
                break

            params = _sample_params(rng, position_usd)
            _, va_bars = _train_val_split(bars, train_frac)
            bt = backtest_strategy3(va_bars, **params)
            m = _equity_metrics(bt.equity)
            trades = len(bt.trades)
            s = _score(m, trades, weights, min_trades)

            run.trials_done = trial

            if s > run.best_score:
                run.best_score = s
                run.best_params = params
                run.best_trial = trial
                run.best_metrics = {**m, "trades": trades, "val_score": s}
                no_improve = 0
                run.logs.append(
                    f"[{trial}/{trials}] best={run.best_score:.6f} (trial={run.best_trial}) "
                    f"ret={m['ret']:.4f} dd={m['dd']:.4f} sharpe={m['sharpe']:.3f} trades={trades} "
                    f"st_factor={params['st_factor']:.2f} adx_nt<{params['adx_no_trade_below']:.1f} atr_mult={params['atr_mult']:.2f} cooldown={params['rev_cooldown_hrs']}h"
                )
            else:
                no_improve += 1

            # occasional progress line
            if trial % max(1, trials // 20) == 0:
                run.logs.append(f"PROGRESS trial={trial} best={run.best_score:.6f} best_trial={run.best_trial}")

            if patience and no_improve >= patience:
                run.logs.append(f"EARLY_STOP patience={patience} reached at trial={trial} best_trial={run.best_trial}")
                break

        run.finished_at = time.time()
        run.status = "done"

        dur = run.finished_at - run.started_at
        run.logs.append(
            f"DONE run_id={run.run_id} trials_done={run.trials_done} best_score={run.best_score:.6f} best_trial={run.best_trial} duration_sec={dur:.1f}"
        )
        if run.best_params:
            run.logs.append("BEST_PARAMS " + json.dumps(run.best_params, ensure_ascii=False))
        if run.best_metrics:
            run.logs.append("BEST_METRICS " + json.dumps(run.best_metrics, ensure_ascii=False))

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
        run.logs.append("ERROR " + run.error)
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
      <div style='font-weight:700;margin-bottom:8px;'>Saved results (final only)</div>
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
    </script>
    """

    return _page_shell("Optimizer", body)


def _optimize_run_html(run_id: str) -> str:
    body = f"""
    <div class='card'>
      <div style='display:flex; justify-content:space-between; align-items:center; gap:10px;'>
        <div>
          <div><b>run_id:</b> <span class='pill'>{html.escape(run_id)}</span></div>
          <div class='muted' id='statusLine' style='margin-top:6px;'>loading...</div>
        </div>
        <div>
          <a href='/optimize'>← back</a>
        </div>
      </div>
    </div>

    <div class='card'>
      <div style='font-weight:700;margin-bottom:8px;'>Best</div>
      <div id='bestBox' class='muted'>—</div>
    </div>

    <div class='card'>
      <div style='font-weight:700;margin-bottom:8px;'>Progress log</div>
      <pre id='logBox' class='muted'>loading...</pre>
    </div>

    <script>
      async function tick() {{
        let res;
        try {{
          res = await fetch(`/api/optimize/status?run_id={html.escape(run_id)}`);
        }} catch (e) {{
          document.getElementById('statusLine').innerText = 'fetch error: ' + (e?.message || e);
          setTimeout(tick, 1000);
          return;
        }}
        const j = await res.json();
        if (!res.ok) {{
          document.getElementById('statusLine').innerText = (j && j.detail) ? j.detail : ('http error ' + res.status);
          setTimeout(tick, 1000);
          return;
        }}
        document.getElementById('statusLine').innerText = `status=${{j.status}} trials_done=${{j.trials_done}} best_score=${{j.best_score ?? ''}}`;
        const best = {{
          best_score: j.best_score,
          best_trial: j.best_trial,
          best_params: j.best_params,
          best_metrics: j.best_metricss,
          cfg: j.cfg,
          error: j.error
        }};
        document.getElementById('bestBox').innerText = JSON.stringify(best, null, 2);
        document.getElementById('logBox').innerText = (j.logs || []).join('\n');
        if (j.status === 'running' || j.status === 'queued') {{
          setTimeout(tick, 1200);
        }}
      }}
      tick();
    </script>
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
    }

    state = RunState(run_id=run_id, cfg=cfg)
    persist_state(state)
    async with RUN_LOCK:
        RUNS[run_id] = state

    # Run in background thread to avoid blocking FastAPI event loop
    async def _bg():
        await asyncio.to_thread(_run_optimizer_sync, state, bars, conn)

    asyncio.create_task(_bg())

    return {"ok": True, "run_id": run_id}


@router.get("/api/optimize/status")
async def optimize_status(run_id: str = Query(...)):
    # 1) if current worker has it in memory, persist to file (helps other workers)
    async with RUN_LOCK:
        st = RUNS.get(run_id)
    if st is not None:
        _persist_throttled(st, force=True)

    # 2) load from file system (works across multiple worker processes)
    prog_path = _run_dir(run_id) / "progress.json"
    if not prog_path.exists():
        if st is None:
            return JSONResponse(status_code=404, content={"detail": "run_id not found"})
        # best-effort fallback
        return {
            "run_id": st.run_id,
            "status": st.status,
            "trials_done": st.trials_done,
            "best_score": None if st.best_score == float("-inf") else st.best_score,
            "best_trial": st.best_trial,
            "best_params": st.best_params,
            "best_metrics": st.best_metrics,
            "cfg": st.cfg,
            "error": st.error,
            "logs": list(st.logs),
        }

    data = json.loads(prog_path.read_text(encoding="utf-8"))
    data["logs"] = _tail_lines(_run_dir(run_id) / "log.txt", n=200)
    return data
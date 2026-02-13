#!/usr/bin/env python3
"""Optimize parameters for Strategy 3 on TF=2h using last N bars from Postgres (Railway) / DB.

Design goals:
- Parameters-only optimization (Variant 1). Strategy logic stays fixed.
- Load last `--limit-bars` bars (default 5000) for symbol+tf from DB.
- Split into TRAIN/VALIDATION (default 70/30) OR use walk-forward folds.
- Random search with weighted score: return + sharpe - drawdown - equity-volatility (+ penalty for too few trades).
- Log progress to stdout (Railway logs). Finish with a clear 'DONE'.
- Persist results to Postgres: opt_runs + opt_trials.

Run:
  python tools/optimize_strategy3_params.py --symbol APTUSDT --tf 120 --limit-bars 5000 --trials 800 --top 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from app.storage.db import init_db
from app.backtest.strategy3_backtest import backtest_strategy3  # type: ignore

Bar = Tuple[int, float, float, float, float, float]  # ts, o, h, l, c, v


def _is_postgres_conn(conn) -> bool:
    return hasattr(conn, "cursor") and not hasattr(conn, "execute")


def ensure_tables(conn) -> None:
    ddl_runs = """"
    CREATE TABLE IF NOT EXISTS opt_runs (
      id              BIGSERIAL PRIMARY KEY,
      strategy        TEXT NOT NULL,
      symbol          TEXT NOT NULL,
      tf              TEXT NOT NULL,
      created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
      started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
      finished_at     TIMESTAMPTZ,
      status          TEXT NOT NULL DEFAULT 'running',
      trials_planned  INT NOT NULL,
      trials_done     INT NOT NULL DEFAULT 0,
      best_score      DOUBLE PRECISION,
      best_metrics    JSONB,
      best_params     JSONB,
      notes           TEXT
    );
    """"

    ddl_trials = """"
    CREATE TABLE IF NOT EXISTS opt_trials (
      run_id     BIGINT NOT NULL REFERENCES opt_runs(id) ON DELETE CASCADE,
      trial      INT NOT NULL,
      split      TEXT NOT NULL,
      score      DOUBLE PRECISION NOT NULL,
      metrics    JSONB NOT NULL,
      params     JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (run_id, trial)
    );
    CREATE INDEX IF NOT EXISTS opt_trials_run_score_idx ON opt_trials(run_id, score DESC);
    """"

    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(ddl_runs)
            cur.execute(ddl_trials)
        conn.commit()
    else:
        conn.execute(ddl_runs.replace("BIGSERIAL", "INTEGER").replace("JSONB", "TEXT"))
        conn.execute(ddl_trials.replace("BIGINT", "INTEGER").replace("JSONB", "TEXT"))
        conn.commit()


def create_run(conn, strategy: str, symbol: str, tf: str, trials_planned: int, notes: str = "") -> int:
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO opt_runs(strategy, symbol, tf, trials_planned, notes)
                   VALUES(%s,%s,%s,%s,%s) RETURNING id""",
                (strategy, symbol, tf, trials_planned, notes),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
        return int(run_id)
    else:
        cur = conn.execute(
            """INSERT INTO opt_runs(strategy, symbol, tf, trials_planned, notes)
               VALUES(?,?,?,?,?)""",
            (strategy, symbol, tf, trials_planned, notes),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_run_progress(conn, run_id: int, trials_done: int,
                        best_score: Optional[float],
                        best_metrics: Optional[Dict[str, Any]],
                        best_params: Optional[Dict[str, Any]]) -> None:
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE opt_runs
                   SET trials_done=%s,
                       best_score=%s,
                       best_metrics=%s,
                       best_params=%s
                   WHERE id=%s""",
                (trials_done, best_score,
                 json.dumps(best_metrics) if best_metrics else None,
                 json.dumps(best_params) if best_params else None,
                 run_id),
            )
        conn.commit()
    else:
        conn.execute(
            """UPDATE opt_runs SET trials_done=?, best_score=?, best_metrics=?, best_params=? WHERE id=?""",
            (trials_done, best_score,
             json.dumps(best_metrics) if best_metrics else None,
             json.dumps(best_params) if best_params else None,
             run_id),
        )
        conn.commit()


def finish_run(conn, run_id: int, status: str, notes: str = "") -> None:
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE opt_runs SET status=%s, finished_at=now(), notes=COALESCE(notes,'') || %s WHERE id=%s""",
                (status, ("\n" + notes) if notes else "", run_id),
            )
        conn.commit()
    else:
        conn.execute(
            """UPDATE opt_runs SET status=?, finished_at=CURRENT_TIMESTAMP, notes=COALESCE(notes,'') || ? WHERE id=?""",
            (status, ("\n" + notes) if notes else "", run_id),
        )
        conn.commit()


def insert_trial(conn, run_id: int, trial: int, split: str,
                 score: float, metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO opt_trials(run_id, trial, split, score, metrics, params)
                   VALUES(%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, trial)
                   DO UPDATE SET score=EXCLUDED.score, metrics=EXCLUDED.metrics, params=EXCLUDED.params""",
                (run_id, trial, split, score, json.dumps(metrics), json.dumps(params)),
            )
        conn.commit()
    else:
        conn.execute(
            """INSERT OR REPLACE INTO opt_trials(run_id, trial, split, score, metrics, params)
               VALUES(?,?,?,?,?,?)""",
            (run_id, trial, split, score, json.dumps(metrics), json.dumps(params)),
        )
        conn.commit()


def load_last_bars(conn, symbol: str, tf: str, limit_bars: int) -> List[Bar]:
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ts,o,h,l,c,v FROM bars
                   WHERE symbol=%s AND tf=%s
                   ORDER BY ts DESC
                   LIMIT %s""",
                (symbol, tf, limit_bars),
            )
            rows = cur.fetchall()
        rows = list(rows)
    else:
        cur = conn.execute(
            """SELECT ts,o,h,l,c,v FROM bars
               WHERE symbol=? AND tf=?
               ORDER BY ts DESC
               LIMIT ?""",
            (symbol, tf, limit_bars),
        )
        rows = list(cur.fetchall())
    rows.reverse()
    return rows


def equity_metrics(equity: List[float]) -> Dict[str, float]:
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

    r = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0 and equity[i] > 0:
            r.append(math.log(equity[i] / equity[i - 1]))

    if len(r) < 2:
        return {"ret": ret, "dd": dd_abs, "vol": 0.0, "sharpe": 0.0}

    mean_r = sum(r) / len(r)
    var = sum((x - mean_r) ** 2 for x in r) / (len(r) - 1)
    vol = math.sqrt(var)
    sharpe = 0.0 if vol == 0 else (mean_r / vol) * math.sqrt(len(r))
    return {"ret": ret, "dd": dd_abs, "vol": vol, "sharpe": sharpe}


def score(m: Dict[str, float], trades: int,
          w_ret: float, w_dd: float, w_vol: float, w_sharpe: float,
          min_trades: int) -> float:
    s = 0.0
    s += w_ret * m["ret"]
    s += w_sharpe * m["sharpe"]
    s -= w_dd * m["dd"]
    s -= w_vol * m["vol"]

    if trades < min_trades:
        s -= 0.2 * (min_trades - trades) / max(1, min_trades)
    return s


def sample_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "position_usd": 1000.0,

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


def train_val_split(bars: List[Bar], train_frac: float) -> Tuple[List[Bar], List[Bar]]:
    n = len(bars)
    k = max(50, int(n * train_frac))
    k = min(k, n - 20)
    return bars[:k], bars[k:]


def walk_forward_splits(bars: List[Bar], folds: int, train_frac: float, val_frac: float):
    n = len(bars)
    if n < 300:
        yield bars, bars
        return
    for i in range(folds):
        train_end = int(n * (train_frac + i * ((1.0 - train_frac - val_frac) / max(1, folds - 1))))
        val_end = min(n, train_end + int(n * val_frac))
        tr = bars[:train_end]
        va = bars[train_end:val_end]
        if len(tr) >= 50 and len(va) >= 20:
            yield tr, va


def run_backtest(bars: List[Bar], params: Dict[str, Any]):
    res = backtest_strategy3(bars, **params)
    m = equity_metrics(res.equity)
    trades = len(res.trades)
    return res, m, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="APTUSDT")
    ap.add_argument("--tf", default="120")
    ap.add_argument("--limit-bars", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2)

    ap.add_argument("--w-ret", type=float, default=2.0)
    ap.add_argument("--w-dd", type=float, default=1.5)
    ap.add_argument("--w-vol", type=float, default=0.1)
    ap.add_argument("--w-sharpe", type=float, default=0.5)
    ap.add_argument("--min-trades", type=int, default=5)

    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()

    conn = init_db()
    ensure_tables(conn)

    bars = load_last_bars(conn, args.symbol, args.tf, args.limit_bars)
    if len(bars) < 300:
        print(f"Not enough bars loaded: {len(bars)} (need >= 300).")
        sys.exit(2)

    notes = f"limit_bars={args.limit_bars} trials={args.trials} split={'walk_forward' if args.walk_forward else 'single'}"
    run_id = create_run(conn, "strategy3", args.symbol, args.tf, args.trials, notes=notes)

    rng = random.Random(args.seed)

    best_score = -1e18
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_trial = 0
    no_improve = 0

    print(f"RUN_START run_id={run_id} symbol={args.symbol} tf={args.tf} bars={len(bars)} trials={args.trials}")

    try:
        trials_done = 0
        for trial in range(1, args.trials + 1):
            if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                print(f"STOP max_seconds reached at trial={trial-1}")
                break

            params = sample_params(rng)

            if args.walk_forward:
                scores = []
                for tr_b, va_b in walk_forward_splits(bars, args.folds, args.train_frac, args.val_frac):
                    _, m_va, t_va = run_backtest(va_b, params)
                    scores.append(score(m_va, t_va, args.w_ret, args.w_dd, args.w_vol, args.w_sharpe, args.min_trades))
                s_final = sum(scores) / len(scores)
                res_full, m_full, t_full = run_backtest(bars, params)
                metrics_out = {"mode": "wf", "ret": m_full["ret"], "dd": m_full["dd"], "vol": m_full["vol"], "sharpe": m_full["sharpe"], "trades": t_full, "wf_score": float(s_final)}
                split_label = "wf"
            else:
                _, va_b = train_val_split(bars, args.train_frac)
                _, m_va, t_va = run_backtest(va_b, params)
                s_final = score(m_va, t_va, args.w_ret, args.w_dd, args.w_vol, args.w_sharpe, args.min_trades)
                res_full, m_full, t_full = run_backtest(bars, params)
                metrics_out = {"mode": "val", "ret": m_full["ret"], "dd": m_full["dd"], "vol": m_full["vol"], "sharpe": m_full["sharpe"], "trades": t_full, "val_score": float(s_final)}
                split_label = "val"

            trials_done = trial
            insert_trial(conn, run_id, trial, split_label, float(s_final), metrics_out, params)

            improved = s_final > best_score
            if improved:
                best_score = float(s_final)
                best_params = params
                best_metrics = metrics_out
                best_trial = trial
                no_improve = 0
            else:
                no_improve += 1

            if trial % args.save_every == 0 or improved:
                update_run_progress(conn, run_id, trial, best_score, best_metrics, best_params)

            if improved or (trial % max(1, args.trials // 20) == 0):
                p = best_params or {}
                bm = best_metrics or {}
                print(
                    f"[{trial}/{args.trials}] best={best_score:.6f} (trial={best_trial}) "
                    f"ret={bm.get('ret',0):.4f} dd={bm.get('dd',0):.4f} sharpe={bm.get('sharpe',0):.3f} trades={bm.get('trades',0)} "
                    f"st_factor={p.get('st_factor',0):.2f} adx_nt<{p.get('adx_no_trade_below',0):.1f} atr_mult={p.get('atr_mult',0):.2f} cooldown={p.get('rev_cooldown_hrs',0)}h"
                )

            if args.patience and no_improve >= args.patience:
                print(f"EARLY_STOP patience={args.patience} reached at trial={trial} (best_trial={best_trial})")
                break

        update_run_progress(conn, run_id, trials_done, best_score, best_metrics, best_params)
        finish_run(conn, run_id, "done")

        print(f"DONE run_id={run_id} trials_done={trials_done} best_score={best_score:.6f} best_trial={best_trial}")
        if best_params:
            print(f"BEST_PARAMS {json.dumps(best_params, ensure_ascii=False)}")
        if best_metrics:
            print(f"BEST_METRICS {json.dumps(best_metrics, ensure_ascii=False)}")

    except Exception as e:
        finish_run(conn, run_id, "error", notes=f"{e!r}")
        print(f"ERROR run_id={run_id} err={e!r}")
        raise


if __name__ == "__main__":
    main()

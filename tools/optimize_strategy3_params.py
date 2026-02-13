#!/usr/bin/env python3
"""CLI optimizer for Strategy 3 parameters.

Matches your requirement:
- Loads last N bars from DB
- Random-searches parameters
- Prints iteration progress to stdout (Railway logs)
- Writes ONLY FINAL result to Postgres/SQLite (table: opt_results)

Example:
  python tools/optimize_strategy3_params.py \
    --symbol APTUSDT --tf 120 --limit-bars 5000 \
    --trials 2000 --max-seconds 900 --patience 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.storage.db import init_db
from app.backtest.strategy3_backtest import backtest_strategy3  # type: ignore

Bar = Tuple[int, float, float, float, float, float]  # ts,o,h,l,c,v


def _is_postgres() -> bool:
    url = settings.DATABASE_URL or ""
    return url.startswith("postgres")


def _ensure_opt_results_table(conn) -> None:
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


def _load_last_bars(conn, symbol: str, tf: str, limit_bars: int) -> List[Bar]:
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ts,o,h,l,c,v FROM bars
                   WHERE symbol=%s AND tf=%s
                   ORDER BY ts DESC
                   LIMIT %s""",
                (symbol, tf, limit_bars),
            )
            rows = list(cur.fetchall())
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


def _equity_metrics(equity: List[float]) -> Dict[str, float]:
    if not equity or len(equity) < 2:
        return {"ret": 0.0, "dd": 0.0, "vol": 0.0, "sharpe": 0.0}
    ret = (equity[-1] / equity[0]) - 1.0 if equity[0] else 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0
        if dd < max_dd:
            max_dd = dd
    dd_abs = abs(max_dd)
    rs = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0 and equity[i] > 0:
            rs.append(math.log(equity[i] / equity[i - 1]))
    if len(rs) < 2:
        return {"ret": ret, "dd": dd_abs, "vol": 0.0, "sharpe": 0.0}
    mean_r = sum(rs) / len(rs)
    var = sum((x - mean_r) ** 2 for x in rs) / (len(rs) - 1)
    vol = math.sqrt(var)
    sharpe = 0.0 if vol == 0 else (mean_r / vol) * math.sqrt(len(rs))
    return {"ret": ret, "dd": dd_abs, "vol": vol, "sharpe": sharpe}


def _score(m: Dict[str, float], trades: int, *, w_ret: float, w_dd: float, w_vol: float, w_sharpe: float, min_trades: int) -> float:
    s = 0.0
    s += w_ret * m["ret"]
    s += w_sharpe * m["sharpe"]
    s -= w_dd * m["dd"]
    s -= w_vol * m["vol"]
    if trades < min_trades:
        s -= 0.2 * (min_trades - trades) / max(1, min_trades)
    return float(s)


def _train_val_split(bars: List[Bar], train_frac: float) -> Tuple[List[Bar], List[Bar]]:
    n = len(bars)
    k = max(50, int(n * train_frac))
    k = min(k, n - 20)
    return bars[:k], bars[k:]


def _sample_params(rng: random.Random, position_usd: float) -> Dict[str, Any]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="APTUSDT")
    ap.add_argument("--tf", default="120")
    ap.add_argument("--limit-bars", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--max-seconds", type=int, default=900)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--position-usd", type=float, default=1000.0)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--w-ret", type=float, default=2.0)
    ap.add_argument("--w-dd", type=float, default=1.5)
    ap.add_argument("--w-vol", type=float, default=0.1)
    ap.add_argument("--w-sharpe", type=float, default=0.5)
    args = ap.parse_args()

    conn = init_db()
    _ensure_opt_results_table(conn)

    symbol = str(args.symbol).upper()
    tf = str(args.tf)
    bars = _load_last_bars(conn, symbol, tf, int(args.limit_bars))
    if len(bars) < 300:
        raise SystemExit(f"Not enough bars: {len(bars)} (need >= 300).")

    cfg = {
        "symbol": symbol,
        "tf": tf,
        "limit_bars": int(args.limit_bars),
        "trials": int(args.trials),
        "max_seconds": int(args.max_seconds),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "position_usd": float(args.position_usd),
        "min_trades": int(args.min_trades),
        "w_ret": float(args.w_ret),
        "w_dd": float(args.w_dd),
        "w_vol": float(args.w_vol),
        "w_sharpe": float(args.w_sharpe),
    }

    rng = random.Random(args.seed)
    t0 = time.time()
    best_score = float("-inf")
    best_trial = 0
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    no_improve = 0
    trials_done = 0

    print(f"RUN_START symbol={symbol} tf={tf} bars={len(bars)} trials={args.trials}")

    try:
        for trial in range(1, int(args.trials) + 1):
            if args.max_seconds and (time.time() - t0) >= int(args.max_seconds):
                print(f"STOP max_seconds reached at trial={trial-1}")
                break

            params = _sample_params(rng, float(args.position_usd))
            _, va = _train_val_split(bars, float(args.train_frac))
            bt = backtest_strategy3(va, **params)
            m = _equity_metrics(bt.equity)
            trades = len(bt.trades)
            s = _score(
                m,
                trades,
                w_ret=float(args.w_ret),
                w_dd=float(args.w_dd),
                w_vol=float(args.w_vol),
                w_sharpe=float(args.w_sharpe),
                min_trades=int(args.min_trades),
            )

            trials_done = trial

            if s > best_score:
                best_score = s
                best_trial = trial
                best_params = params
                best_metrics = {**m, "trades": trades, "val_score": s}
                no_improve = 0
                print(
                    f"[{trial}/{args.trials}] best={best_score:.6f} (trial={best_trial}) "
                    f"ret={m['ret']:.4f} dd={m['dd']:.4f} sharpe={m['sharpe']:.3f} trades={trades} "
                    f"st_factor={params['st_factor']:.2f} adx_nt<{params['adx_no_trade_below']:.1f} atr_mult={params['atr_mult']:.2f} cooldown={params['rev_cooldown_hrs']}h"
                )
            else:
                no_improve += 1

            if trial % max(1, int(args.trials) // 20) == 0:
                print(f"PROGRESS trial={trial} best={best_score:.6f} best_trial={best_trial}")

            if args.patience and no_improve >= int(args.patience):
                print(f"EARLY_STOP patience={args.patience} reached at trial={trial} best_trial={best_trial}")
                break

        duration = time.time() - t0
        _insert_opt_result(
            conn,
            strategy="strategy3",
            symbol=symbol,
            tf=tf,
            config=cfg,
            status="done",
            duration_sec=float(duration),
            trials_done=int(trials_done),
            best_score=None if best_score == float("-inf") else float(best_score),
            best_params=best_params,
            best_metrics=best_metrics,
        )

        print(f"DONE trials_done={trials_done} best_score={best_score:.6f} best_trial={best_trial} duration_sec={duration:.1f}")
        if best_params:
            print("BEST_PARAMS " + json.dumps(best_params, ensure_ascii=False))
        if best_metrics:
            print("BEST_METRICS " + json.dumps(best_metrics, ensure_ascii=False))

    except Exception as e:
        duration = time.time() - t0
        try:
            _insert_opt_result(
                conn,
                strategy="strategy3",
                symbol=symbol,
                tf=tf,
                config=cfg,
                status="error",
                duration_sec=float(duration),
                trials_done=int(trials_done),
                best_score=None if best_score == float("-inf") else float(best_score),
                best_params=best_params,
                best_metrics={"error": repr(e), **(best_metrics or {})},
            )
        finally:
            print("ERROR " + repr(e))
            raise


if __name__ == "__main__":
    main()

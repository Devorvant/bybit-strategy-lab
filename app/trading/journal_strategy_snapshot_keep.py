from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _payload_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False)


def _is_sqlite_conn(conn: Any) -> bool:
    return isinstance(conn, sqlite3.Connection)


def _exec_insert(conn: Any, pg_sql: str, sqlite_sql: str, params: dict[str, Any]) -> None:
    if _is_sqlite_conn(conn):
        values = tuple(params[k] for k in params.keys())
        conn.execute(sqlite_sql, values)
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(pg_sql, params)
    try:
        conn.commit()
    except Exception:
        pass


def log_strategy_decision(conn, row: dict) -> None:
    pg_sql = """
    INSERT INTO strategy_decisions (
        run_id,
        symbol,
        tf,
        bar_ts,
        opt_id,
        strategy_name,
        signal,
        action,
        reason,
        signal_note,
        exchange_side,
        exchange_size,
        cooldown_active,
        cooldown_until,
        blocked,
        blocked_reason,
        auto_enabled,
        execute_requested,
        executed,
        payload
    )
    VALUES (
        %(run_id)s,
        %(symbol)s,
        %(tf)s,
        %(bar_ts)s,
        %(opt_id)s,
        %(strategy_name)s,
        %(signal)s,
        %(action)s,
        %(reason)s,
        %(signal_note)s,
        %(exchange_side)s,
        %(exchange_size)s,
        %(cooldown_active)s,
        %(cooldown_until)s,
        %(blocked)s,
        %(blocked_reason)s,
        %(auto_enabled)s,
        %(execute_requested)s,
        %(executed)s,
        %(payload)s::jsonb
    )
    """

    sqlite_sql = """
    INSERT INTO strategy_decisions (
        run_id, symbol, tf, bar_ts, opt_id, strategy_name, signal, action, reason,
        signal_note, exchange_side, exchange_size, cooldown_active, cooldown_until,
        blocked, blocked_reason, auto_enabled, execute_requested, executed, payload
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    payload = dict(row or {})
    params = {
        "run_id": row.get("run_id"),
        "symbol": row.get("symbol"),
        "tf": row.get("tf"),
        "bar_ts": row.get("bar_ts"),
        "opt_id": row.get("opt_id"),
        "strategy_name": row.get("strategy_name"),
        "signal": row.get("signal"),
        "action": row.get("action"),
        "reason": row.get("reason"),
        "signal_note": row.get("signal_note"),
        "exchange_side": row.get("exchange_side"),
        "exchange_size": row.get("exchange_size"),
        "cooldown_active": row.get("cooldown_active"),
        "cooldown_until": row.get("cooldown_until"),
        "blocked": row.get("blocked"),
        "blocked_reason": row.get("blocked_reason"),
        "auto_enabled": row.get("auto_enabled"),
        "execute_requested": row.get("execute_requested"),
        "executed": row.get("executed"),
        "payload": _payload_dumps(payload),
    }
    _exec_insert(conn, pg_sql, sqlite_sql, params)


def log_execution_event(conn, row: dict) -> None:
    pg_sql = """
    INSERT INTO execution_events (
        run_id,
        symbol,
        tf,
        bar_ts,
        event_type,
        requested_action,
        side,
        qty,
        price,
        reduce_only,
        ok,
        error,
        order_id,
        order_link_id,
        response
    )
    VALUES (
        %(run_id)s,
        %(symbol)s,
        %(tf)s,
        %(bar_ts)s,
        %(event_type)s,
        %(requested_action)s,
        %(side)s,
        %(qty)s,
        %(price)s,
        %(reduce_only)s,
        %(ok)s,
        %(error)s,
        %(order_id)s,
        %(order_link_id)s,
        %(response)s::jsonb
    )
    """

    sqlite_sql = """
    INSERT INTO execution_events (
        run_id, symbol, tf, bar_ts, event_type, requested_action, side,
        qty, price, reduce_only, ok, error, order_id, order_link_id, response
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    response = row.get("response")
    params = {
        "run_id": row.get("run_id"),
        "symbol": row.get("symbol"),
        "tf": row.get("tf"),
        "bar_ts": row.get("bar_ts"),
        "event_type": row.get("event_type"),
        "requested_action": row.get("requested_action"),
        "side": row.get("side"),
        "qty": row.get("qty"),
        "price": row.get("price"),
        "reduce_only": row.get("reduce_only"),
        "ok": row.get("ok"),
        "error": row.get("error"),
        "order_id": row.get("order_id"),
        "order_link_id": row.get("order_link_id"),
        "response": _payload_dumps(response),
    }
    _exec_insert(conn, pg_sql, sqlite_sql, params)


def make_run_id(symbol: str, tf: str, opt_id: Any, started_at: float | int) -> str:
    s = str(symbol or "").upper()
    t = str(tf or "")
    o = str(opt_id if opt_id is not None else "")
    st = str(int(started_at))
    return f"{s}:{t}:{o}:{st}"


def log_exchange_snapshot(conn, row: dict) -> None:
    pg_sql = """
    INSERT INTO exchange_snapshots (
        run_id,
        symbol,
        tf,
        bar_ts,
        source,
        side,
        size,
        avg_price,
        mark_price,
        unrealised_pnl,
        leverage,
        liq_price,
        take_profit,
        stop_loss,
        has_position,
        wallet_equity,
        wallet_balance,
        available_balance,
        payload
    )
    VALUES (
        %(run_id)s,
        %(symbol)s,
        %(tf)s,
        %(bar_ts)s,
        %(source)s,
        %(side)s,
        %(size)s,
        %(avg_price)s,
        %(mark_price)s,
        %(unrealised_pnl)s,
        %(leverage)s,
        %(liq_price)s,
        %(take_profit)s,
        %(stop_loss)s,
        %(has_position)s,
        %(wallet_equity)s,
        %(wallet_balance)s,
        %(available_balance)s,
        %(payload)s::jsonb
    )
    """

    sqlite_sql = """
    INSERT INTO exchange_snapshots (
        run_id, symbol, tf, bar_ts, source, side, size, avg_price, mark_price,
        unrealised_pnl, leverage, liq_price, take_profit, stop_loss, has_position,
        wallet_equity, wallet_balance, available_balance, payload
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = {
        "run_id": row.get("run_id"),
        "symbol": row.get("symbol"),
        "tf": row.get("tf"),
        "bar_ts": row.get("bar_ts"),
        "source": row.get("source"),
        "side": row.get("side"),
        "size": row.get("size"),
        "avg_price": row.get("avg_price"),
        "mark_price": row.get("mark_price"),
        "unrealised_pnl": row.get("unrealised_pnl"),
        "leverage": row.get("leverage"),
        "liq_price": row.get("liq_price"),
        "take_profit": row.get("take_profit"),
        "stop_loss": row.get("stop_loss"),
        "has_position": row.get("has_position"),
        "wallet_equity": row.get("wallet_equity"),
        "wallet_balance": row.get("wallet_balance"),
        "available_balance": row.get("available_balance"),
        "payload": _payload_dumps(row.get("payload")),
    }
    _exec_insert(conn, pg_sql, sqlite_sql, params)


def snapshot_from_exchange(
    symbol: str,
    position_snapshot: dict | None,
    account_snapshot: dict | None,
    *,
    source: str,
    run_id: str | None = None,
    tf: str | None = None,
    bar_ts: int | None = None,
) -> dict:
    p = (position_snapshot or {}).get("position") or {}
    a = account_snapshot or {}

    return {
        "run_id": run_id,
        "symbol": symbol,
        "tf": tf,
        "bar_ts": bar_ts,
        "source": source,
        "side": p.get("side"),
        "size": p.get("size"),
        "avg_price": p.get("avgPrice"),
        "mark_price": p.get("markPrice"),
        "unrealised_pnl": p.get("unrealisedPnl"),
        "leverage": p.get("leverage"),
        "liq_price": p.get("liqPrice"),
        "take_profit": p.get("takeProfit"),
        "stop_loss": p.get("stopLoss"),
        "has_position": (position_snapshot or {}).get("has_position"),
        "wallet_equity": a.get("equity"),
        "wallet_balance": a.get("walletBalance"),
        "available_balance": a.get("availableBalance"),
        "payload": {
            "position_snapshot": position_snapshot,
            "account_snapshot": account_snapshot,
        },
    }


def log_execution_fill(conn, row: dict) -> None:
    pg_sql = """
    INSERT INTO execution_fills (
        ts,
        symbol,
        tf,
        order_link_id,
        order_id,
        exec_id,
        side,
        exec_price,
        exec_qty,
        exec_fee,
        fee_currency,
        is_maker,
        reduce_only,
        exec_time,
        payload
    )
    VALUES (
        %(ts)s,
        %(symbol)s,
        %(tf)s,
        %(order_link_id)s,
        %(order_id)s,
        %(exec_id)s,
        %(side)s,
        %(exec_price)s,
        %(exec_qty)s,
        %(exec_fee)s,
        %(fee_currency)s,
        %(is_maker)s,
        %(reduce_only)s,
        %(exec_time)s,
        %(payload)s::jsonb
    )
    ON CONFLICT (exec_id) DO NOTHING
    """

    sqlite_sql = """
    INSERT OR IGNORE INTO execution_fills (
        ts, symbol, tf, order_link_id, order_id, exec_id, side,
        exec_price, exec_qty, exec_fee, fee_currency, is_maker,
        reduce_only, exec_time, payload
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = {
        "ts": row.get("ts"),
        "symbol": row.get("symbol"),
        "tf": row.get("tf"),
        "order_link_id": row.get("order_link_id"),
        "order_id": row.get("order_id"),
        "exec_id": row.get("exec_id"),
        "side": row.get("side"),
        "exec_price": row.get("exec_price"),
        "exec_qty": row.get("exec_qty"),
        "exec_fee": row.get("exec_fee"),
        "fee_currency": row.get("fee_currency"),
        "is_maker": row.get("is_maker"),
        "reduce_only": row.get("reduce_only"),
        "exec_time": row.get("exec_time"),
        "payload": _payload_dumps(row.get("payload")),
    }
    _exec_insert(conn, pg_sql, sqlite_sql, params)


def log_execution_fills(conn, rows: list[dict]) -> None:
    for row in rows or []:
        log_execution_fill(conn, row)


def upsert_execution_fill_summary(
    conn,
    *,
    order_link_id: str,
    symbol: str,
    tf: str | None = None,
    updated_ts: int | None = None,
) -> None:
    if not order_link_id:
        return

    if _is_sqlite_conn(conn):
        cur = conn.execute(
            """
            SELECT
                COUNT(*) AS fill_count,
                SUM(exec_qty) AS sum_exec_qty,
                CASE WHEN SUM(exec_qty) > 0
                     THEN SUM(exec_qty * exec_price) / SUM(exec_qty)
                     ELSE NULL
                END AS avg_exec_price,
                SUM(exec_fee) AS sum_exec_fee,
                MIN(exec_time) AS first_exec_time,
                MAX(exec_time) AS last_exec_time,
                MAX(fee_currency) AS fee_currency,
                MAX(symbol) AS symbol,
                MAX(tf) AS tf
            FROM execution_fills
            WHERE order_link_id = ?
            """,
            (order_link_id,),
        )
        agg = cur.fetchone()
        if not agg:
            return
        conn.execute(
            """
            INSERT OR REPLACE INTO execution_fill_summary (
                order_link_id, symbol, tf, fill_count, sum_exec_qty, avg_exec_price,
                sum_exec_fee, fee_currency, first_exec_time, last_exec_time, updated_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_link_id,
                agg[7] or symbol,
                agg[8] or tf,
                agg[0],
                agg[1],
                agg[2],
                agg[3],
                agg[6],
                agg[4],
                agg[5],
                updated_ts,
            ),
        )
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH agg AS (
                SELECT
                    %(order_link_id)s::text AS order_link_id,
                    COALESCE(MAX(symbol), %(symbol)s) AS symbol,
                    COALESCE(MAX(tf), %(tf)s) AS tf,
                    COUNT(*)::int AS fill_count,
                    SUM(exec_qty)::double precision AS sum_exec_qty,
                    CASE
                        WHEN SUM(exec_qty) > 0
                        THEN SUM(exec_qty * exec_price) / SUM(exec_qty)
                        ELSE NULL
                    END AS avg_exec_price,
                    SUM(exec_fee)::double precision AS sum_exec_fee,
                    MAX(fee_currency) AS fee_currency,
                    MIN(exec_time)::bigint AS first_exec_time,
                    MAX(exec_time)::bigint AS last_exec_time,
                    %(updated_ts)s::bigint AS updated_ts
                FROM execution_fills
                WHERE order_link_id = %(order_link_id)s
            )
            INSERT INTO execution_fill_summary (
                order_link_id, symbol, tf, fill_count, sum_exec_qty, avg_exec_price,
                sum_exec_fee, fee_currency, first_exec_time, last_exec_time, updated_ts
            )
            SELECT
                order_link_id, symbol, tf, fill_count, sum_exec_qty, avg_exec_price,
                sum_exec_fee, fee_currency, first_exec_time, last_exec_time, updated_ts
            FROM agg
            ON CONFLICT (order_link_id) DO UPDATE
            SET
                symbol = EXCLUDED.symbol,
                tf = EXCLUDED.tf,
                fill_count = EXCLUDED.fill_count,
                sum_exec_qty = EXCLUDED.sum_exec_qty,
                avg_exec_price = EXCLUDED.avg_exec_price,
                sum_exec_fee = EXCLUDED.sum_exec_fee,
                fee_currency = EXCLUDED.fee_currency,
                first_exec_time = EXCLUDED.first_exec_time,
                last_exec_time = EXCLUDED.last_exec_time,
                updated_ts = EXCLUDED.updated_ts
            """,
            {
                "order_link_id": order_link_id,
                "symbol": symbol,
                "tf": tf,
                "updated_ts": updated_ts,
            },
        )
    try:
        conn.commit()
    except Exception:
        pass


def make_strategy_trade_key(symbol: str, tf: str, opt_id: Any, entry_ts: Any) -> str:
    sym = str(symbol or "").upper().strip()
    tfv = str(tf or "").strip()
    optv = str(opt_id or "").strip()
    ts = int(entry_ts or 0)
    return f"{sym}:{tfv}:{optv}:{ts}"


def upsert_strategy_trade_open(conn, row: dict) -> None:
    pg_sql = """
    INSERT INTO strategy_trade_journal (
        trade_key, symbol, tf, strategy, opt_id, side,
        entry_ts, entry_price, tp_price, sl_price,
        exit_ts, exit_price, close_reason, status,
        bybit_order_link_id, created_ts, updated_ts
    )
    VALUES (
        %(trade_key)s, %(symbol)s, %(tf)s, %(strategy)s, %(opt_id)s, %(side)s,
        %(entry_ts)s, %(entry_price)s, %(tp_price)s, %(sl_price)s,
        %(exit_ts)s, %(exit_price)s, %(close_reason)s, %(status)s,
        %(bybit_order_link_id)s, %(created_ts)s, %(updated_ts)s
    )
    ON CONFLICT (trade_key) DO UPDATE
    SET
        symbol = EXCLUDED.symbol,
        tf = EXCLUDED.tf,
        strategy = EXCLUDED.strategy,
        opt_id = EXCLUDED.opt_id,
        side = EXCLUDED.side,
        entry_ts = COALESCE(strategy_trade_journal.entry_ts, EXCLUDED.entry_ts),
        entry_price = COALESCE(strategy_trade_journal.entry_price, EXCLUDED.entry_price),
        tp_price = COALESCE(strategy_trade_journal.tp_price, EXCLUDED.tp_price),
        sl_price = COALESCE(strategy_trade_journal.sl_price, EXCLUDED.sl_price),
        bybit_order_link_id = COALESCE(strategy_trade_journal.bybit_order_link_id, EXCLUDED.bybit_order_link_id),
        updated_ts = EXCLUDED.updated_ts
    """

    sqlite_sql = """
    INSERT OR REPLACE INTO strategy_trade_journal (
        trade_key, symbol, tf, strategy, opt_id, side,
        entry_ts, entry_price, tp_price, sl_price,
        exit_ts, exit_price, close_reason, status,
        bybit_order_link_id, created_ts, updated_ts
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = {
        "trade_key": row.get("trade_key"),
        "symbol": row.get("symbol"),
        "tf": row.get("tf"),
        "strategy": row.get("strategy"),
        "opt_id": row.get("opt_id"),
        "side": row.get("side"),
        "entry_ts": row.get("entry_ts"),
        "entry_price": row.get("entry_price"),
        "tp_price": row.get("tp_price"),
        "sl_price": row.get("sl_price"),
        "exit_ts": row.get("exit_ts"),
        "exit_price": row.get("exit_price"),
        "close_reason": row.get("close_reason"),
        "status": row.get("status") or "open",
        "bybit_order_link_id": row.get("bybit_order_link_id"),
        "created_ts": row.get("created_ts"),
        "updated_ts": row.get("updated_ts"),
    }
    _exec_insert(conn, pg_sql, sqlite_sql, params)


def get_open_strategy_trade(
    conn,
    *,
    symbol: str,
    tf: str,
    strategy: str,
    opt_id: Any = None,
) -> dict | None:
    if _is_sqlite_conn(conn):
        cur = conn.execute(
            """
            SELECT trade_key, symbol, tf, strategy, opt_id, side, entry_ts, entry_price,
                   tp_price, sl_price, exit_ts, exit_price, close_reason, status,
                   bybit_order_link_id, created_ts, updated_ts
            FROM strategy_trade_journal
            WHERE symbol=? AND tf=? AND strategy=? AND (opt_id=? OR ? IS NULL) AND status='open'
            ORDER BY updated_ts DESC
            LIMIT 1
            """,
            (symbol, tf, strategy, opt_id, opt_id),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
        return dict(zip(cols, row)) if row and cols else None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_key, symbol, tf, strategy, opt_id, side, entry_ts, entry_price,
                   tp_price, sl_price, exit_ts, exit_price, close_reason, status,
                   bybit_order_link_id, created_ts, updated_ts
            FROM strategy_trade_journal
            WHERE symbol=%s AND tf=%s AND strategy=%s AND (opt_id=%s OR %s IS NULL) AND status='open'
            ORDER BY updated_ts DESC
            LIMIT 1
            """,
            (symbol, tf, strategy, opt_id, opt_id),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
        return dict(zip(cols, row)) if row and cols else None


def update_strategy_trade_close(
    conn,
    *,
    trade_key: str,
    exit_ts: Any,
    exit_price: Any = None,
    close_reason: str | None = None,
    updated_ts: Any = None,
) -> None:
    if not trade_key:
        return

    if _is_sqlite_conn(conn):
        conn.execute(
            """
            UPDATE strategy_trade_journal
            SET exit_ts = COALESCE(exit_ts, ?),
                exit_price = COALESCE(exit_price, ?),
                close_reason = COALESCE(close_reason, ?),
                status = 'closed',
                updated_ts = ?
            WHERE trade_key = ?
            """,
            (exit_ts, exit_price, close_reason, updated_ts, trade_key),
        )
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE strategy_trade_journal
            SET exit_ts = COALESCE(exit_ts, %(exit_ts)s),
                exit_price = COALESCE(exit_price, %(exit_price)s),
                close_reason = COALESCE(close_reason, %(close_reason)s),
                status = 'closed',
                updated_ts = %(updated_ts)s
            WHERE trade_key = %(trade_key)s
            """,
            {
                "trade_key": trade_key,
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "close_reason": close_reason,
                "updated_ts": updated_ts,
            },
        )
    try:
        conn.commit()
    except Exception:
        pass


def fetch_strategy_trade_journal(
    conn,
    *,
    symbol: str,
    tf: str | None = None,
    strategy: str | None = None,
    opt_id: Any = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if _is_sqlite_conn(conn):
        cur = conn.execute(
            """
            SELECT trade_key, symbol, tf, strategy, opt_id, side,
                   entry_ts, entry_price, tp_price, sl_price,
                   exit_ts, exit_price, close_reason, status,
                   bybit_order_link_id, created_ts, updated_ts
            FROM strategy_trade_journal
            WHERE symbol=? AND (? IS NULL OR tf=?) AND (? IS NULL OR strategy=?) AND (? IS NULL OR opt_id=?)
            ORDER BY updated_ts DESC
            LIMIT ?
            """,
            (symbol, tf, tf, strategy, strategy, opt_id, opt_id, limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in rows]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_key, symbol, tf, strategy, opt_id, side,
                   entry_ts, entry_price, tp_price, sl_price,
                   exit_ts, exit_price, close_reason, status,
                   bybit_order_link_id, created_ts, updated_ts
            FROM strategy_trade_journal
            WHERE symbol=%s AND (%s IS NULL OR tf=%s) AND (%s IS NULL OR strategy=%s) AND (%s IS NULL OR opt_id=%s)
            ORDER BY updated_ts DESC
            LIMIT %s
            """,
            (symbol, tf, tf, strategy, strategy, opt_id, opt_id, limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in rows]

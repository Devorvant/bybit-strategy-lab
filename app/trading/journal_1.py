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

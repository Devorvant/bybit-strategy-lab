from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.config import settings
from app.storage.db import get_conn
from app.reporting.optimize_web import _ensure_opt_results_table


def _is_postgres() -> bool:
    url = settings.DATABASE_URL or ""
    return url.startswith("postgres")


def ensure_opt_auto_columns(conn=None) -> None:
    own = conn is None
    conn = conn or get_conn()
    _ensure_opt_results_table(conn)
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE opt_results ADD COLUMN IF NOT EXISTS auto_trade_enabled BOOLEAN NOT NULL DEFAULT FALSE;")
                cur.execute("ALTER TABLE opt_results ADD COLUMN IF NOT EXISTS allowed_symbols JSONB;")
                cur.execute("ALTER TABLE opt_results ADD COLUMN IF NOT EXISTS allowed_timeframes JSONB;")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(opt_results)")
        cols = {str(r[1]) for r in (cur.fetchall() or [])}
        changed = False
        if 'auto_trade_enabled' not in cols:
            cur.execute("ALTER TABLE opt_results ADD COLUMN auto_trade_enabled INTEGER NOT NULL DEFAULT 0")
            changed = True
        if 'allowed_symbols' not in cols:
            cur.execute("ALTER TABLE opt_results ADD COLUMN allowed_symbols TEXT")
            changed = True
        if 'allowed_timeframes' not in cols:
            cur.execute("ALTER TABLE opt_results ADD COLUMN allowed_timeframes TEXT")
            changed = True
        if changed:
            conn.commit()
    if own:
        try:
            conn.close()
        except Exception:
            pass


def _loads_json(x: Any) -> Any:
    if x is None or x == '':
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, memoryview):
        x = x.tobytes().decode('utf-8', errors='ignore')
    if isinstance(x, bytes):
        x = x.decode('utf-8', errors='ignore')
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def get_opt_strategy_entry(opt_id: int, conn=None) -> Optional[Dict[str, Any]]:
    own = conn is None
    conn = conn or get_conn()
    ensure_opt_auto_columns(conn)
    row = None
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, strategy, symbol, tf, status, best_params, auto_trade_enabled, allowed_symbols, allowed_timeframes "
                "FROM opt_results WHERE id=%s LIMIT 1",
                (int(opt_id),),
            )
            row = cur.fetchone()
    else:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, strategy, symbol, tf, status, best_params, auto_trade_enabled, allowed_symbols, allowed_timeframes "
            "FROM opt_results WHERE id=? LIMIT 1",
            (int(opt_id),),
        )
        row = cur.fetchone()
    if own:
        try:
            conn.close()
        except Exception:
            pass
    if not row:
        return None
    return {
        'id': int(row[0]),
        'strategy': row[1],
        'symbol': row[2],
        'tf': str(row[3]),
        'status': row[4],
        'best_params': _loads_json(row[5]) or {},
        'auto_trade_enabled': bool(row[6]),
        'allowed_symbols': _loads_json(row[7]),
        'allowed_timeframes': _loads_json(row[8]),
    }


def set_opt_auto_status(opt_id: int, enabled: bool, allowed_symbols=None, allowed_timeframes=None, conn=None) -> Optional[Dict[str, Any]]:
    own = conn is None
    conn = conn or get_conn()
    ensure_opt_auto_columns(conn)
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE opt_results SET auto_trade_enabled=%s, allowed_symbols=%s, allowed_timeframes=%s WHERE id=%s",
                    (bool(enabled), json.dumps(allowed_symbols) if allowed_symbols is not None else None, json.dumps(allowed_timeframes) if allowed_timeframes is not None else None, int(opt_id)),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        cur = conn.cursor()
        cur.execute(
            "UPDATE opt_results SET auto_trade_enabled=?, allowed_symbols=?, allowed_timeframes=? WHERE id=?",
            (1 if enabled else 0, json.dumps(allowed_symbols) if allowed_symbols is not None else None, json.dumps(allowed_timeframes) if allowed_timeframes is not None else None, int(opt_id)),
        )
        conn.commit()
    row = get_opt_strategy_entry(int(opt_id), conn)
    if own:
        try:
            conn.close()
        except Exception:
            pass
    return row

from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

from app.config import settings
from app.storage.db import get_conn
from app.reporting.optimize_web import _ensure_opt_results_table


def _is_postgres() -> bool:
    url = settings.DATABASE_URL or ""
    return url.startswith("postgres")


_DEFAULT_RISK_PROFILES: List[Dict[str, Any]] = [
    {
        'id': 'safe_testnet_25usd',
        'name': 'safe_testnet_25usd',
        'description': '25 USDT, 1x, обязательный SL 1%',
        'position_mode': 'fixed_usd',
        'position_usd': 25.0,
        'max_margin_usdt': 25.0,
        'default_leverage': 1,
        'max_leverage': 2,
        'sl_percent': 1.0,
        'tp_percent': None,
        'require_sl': True,
        'require_tp': False,
        'wait_next_bar_on_start': True,
        'block_on_sl_failure': True,
        'close_on_sl_failure': True,
        'is_active': True,
    },
    {
        'id': 'testnet_medium_100usd',
        'name': 'testnet_medium_100usd',
        'description': '100 USDT, 1x, обязательный SL 1%',
        'position_mode': 'fixed_usd',
        'position_usd': 100.0,
        'max_margin_usdt': 100.0,
        'default_leverage': 1,
        'max_leverage': 2,
        'sl_percent': 1.0,
        'tp_percent': None,
        'require_sl': True,
        'require_tp': False,
        'wait_next_bar_on_start': True,
        'block_on_sl_failure': True,
        'close_on_sl_failure': True,
        'is_active': True,
    },
    {
        'id': 'aggressive_testnet',
        'name': 'aggressive_testnet',
        'description': '250 USDT, 2x, тестовый агрессивный профиль',
        'position_mode': 'fixed_usd',
        'position_usd': 250.0,
        'max_margin_usdt': 250.0,
        'default_leverage': 2,
        'max_leverage': 3,
        'sl_percent': 1.0,
        'tp_percent': None,
        'require_sl': True,
        'require_tp': False,
        'wait_next_bar_on_start': True,
        'block_on_sl_failure': True,
        'close_on_sl_failure': True,
        'is_active': True,
    },
]


def list_risk_profiles() -> List[Dict[str, Any]]:
    return [dict(x) for x in _DEFAULT_RISK_PROFILES]


def get_risk_profile(profile_id: Any) -> Optional[Dict[str, Any]]:
    pid = str(profile_id or '').strip()
    if not pid:
        return None
    for item in _DEFAULT_RISK_PROFILES:
        if str(item.get('id')) == pid:
            return dict(item)
    return None


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
                cur.execute("ALTER TABLE opt_results ADD COLUMN IF NOT EXISTS risk_profile_id TEXT;")
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
        if 'risk_profile_id' not in cols:
            cur.execute("ALTER TABLE opt_results ADD COLUMN risk_profile_id TEXT")
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
                "SELECT id, strategy, symbol, tf, status, best_params, auto_trade_enabled, allowed_symbols, allowed_timeframes, risk_profile_id "
                "FROM opt_results WHERE id=%s LIMIT 1",
                (int(opt_id),),
            )
            row = cur.fetchone()
    else:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, strategy, symbol, tf, status, best_params, auto_trade_enabled, allowed_symbols, allowed_timeframes, risk_profile_id "
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
    risk_profile_id = row[9]
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
        'risk_profile_id': risk_profile_id,
        'risk_profile': get_risk_profile(risk_profile_id),
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


def set_opt_risk_profile(opt_id: int, risk_profile_id: Optional[str], conn=None) -> Optional[Dict[str, Any]]:
    own = conn is None
    conn = conn or get_conn()
    ensure_opt_auto_columns(conn)
    pid = str(risk_profile_id or '').strip() or None
    if pid is not None and get_risk_profile(pid) is None:
        raise ValueError(f'unknown risk profile: {pid}')
    if _is_postgres():
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE opt_results SET risk_profile_id=%s WHERE id=%s", (pid, int(opt_id)))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        cur = conn.cursor()
        cur.execute("UPDATE opt_results SET risk_profile_id=? WHERE id=?", (pid, int(opt_id)))
        conn.commit()
    row = get_opt_strategy_entry(int(opt_id), conn)
    if own:
        try:
            conn.close()
        except Exception:
            pass
    return row

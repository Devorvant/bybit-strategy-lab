from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.storage.db import get_conn


STRATEGY_TO_OPT_KEY = {
    "my_strategy3.py": "strategy3",
}


class TradeChartRepo:
    def __init__(self, conn=None):
        self.conn = conn or get_conn()
        self.is_sqlite = isinstance(self.conn, sqlite3.Connection)

    def table_exists(self, table_name: str) -> bool:
        if self.is_sqlite:
            cur = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name=%s LIMIT 1",
                (table_name,),
            )
            return cur.fetchone() is not None

    def fetch_bars(self, symbol: str, tf: str, limit: int = 5000) -> list[tuple[int, float, float, float, float, float]]:
        if self.is_sqlite:
            cur = self.conn.execute(
                "SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT ?",
                (symbol, tf, limit),
            )
            rows = cur.fetchall()
        else:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT ts,o,h,l,c,v FROM bars WHERE symbol=%s AND tf=%s ORDER BY ts DESC LIMIT %s",
                    (symbol, tf, limit),
                )
                rows = cur.fetchall()
        return list(reversed(rows))

    def fetch_opt_params(self, symbol: str, tf: str, strategy: str, opt_id: int | None) -> dict[str, Any] | None:
        if not opt_id or not self.table_exists("opt_results"):
            return None
        opt_strategy = STRATEGY_TO_OPT_KEY.get(strategy)
        if not opt_strategy:
            return None

        if self.is_sqlite:
            cur = self.conn.execute(
                "SELECT best_params FROM opt_results WHERE id=? AND status='done' AND strategy=? AND symbol=? AND tf=?",
                (opt_id, opt_strategy, symbol, tf),
            )
            row = cur.fetchone()
        else:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT best_params FROM opt_results WHERE id=%s AND status='done' AND strategy=%s AND symbol=%s AND tf=%s",
                    (opt_id, opt_strategy, symbol, tf),
                )
                row = cur.fetchone()

        if not row:
            return None
        raw = row[0]
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    def fetch_execution_events(self, symbol: str, tf: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if not self.table_exists("execution_events"):
            return []

        if self.is_sqlite:
            if tf:
                cur = self.conn.execute(
                    """
                    SELECT id, ts, run_id, symbol, tf, bar_ts, event_type, requested_action, side,
                           qty, price, reduce_only, ok, error, response, order_id, order_link_id
                    FROM execution_events
                    WHERE symbol=? AND (tf=? OR tf IS NULL OR tf='')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, tf, limit),
                )
            else:
                cur = self.conn.execute(
                    """
                    SELECT id, ts, run_id, symbol, tf, bar_ts, event_type, requested_action, side,
                           qty, price, reduce_only, ok, error, response, order_id, order_link_id
                    FROM execution_events
                    WHERE symbol=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        else:
            with self.conn.cursor() as cur:
                if tf:
                    cur.execute(
                        """
                        SELECT id, ts, run_id, symbol, tf, bar_ts, event_type, requested_action, side,
                               qty, price, reduce_only, ok, error, response, order_id, order_link_id
                        FROM execution_events
                        WHERE symbol=%s AND (tf=%s OR tf IS NULL OR tf='')
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (symbol, tf, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, ts, run_id, symbol, tf, bar_ts, event_type, requested_action, side,
                               qty, price, reduce_only, ok, error, response, order_id, order_link_id
                        FROM execution_events
                        WHERE symbol=%s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (symbol, limit),
                    )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def fetch_exchange_snapshots(self, symbol: str, tf: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if not self.table_exists("exchange_snapshots"):
            return []

        if self.is_sqlite:
            if tf:
                cur = self.conn.execute(
                    """
                    SELECT id, ts, run_id, symbol, tf, bar_ts, source, side, size, avg_price,
                           mark_price, unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
                           has_position, wallet_equity, wallet_balance, available_balance, payload
                    FROM exchange_snapshots
                    WHERE symbol=? AND (tf=? OR tf IS NULL OR tf='')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, tf, limit),
                )
            else:
                cur = self.conn.execute(
                    """
                    SELECT id, ts, run_id, symbol, tf, bar_ts, source, side, size, avg_price,
                           mark_price, unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
                           has_position, wallet_equity, wallet_balance, available_balance, payload
                    FROM exchange_snapshots
                    WHERE symbol=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        else:
            with self.conn.cursor() as cur:
                if tf:
                    cur.execute(
                        """
                        SELECT id, ts, run_id, symbol, tf, bar_ts, source, side, size, avg_price,
                               mark_price, unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
                               has_position, wallet_equity, wallet_balance, available_balance, payload
                        FROM exchange_snapshots
                        WHERE symbol=%s AND (tf=%s OR tf IS NULL OR tf='')
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (symbol, tf, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, ts, run_id, symbol, tf, bar_ts, source, side, size, avg_price,
                               mark_price, unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
                               has_position, wallet_equity, wallet_balance, available_balance, payload
                        FROM exchange_snapshots
                        WHERE symbol=%s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (symbol, limit),
                    )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def fetch_opt_strategy_entry(self, opt_id: int | None) -> dict[str, Any] | None:
        if not opt_id or not self.table_exists("opt_results"):
            return None
        if self.is_sqlite:
            cur = self.conn.execute("SELECT * FROM opt_results WHERE id=? LIMIT 1", (opt_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if cur.description else []
        else:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM opt_results WHERE id=%s LIMIT 1", (opt_id,))
                row = cur.fetchone()
                cols = [d[0] for d in cur.description] if cur.description else []
        return dict(zip(cols, row)) if row and cols else None


__all__ = ["TradeChartRepo", "STRATEGY_TO_OPT_KEY"]

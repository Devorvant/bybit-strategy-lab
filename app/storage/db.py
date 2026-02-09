"""DB layer.

Default: SQLite (DB_PATH).
If DATABASE_URL is set (e.g. Railway Postgres), uses Postgres.

Schema is shared across both engines (bars + signals).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Optional, Sequence, Tuple

from app.config import settings


def _is_postgres() -> bool:
    url = settings.DATABASE_URL
    return bool(url) and url.startswith("postgres")


def get_conn():
    if _is_postgres():
        try:
            import psycopg2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "DATABASE_URL is set but psycopg2 is not installed. Add psycopg2-binary to requirements.txt"
            ) from e
        return psycopg2.connect(settings.DATABASE_URL)

    # SQLite
    db_path = settings.DB_PATH
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db():
    conn = get_conn()
    schema_path = os.path.join("app", "storage", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    if _is_postgres():
        # Execute each statement separately (simple splitter is OK for our tiny schema)
        cur = conn.cursor()
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            cur.execute(stmt + ";")
        conn.commit()
    else:
        conn.executescript(ddl)
        conn.commit()
    return conn


def upsert_bars(conn, symbol: str, tf: str, rows: Iterable[Tuple[int, float, float, float, float, float]]):
    """Upsert bars.

    rows: (ts,o,h,l,c,v) with ts = candle open timestamp in ms.
    """
    rows_list = list(rows)
    if not rows_list:
        return

    if _is_postgres():
        from psycopg2.extras import execute_values  # type: ignore

        sql = (
            "INSERT INTO bars(symbol, tf, ts, o, h, l, c, v) VALUES %s "
            "ON CONFLICT(symbol, tf, ts) DO UPDATE SET "
            "o=EXCLUDED.o, h=EXCLUDED.h, l=EXCLUDED.l, c=EXCLUDED.c, v=EXCLUDED.v"
        )
        values = [(symbol, tf, *r) for r in rows_list]
        with conn.cursor() as cur:
            execute_values(cur, sql, values, page_size=1000)
        conn.commit()
        return

    # SQLite
    conn.executemany(
        """
        INSERT INTO bars(symbol, tf, ts, o, h, l, c, v)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol, tf, ts) DO UPDATE SET
            o=excluded.o,
            h=excluded.h,
            l=excluded.l,
            c=excluded.c,
            v=excluded.v
        """,
        [(symbol, tf, *r) for r in rows_list],
    )
    conn.commit()


def load_bars(conn, symbol: str, tf: str, limit: int = 500):
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts,o,h,l,c,v FROM bars WHERE symbol=%s AND tf=%s ORDER BY ts DESC LIMIT %s",
                (symbol, tf, limit),
            )
            rows = cur.fetchall()
    else:
        cur = conn.execute(
            "SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT ?",
            (symbol, tf, limit),
        )
        rows = cur.fetchall()
    return list(reversed(rows))


def bars_min_max_ts(conn, symbol: str, tf: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (min_ts, max_ts) for stored bars."""
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(ts), MAX(ts) FROM bars WHERE symbol=%s AND tf=%s",
                (symbol, tf),
            )
            row = cur.fetchone()
            return (row[0], row[1]) if row else (None, None)
    else:
        cur = conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM bars WHERE symbol=? AND tf=?",
            (symbol, tf),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)


def save_signal(conn, symbol: str, tf: str, ts: int, signal: str, note: str = ""):
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO signals(symbol, tf, ts, signal, note)
                VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT(symbol, tf, ts) DO UPDATE SET
                    signal=EXCLUDED.signal,
                    note=EXCLUDED.note
                """,
                (symbol, tf, ts, signal, note),
            )
        conn.commit()
    else:
        conn.execute(
            """
            INSERT INTO signals(symbol, tf, ts, signal, note)
            VALUES(?,?,?,?,?)
            ON CONFLICT(symbol, tf, ts) DO UPDATE SET
                signal=excluded.signal,
                note=excluded.note
            """,
            (symbol, tf, ts, signal, note),
        )
        conn.commit()


def last_signal(conn, symbol: str, tf: str) -> Optional[tuple]:
    if _is_postgres():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts,signal,note FROM signals WHERE symbol=%s AND tf=%s ORDER BY ts DESC LIMIT 1",
                (symbol, tf),
            )
            return cur.fetchone()
    else:
        cur = conn.execute(
            "SELECT ts,signal,note FROM signals WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT 1",
            (symbol, tf),
        )
        return cur.fetchone()

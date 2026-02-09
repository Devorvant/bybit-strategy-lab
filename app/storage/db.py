import sqlite3
from typing import Iterable, Optional
from app.config import settings

def get_conn():
    return sqlite3.connect(settings.DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    with open("app/storage/schema.sql", "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn

def upsert_bars(conn, symbol: str, tf: str, rows: Iterable[tuple]):
    # rows: (ts,o,h,l,c,v)
    conn.executemany(
        "INSERT OR REPLACE INTO bars(symbol, tf, ts, o, h, l, c, v) VALUES(?,?,?,?,?,?,?,?)",
        [(symbol, tf, *r) for r in rows],
    )
    conn.commit()

def load_bars(conn, symbol: str, tf: str, limit: int = 500):
    cur = conn.execute(
        "SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT ?",
        (symbol, tf, limit),
    )
    rows = cur.fetchall()
    return list(reversed(rows))  # ascending by time

def save_signal(conn, symbol: str, tf: str, ts: int, signal: str, note: str = ""):
    conn.execute(
        "INSERT OR REPLACE INTO signals(symbol, tf, ts, signal, note) VALUES(?,?,?,?,?)",
        (symbol, tf, ts, signal, note),
    )
    conn.commit()

def last_signal(conn, symbol: str, tf: str) -> Optional[tuple]:
    cur = conn.execute(
        "SELECT ts,signal,note FROM signals WHERE symbol=? AND tf=? ORDER BY ts DESC LIMIT 1",
        (symbol, tf),
    )
    return cur.fetchone()

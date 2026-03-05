import asyncio
import json
import time
from typing import Any, Dict, List

import websockets

from app.config import settings
from app.storage.db import init_db, upsert_bars


def kline_topic(tf: str, symbol: str) -> str:
    # Bybit v5: topic like "kline.5.BTCUSDT"
    return f"kline.{tf}.{symbol}"


def _as_kline_rows(payload: Any) -> List[tuple]:
    """Normalize Bybit WS kline 'data' field to rows: (ts,o,h,l,c,v)."""
    if payload is None:
        return []

    items: List[Dict[str, Any]]
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = [x for x in payload if isinstance(x, dict)]
    else:
        return []

    rows = []
    for k in items:
        # Bybit v5 uses ms timestamps in 'start'
        if "start" not in k:
            continue
        ts = int(k["start"])
        o = float(k.get("open", 0) or 0)
        h = float(k.get("high", 0) or 0)
        l = float(k.get("low", 0) or 0)
        c = float(k.get("close", 0) or 0)
        v = float(k.get("volume", 0) or 0)
        rows.append((ts, o, h, l, c, v))
    return rows


async def ws_collect(conn):
    """
    Collect kline updates from Bybit WebSocket and upsert into DB.

    Improvements:
    - Handles WS 'data' being either a list of dicts or a single dict (both appear in the wild)
    - Recreates DB connection on write failure (Railway Postgres connections may drop)
    - Logs liveness counters once per minute: recv/min and write/min
    """
    url = settings.BYBIT_WS_URL
    symbols = [s.strip().upper() for s in settings.SYMBOLS if str(s).strip()]
    tfs = list(settings.TFS)

    # liveness counters
    recv_cnt = 0
    write_cnt = 0
    last_print = time.time()

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                sub_args = [kline_topic(tf, s) for s in symbols for tf in tfs]
                await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))
                print(f"[ws] connected, subscribed {len(sub_args)} topics, symbols={symbols}, tfs={tfs}")

                while True:
                    msg = await ws.recv()
                    recv_cnt += 1

                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    # print subscription ack / errors
                    if isinstance(data, dict) and data.get("op") in {"subscribe", "unsubscribe"}:
                        print(f"[ws] ack: {data}")
                        continue
                    if isinstance(data, dict) and data.get("success") is False:
                        print(f"[ws] error: {data}")
                        continue

                    topic = data.get("topic", "") if isinstance(data, dict) else ""
                    if not isinstance(topic, str) or not topic.startswith("kline."):
                        # ignore non-kline topics
                        pass
                    else:
                        parts = topic.split(".")
                        if len(parts) == 3:
                            _, tf_in, sym = parts
                            rows = _as_kline_rows(data.get("data"))  # type: ignore[arg-type]
                            if rows:
                                try:
                                    upsert_bars(conn, sym, tf_in, rows)
                                    write_cnt += 1
                                except Exception as e:
                                    print("[ws] DB ERROR:", repr(e), "-> reinit db conn")
                                    try:
                                        conn = init_db()
                                    except Exception as e2:
                                        print("[ws] DB REINIT ERROR:", repr(e2), "sleep 2s")
                                        await asyncio.sleep(2)

                    now = time.time()
                    if now - last_print >= 60:
                        print(f"[ws] alive: recv={recv_cnt}/min write={write_cnt}/min")
                        recv_cnt = 0
                        write_cnt = 0
                        last_print = now

        except Exception as e:
            print("WS ERROR:", repr(e), "reconnect in 2s")
            await asyncio.sleep(2)

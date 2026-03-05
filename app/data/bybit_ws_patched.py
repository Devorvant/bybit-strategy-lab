import asyncio
import json

import websockets

from app.config import settings
from app.storage.db import init_db, upsert_bars


def kline_topic(tf: str, symbol: str) -> str:
    return f"kline.{tf}.{symbol}"


async def ws_collect(conn):
    """
    Collect kline updates from Bybit WebSocket and upsert into DB.

    Key reliability improvement vs. older versions:
    - if DB write fails (dead/stale connection), we recreate DB connection and keep going
    - we log connect/subscribe so Railway logs clearly show WS is alive
    """
    url = settings.BYBIT_WS_URL
    symbols = [s.strip().upper() for s in settings.SYMBOLS if str(s).strip()]
    tfs = list(settings.TFS)

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                sub_args = [kline_topic(tf, s) for s in symbols for tf in tfs]
                await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))
                print(f"[ws] connected, subscribed {len(sub_args)} topics, symbols={symbols}, tfs={tfs}")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    topic = data.get("topic", "")
                    if not isinstance(topic, str) or not topic.startswith("kline."):
                        continue

                    parts = topic.split(".")
                    if len(parts) != 3:
                        continue
                    _, tf_in, sym = parts

                    kl = data.get("data", [])
                    rows = []
                    for k in kl:
                        # Bybit v5 uses ms timestamps in 'start'
                        ts = int(k["start"])
                        o = float(k["open"])
                        h = float(k["high"])
                        l = float(k["low"])
                        c = float(k["close"])
                        v = float(k.get("volume", 0) or 0)
                        rows.append((ts, o, h, l, c, v))

                    if rows:
                        try:
                            upsert_bars(conn, sym, tf_in, rows)
                        except Exception as e:
                            # On Railway / long-running connections, Postgres can drop connections.
                            # Recreate DB connection and continue.
                            print("[ws] DB ERROR:", repr(e), "-> reinit db conn")
                            try:
                                conn = init_db()
                            except Exception as e2:
                                print("[ws] DB REINIT ERROR:", repr(e2), "sleep 2s")
                                await asyncio.sleep(2)

        except Exception as e:
            print("WS ERROR:", repr(e), "reconnect in 2s")
            await asyncio.sleep(2)

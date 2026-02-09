import asyncio
import json
import websockets
from app.config import settings
from app.storage.db import upsert_bars

# Bybit V5 WS: topic = kline.{interval}.{symbol}
def kline_topic(tf: str, symbol: str) -> str:
    return f"kline.{tf}.{symbol}"

async def ws_collect(conn):
    url = settings.BYBIT_WS_URL
    symbols = [s.strip().upper() for s in settings.SYMBOLS if s.strip()]
    tf = settings.TF

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                sub_args = [kline_topic(tf, s) for s in symbols]
                await ws.send(json.dumps({"op": "subscribe", "args": sub_args}))

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
                        # Bybit часто отдаёт числа как строки
                        ts = int(k["start"])  # ms
                        o = float(k["open"])
                        h = float(k["high"])
                        l = float(k["low"])
                        c = float(k["close"])
                        v = float(k.get("volume", 0) or 0)
                        rows.append((ts, o, h, l, c, v))

                    if rows:
                        upsert_bars(conn, sym, tf_in, rows)

        except Exception as e:
            print("WS ERROR:", repr(e), "reconnect in 2s")
            await asyncio.sleep(2)

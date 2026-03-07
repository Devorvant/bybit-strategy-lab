"""Historical kline backfill via Bybit REST.

Bybit V5 endpoint: GET /v5/market/kline

- Returns candles sorted **reverse** by startTime.
- limit per page up to 1000.

We paginate backwards using `end` cursor (ms epoch), and UPSERT into DB.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, List, Optional, Tuple

import httpx

from app.config import settings
from app.storage.db import bars_min_max_ts, upsert_bars


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _fetch_kline_page(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[Tuple[int, float, float, float, float, float]]:
    """Fetch one page and return rows (ts,o,h,l,c,v) in **ascending** order."""
    params = {
        "category": settings.CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }
    r = await client.get("/v5/market/kline", params=params)
    r.raise_for_status()
    payload = r.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

    lst = ((payload.get("result") or {}).get("list")) or []
    rows: List[Tuple[int, float, float, float, float, float]] = []
    for item in lst:
        # item = [startTime, open, high, low, close, volume, turnover]
        ts = int(item[0])
        o = float(item[1])
        h = float(item[2])
        l = float(item[3])
        c = float(item[4])
        v = float(item[5])
        rows.append((ts, o, h, l, c, v))

    # API returns reverse order by startTime -> sort ascending for DB writes/consistency
    rows.sort(key=lambda x: x[0])
    return rows


async def backfill_symbol_tf(
    conn,
    symbol: str,
    tf: str,
    days: int = 365,
    sleep_ms: int = 120,
):
    """Backfill last `days` for a single (symbol, tf) into DB."""
    symbol = symbol.strip().upper()
    tf = str(tf).strip()
    end_ms_target = _now_ms()
    start_ms_target = end_ms_target - int(days) * 24 * 60 * 60 * 1000

    # If we already have data for the target range, skip.
    min_ts, max_ts = bars_min_max_ts(conn, symbol, tf)
    if min_ts is not None and max_ts is not None:
        if min_ts <= start_ms_target and max_ts >= end_ms_target - 60_000:
            print(f"[backfill] {symbol} tf={tf}: already has range, skip")
            return

    print(f"[backfill] {symbol} tf={tf}: start")

    async with httpx.AsyncClient(
        base_url=settings.BYBIT_REST_URL,
        timeout=20,
        headers={"Accept": "application/json"},
    ) as client:
        # Paginate backwards using end cursor.
        cursor_end = end_ms_target
        pages = 0
        total = 0

        while cursor_end > start_ms_target:
            rows = await _fetch_kline_page(
                client,
                symbol=symbol,
                interval=tf,
                start_ms=start_ms_target,
                end_ms=cursor_end,
                limit=1000,
            )
            if not rows:
                break

            upsert_bars(conn, symbol, tf, rows)
            pages += 1
            total += len(rows)

            oldest_ts = rows[0][0]
            if oldest_ts <= start_ms_target:
                break
            # next page goes further into the past
            cursor_end = oldest_ts - 1

            if sleep_ms:
                await asyncio.sleep(sleep_ms / 1000.0)

        print(f"[backfill] {symbol} tf={tf}: done pages={pages} rows={total}")


async def backfill_on_startup(conn):
    """Convenience: backfill configured symbols/timeframe."""
    if not settings.BACKFILL_ON_START:
        return
    for sym in settings.SYMBOLS:
        for tf in settings.TFS:
            try:
                await backfill_symbol_tf(
                    conn,
                    symbol=sym,
                    tf=tf,
                    days=settings.BACKFILL_DAYS,
                    sleep_ms=settings.BACKFILL_SLEEP_MS,
                )
            except Exception as e:
                print(f"[backfill] ERROR {sym} tf={tf}: {e!r}")

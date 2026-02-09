"""Фоновый пересчёт стратегии и запись сигналов в SQLite.

Логика:
  - берём последние N баров из БД
  - считаем текущий сигнал
  - если (ts сигнала) новый — сохраняем в таблицу signals

Это MVP-цикл без сложной синхронизации.
Для продакшена обычно добавляют:
  - события «пришёл новый бар» вместо polling
  - блокировки/транзакции на уровне БД
  - учёт «confirm» из WS (закрыта ли свеча)
"""

from __future__ import annotations

import asyncio
from typing import Dict

from app.config import settings
from app.storage.db import load_bars, last_signal, save_signal
from app.strategy.supertrend_adx import StrategyParams, latest_signal


async def strategy_loop(conn, interval_sec: float = 5.0):
    symbols = [s.strip().upper() for s in settings.SYMBOLS if s.strip()]
    tf = settings.TF
    tf_minutes = int(tf)
    lookback = settings.LOOKBACK

    # В памяти кешируем последний обработанный ts, чтобы меньше ходить в БД
    processed: Dict[str, int] = {}

    params = StrategyParams(
        atr_len=int(getattr(settings, "ATR_LEN", 10)),
        st_factor=float(getattr(settings, "ST_FACTOR", 3.0)),
        adx_len=int(getattr(settings, "ADX_LEN", 14)),
        adx_min=float(getattr(settings, "ADX_MIN", 20.0)),
    )

    while True:
        try:
            for sym in symbols:
                bars = load_bars(conn, sym, tf, limit=lookback)
                if len(bars) < 60:
                    continue

                out = latest_signal(bars, tf_minutes=tf_minutes, params=params, use_closed_bar=True)
                if not out:
                    continue

                ts, sig, note = out

                key = f"{sym}:{tf}"
                if processed.get(key) == ts:
                    continue

                prev = last_signal(conn, sym, tf)
                if prev and int(prev[0]) == ts:
                    processed[key] = ts
                    continue

                save_signal(conn, sym, tf, ts, sig, note)
                processed[key] = ts

            await asyncio.sleep(interval_sec)

        except Exception as e:
            print("STRATEGY ERROR:", repr(e), "retry in 2s")
            await asyncio.sleep(2)

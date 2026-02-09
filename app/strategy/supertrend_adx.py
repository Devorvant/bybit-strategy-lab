"""Пример стратегии: Supertrend + ADX.

Идея простая:
  - Supertrend определяет направление тренда (бычий/медвежий)
  - ADX фильтрует слабые движения (только если ADX >= порога)

На выходе — сигнал: LONG / SHORT / FLAT.

Важно:
  - В БД bars.ts — это *время открытия свечи* (ms).
  - Чтобы не торговать по «незакрытой» свече (WS может присылать обновления),
    по умолчанию мы генерируем сигнал по предпоследней свече.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from app.strategy.indicators import adx, supertrend


@dataclass(frozen=True)
class StrategyParams:
    atr_len: int = 10
    st_factor: float = 3.0
    adx_len: int = 14
    adx_min: float = 20.0


def _bars_to_df(bars: Iterable[tuple]) -> pd.DataFrame:
    # bars: [(ts,o,h,l,c,v)...]
    df = pd.DataFrame(list(bars), columns=["ts", "o", "h", "l", "c", "v"])
    if not df.empty:
        df["ts"] = df["ts"].astype(np.int64)
    return df


def compute_supertrend_adx(
    bars: Iterable[tuple],
    params: StrategyParams = StrategyParams(),
) -> pd.DataFrame:
    """Возвращает DF с колонками st, st_dir, adx, plus_di, minus_di."""
    df = _bars_to_df(bars)
    if df.empty:
        return df

    st_line, st_dir = supertrend(df["h"], df["l"], df["c"], atr_length=params.atr_len, factor=params.st_factor)
    adx_, pdi, mdi = adx(df["h"], df["l"], df["c"], length=params.adx_len)

    df["st"] = st_line
    df["st_dir"] = st_dir
    df["adx"] = adx_
    df["plus_di"] = pdi
    df["minus_di"] = mdi
    return df


def latest_signal(
    bars: Iterable[tuple],
    tf_minutes: int,
    params: StrategyParams = StrategyParams(),
    use_closed_bar: bool = True,
) -> Optional[Tuple[int, str, str]]:
    """Считает самый свежий сигнал.

    Returns
    -------
    (ts, signal, note) или None, если данных мало.

    ts — ms (время *закрытия* свечи, чтобы удобно сравнивать с уже записанным сигналом).
    """
    df = compute_supertrend_adx(bars, params=params)
    if df.empty:
        return None

    # Для устойчивости используем предпоследнюю свечу (последняя может быть незакрыта)
    idx = -2 if use_closed_bar else -1
    if len(df) < (2 if use_closed_bar else 1):
        return None

    row = df.iloc[idx]
    if np.isnan(row.get("st_dir", np.nan)) or np.isnan(row.get("adx", np.nan)):
        return None

    direction = 1 if row["st_dir"] > 0 else -1
    adx_val = float(row["adx"])
    filt = adx_val >= params.adx_min

    if direction == 1 and filt:
        sig = "LONG"
    elif direction == -1 and filt:
        sig = "SHORT"
    else:
        sig = "FLAT"

    # bars.ts — open time. Переведём в close time.
    ts_open = int(row["ts"])
    ts_close = ts_open + int(tf_minutes) * 60_000
    note = f"st_dir={direction}, adx={adx_val:.2f}, adx_min={params.adx_min}"
    return ts_close, sig, note

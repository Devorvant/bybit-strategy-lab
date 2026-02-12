# app/strategy/my_strategy_tv_like.py

"""SMA(20/50) crossover signals with TradingView-like semantics.

This module emits *signals on the close of the latest bar*.

Important: TradingView's strategy tester (by default) will execute the order on
the **next bar open**. Your Python execution layer can mimic that by placing the
order right after the bar closes and expecting the fill on the next bar.

For apples-to-apples comparison use the backtester:
`app/backtest/sma_backtest_tv_like.py`.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SignalResult:
    ts: int
    signal: str  # "LONG" / "SHORT" / "FLAT"
    note: str = ""


def sma(values: List[float], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if n <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def generate_signal(bars: List[Tuple[int, float, float, float, float, float]]) -> Optional[SignalResult]:
    """Return the latest signal computed on the last bar close.

    bars: [(ts,o,h,l,c,v), ...] sorted ascending by time.
    """
    if len(bars) < 50:
        return None

    ts = [b[0] for b in bars]
    close = [b[4] for b in bars]

    fast = sma(close, 20)
    slow = sma(close, 50)

    i = len(bars) - 1
    if fast[i] is None or slow[i] is None or fast[i - 1] is None or slow[i - 1] is None:
        return None

    if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
        return SignalResult(ts=ts[i], signal="LONG", note="SMA20 cross up SMA50 (exec next open)")
    if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
        return SignalResult(ts=ts[i], signal="SHORT", note="SMA20 cross down SMA50 (exec next open)")

    return SignalResult(ts=ts[i], signal="FLAT", note="no cross")

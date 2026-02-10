# app/strategy/my_strategy.py

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SignalResult:
    ts: int
    signal: str   # "LONG" / "SHORT" / "FLAT"
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

def generate_signal(bars: List[Tuple[int,float,float,float,float,float]]) -> Optional[SignalResult]:
    """
    bars: [(ts,o,h,l,c,v), ...] по возрастанию времени
    """
    if len(bars) < 50:
        return None

    ts = [b[0] for b in bars]
    close = [b[4] for b in bars]

    fast = sma(close, 20)
    slow = sma(close, 50)

    i = len(bars) - 1
    if fast[i] is None or slow[i] is None or fast[i-1] is None or slow[i-1] is None:
        return None

    # пересечение
    if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
        return SignalResult(ts=ts[i], signal="LONG", note="SMA20 cross up SMA50")
    if fast[i-1] >= slow[i-1] and fast[i] < slow[i]:
        return SignalResult(ts=ts[i], signal="SHORT", note="SMA20 cross down SMA50")

    return SignalResult(ts=ts[i], signal="FLAT", note="no cross")

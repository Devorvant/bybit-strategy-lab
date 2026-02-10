# app/strategy/my_strategy2.py

"""SMA20/50 cross with ADX filter (non always-in-market).

Idea: open only when a cross happens *and* trend strength (ADX) is high.
This naturally produces periods of no trading (FLAT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

Bar = Tuple[int, float, float, float, float, float]  # (ts,o,h,l,c,v)


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


def adx(bars: List[Bar], n: int = 14) -> List[Optional[float]]:
    """Wilder ADX.

    Returns list of length len(bars) with None until enough data.
    """
    out: List[Optional[float]] = [None] * len(bars)
    if len(bars) < n + 2 or n <= 0:
        return out

    # Unpack
    high = [b[2] for b in bars]
    low = [b[3] for b in bars]
    close = [b[4] for b in bars]

    tr: List[float] = [0.0] * len(bars)
    pdm: List[float] = [0.0] * len(bars)
    ndm: List[float] = [0.0] * len(bars)

    for i in range(1, len(bars)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        pdm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        ndm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Wilder smoothing for TR, +DM, -DM
    tr14 = sum(tr[1 : n + 1])
    pdm14 = sum(pdm[1 : n + 1])
    ndm14 = sum(ndm[1 : n + 1])

    def safe_div(a: float, b: float) -> float:
        return 0.0 if b == 0 else a / b

    dx: List[Optional[float]] = [None] * len(bars)
    pdi = 100.0 * safe_div(pdm14, tr14)
    ndi = 100.0 * safe_div(ndm14, tr14)
    dx[n] = 100.0 * safe_div(abs(pdi - ndi), (pdi + ndi))

    for i in range(n + 1, len(bars)):
        tr14 = tr14 - tr14 / n + tr[i]
        pdm14 = pdm14 - pdm14 / n + pdm[i]
        ndm14 = ndm14 - ndm14 / n + ndm[i]

        pdi = 100.0 * safe_div(pdm14, tr14)
        ndi = 100.0 * safe_div(ndm14, tr14)
        dx[i] = 100.0 * safe_div(abs(pdi - ndi), (pdi + ndi))

    # ADX: Wilder smoothing of DX
    start = 2 * n
    if start >= len(bars):
        return out
    adx_val = sum(x for x in dx[n : start + 1] if x is not None) / n
    out[start] = adx_val
    for i in range(start + 1, len(bars)):
        if dx[i] is None:
            continue
        adx_val = (adx_val * (n - 1) + dx[i]) / n
        out[i] = adx_val

    return out


def generate_signal(
    bars: List[Bar],
    *,
    fast_n: int = 20,
    slow_n: int = 50,
    adx_n: int = 14,
    adx_enter: float = 20.0,
) -> Optional[SignalResult]:
    """bars: [(ts,o,h,l,c,v), ...] in ascending timestamp."""
    if len(bars) < max(slow_n + 2, 2 * adx_n + 2):
        return None

    ts = [b[0] for b in bars]
    close = [b[4] for b in bars]

    fast = sma(close, fast_n)
    slow = sma(close, slow_n)
    adx_list = adx(bars, adx_n)

    i = len(bars) - 1
    if (
        fast[i] is None
        or slow[i] is None
        or fast[i - 1] is None
        or slow[i - 1] is None
        or adx_list[i] is None
    ):
        return None

    cross_up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
    cross_dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]

    if not (cross_up or cross_dn):
        return SignalResult(ts=ts[i], signal="FLAT", note="no cross")

    if adx_list[i] < adx_enter:
        return SignalResult(ts=ts[i], signal="FLAT", note=f"ADX<{adx_enter:.1f} filter")

    if cross_up:
        return SignalResult(ts=ts[i], signal="LONG", note=f"SMA{fast_n} cross up SMA{slow_n} (ADX={adx_list[i]:.1f})")
    return SignalResult(ts=ts[i], signal="SHORT", note=f"SMA{fast_n} cross down SMA{slow_n} (ADX={adx_list[i]:.1f})")

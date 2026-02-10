from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

Bar = Tuple[int, float, float, float, float, float]  # (ts,o,h,l,c,v)
Side = Literal["LONG", "SHORT"]

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

@dataclass
class Trade:
    side: Side
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float  # USD

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_ts: List[int]
    equity: List[float]
    fast: List[Optional[float]]
    slow: List[Optional[float]]

def backtest_sma_cross(
    bars: List[Bar],
    position_usd: float = 1000.0,
    fast_n: int = 20,
    slow_n: int = 50,
    close_at_end: bool = True,
) -> BacktestResult:
    if not bars:
        return BacktestResult([], [], [], [], [])

    ts = [b[0] for b in bars]
    close = [b[4] for b in bars]

    fast = sma(close, fast_n)
    slow = sma(close, slow_n)

    trades: List[Trade] = []
    equity_ts: List[int] = []
    equity: List[float] = []

    pos = 0  # 1 long, -1 short, 0 flat
    entry_ts = 0
    entry_price = 0.0
    realized = 0.0

    def m2m(i: int) -> float:
        if pos == 0:
            return realized
        # PnL на фиксированный notional (USD)
        unreal = position_usd * pos * (close[i] - entry_price) / entry_price
        return realized + unreal

    for i in range(len(bars)):
        equity_ts.append(ts[i])
        equity.append(m2m(i))

        if i == 0:
            continue
        if fast[i] is None or slow[i] is None or fast[i-1] is None or slow[i-1] is None:
            continue

        cross_up = fast[i-1] <= slow[i-1] and fast[i] > slow[i]
        cross_dn = fast[i-1] >= slow[i-1] and fast[i] < slow[i]

        px = close[i]
        t = ts[i]

        if cross_up:
            # закрыть short
            if pos == -1:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(Trade("SHORT", entry_ts, entry_price, t, px, pnl))
                pos = 0

            # открыть long
            if pos == 0:
                pos = 1
                entry_ts = t
                entry_price = px

        elif cross_dn:
            # закрыть long
            if pos == 1:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(Trade("LONG", entry_ts, entry_price, t, px, pnl))
                pos = 0

            # открыть short
            if pos == 0:
                pos = -1
                entry_ts = t
                entry_price = px

    # закрыть позицию в конце (чтобы equity финализировалась)
    if close_at_end and pos != 0:
        px = close[-1]
        t = ts[-1]
        pnl = position_usd * pos * (px - entry_price) / entry_price
        realized += pnl
        trades.append(Trade("LONG" if pos == 1 else "SHORT", entry_ts, entry_price, t, px, pnl))
        if equity:
            equity[-1] = realized

    return BacktestResult(trades, equity_ts, equity, fast, slow)

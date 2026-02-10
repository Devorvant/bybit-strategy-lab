from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

Bar = Tuple[int, float, float, float, float, float]  # (ts,o,h,l,c,v)
Side = Literal["LONG", "SHORT"]
ExitReason = Literal["CROSS", "STOP", "EOD"]

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
    exit_reason: ExitReason = "CROSS"

@dataclass
class OpenPosition:
    side: Side
    entry_ts: int
    entry_price: float
    current_ts: int
    current_price: float
    unrealized_pnl: float  # USD


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_ts: List[int]
    equity: List[float]
    fast: List[Optional[float]]
    slow: List[Optional[float]]
    open_position: Optional[OpenPosition] = None

def backtest_sma_cross(
    bars: List[Bar],
    position_usd: float = 1000.0,
    fast_n: int = 20,
    slow_n: int = 50,
    close_at_end: bool = True,
    stop_pct: Optional[float] = None,
) -> BacktestResult:
    if not bars:
        return BacktestResult([], [], [], [], [], None)

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

        # stop-loss (если включен)
        if stop_pct is not None and stop_pct > 0 and pos != 0:
            # используем High/Low бара как триггер. Выход фиксируем по close (упрощение).
            hi = bars[i][2]
            lo = bars[i][3]
            stop_hit = False
            if pos == 1 and lo <= entry_price * (1.0 - stop_pct):
                stop_hit = True
            if pos == -1 and hi >= entry_price * (1.0 + stop_pct):
                stop_hit = True

            if stop_hit:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(
                    Trade(
                        "LONG" if pos == 1 else "SHORT",
                        entry_ts,
                        entry_price,
                        t,
                        px,
                        pnl,
                        exit_reason="STOP",
                    )
                )
                pos = 0

        if cross_up:
            # закрыть short
            if pos == -1:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(Trade("SHORT", entry_ts, entry_price, t, px, pnl, exit_reason="CROSS"))
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
                trades.append(Trade("LONG", entry_ts, entry_price, t, px, pnl, exit_reason="CROSS"))
                pos = 0

            # открыть short
            if pos == 0:
                pos = -1
                entry_ts = t
                entry_price = px

    open_position: Optional[OpenPosition] = None

    # закрыть позицию в конце (чтобы equity финализировалась)
    if close_at_end and pos != 0:
        px = close[-1]
        t = ts[-1]
        pnl = position_usd * pos * (px - entry_price) / entry_price
        realized += pnl
        trades.append(
            Trade(
                "LONG" if pos == 1 else "SHORT",
                entry_ts,
                entry_price,
                t,
                px,
                pnl,
                exit_reason="EOD",
            )
        )
        if equity:
            equity[-1] = realized
        pos = 0

    # информация об открытой позиции (если НЕ закрываем в конце)
    if not close_at_end and pos != 0:
        current_px = close[-1]
        current_t = ts[-1]
        unreal = position_usd * pos * (current_px - entry_price) / entry_price
        open_position = OpenPosition(
            side="LONG" if pos == 1 else "SHORT",
            entry_ts=entry_ts,
            entry_price=entry_price,
            current_ts=current_t,
            current_price=current_px,
            unrealized_pnl=unreal,
        )

    return BacktestResult(trades, equity_ts, equity, fast, slow, open_position)

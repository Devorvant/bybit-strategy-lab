"""TradingView-like backtest for SMA crossover.

Goal
----
Match TradingView strategy engine *default* execution semantics as closely as
possible using only OHLCV bars:

* Signal is detected on bar CLOSE (based on close-derived SMAs).
* Orders are executed on NEXT bar OPEN (TradingView default).
* Reversals (long<->short) happen at the same next-open price.
* Optional: commission (percent of notional) and slippage (basis points).

Notes / limitations
-------------------
TradingView also has settings like Bar Magnifier, intra-bar order filling rules,
tick size slippage, etc. With only OHLC bars we approximate slippage as a bps
price impact on the fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

Bar = Tuple[int, float, float, float, float, float]  # (ts,o,h,l,c,v)
Side = Literal["LONG", "SHORT"]
ExitReason = Literal["CROSS", "EOD"]


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
    pnl: float  # USD, AFTER fees
    fees: float  # USD
    exit_reason: ExitReason = "CROSS"


@dataclass
class OpenPosition:
    side: Side
    entry_ts: int
    entry_price: float
    current_ts: int
    current_price: float
    unrealized_pnl: float  # USD, AFTER estimated exit fees


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_ts: List[int]
    equity: List[float]
    fast: List[Optional[float]]
    slow: List[Optional[float]]
    open_position: Optional[OpenPosition] = None


def _apply_slippage(fill_px: float, side: int, slippage_bps: float) -> float:
    """side: +1 buy, -1 sell"""
    if slippage_bps <= 0:
        return fill_px
    impact = slippage_bps / 10000.0
    if side > 0:
        return fill_px * (1.0 + impact)
    return fill_px * (1.0 - impact)


def _fee_for_notional(notional_usd: float, fee_rate: float) -> float:
    """fee_rate: e.g. 0.0006 for 0.06%"""
    if fee_rate <= 0:
        return 0.0
    return abs(notional_usd) * fee_rate


def backtest_sma_cross_tv_like(
    bars: List[Bar],
    position_usd: float = 1000.0,
    fast_n: int = 20,
    slow_n: int = 50,
    close_at_end: bool = True,
    fee_rate: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """SMA cross, TradingView-like execution.

    Parameters
    ----------
    position_usd:
        Fixed notional per position (like TradingView's `strategy.cash`).
    fee_rate:
        Commission per fill as fraction of notional (0.0006 = 0.06%).
        We charge it on entry AND on exit (two fills).
    slippage_bps:
        Slippage applied to fills at next OPEN (bps).
    """

    if not bars:
        return BacktestResult([], [], [], [], [], None)

    ts = [b[0] for b in bars]
    o = [b[1] for b in bars]
    c = [b[4] for b in bars]

    fast = sma(c, fast_n)
    slow = sma(c, slow_n)

    trades: List[Trade] = []
    equity_ts: List[int] = []
    equity: List[float] = []

    # position state
    pos = 0  # 1 long, -1 short, 0 flat
    entry_ts = 0
    entry_px = 0.0
    entry_fee_paid = 0.0
    realized = 0.0

    # pending target position to be executed at next bar OPEN
    # target_pos in {-1,0,1} or None means no order
    pending_target_pos: Optional[int] = None
    pending_reason: ExitReason = "CROSS"

    def m2m(i: int) -> float:
        if pos == 0:
            return realized
        # mark-to-market on close; unrealized before exit fees
        unreal = position_usd * pos * (c[i] - entry_px) / entry_px
        # conservative: subtract estimated exit fee if we were to close now
        est_exit_fee = _fee_for_notional(position_usd, fee_rate)
        return realized + unreal - est_exit_fee

    # iterate bars
    for i in range(len(bars)):
        # 1) execute pending order on OPEN of this bar
        if pending_target_pos is not None:
            fill_open = o[i]

            # if we need to close existing position
            if pos != 0 and pending_target_pos != pos:
                exit_side = -pos  # sell if long, buy if short
                exit_px = _apply_slippage(fill_open, exit_side, slippage_bps)
                gross_pnl = position_usd * pos * (exit_px - entry_px) / entry_px
                exit_fee = _fee_for_notional(position_usd, fee_rate)
                realized += gross_pnl - exit_fee

                trades.append(
                    Trade(
                        side="LONG" if pos == 1 else "SHORT",
                        entry_ts=entry_ts,
                        entry_price=entry_px,
                        exit_ts=ts[i],
                        exit_price=exit_px,
                        pnl=gross_pnl - (entry_fee_paid + exit_fee),
                        fees=entry_fee_paid + exit_fee,
                        exit_reason=pending_reason,
                    )
                )
                pos = 0
                entry_fee_paid = 0.0

            # if we need to open a new position
            if pending_target_pos != 0 and pending_target_pos != pos:
                entry_side = pending_target_pos  # buy for long, sell for short
                entry_px_new = _apply_slippage(fill_open, entry_side, slippage_bps)
                entry_fee = _fee_for_notional(position_usd, fee_rate)
                realized -= entry_fee

                pos = pending_target_pos
                entry_ts = ts[i]
                entry_px = entry_px_new
                entry_fee_paid = entry_fee

            pending_target_pos = None

        # 2) record equity at CLOSE of this bar
        equity_ts.append(ts[i])
        equity.append(m2m(i))

        # 3) generate signal on CLOSE of this bar, schedule order for NEXT OPEN
        # cannot schedule if no next bar
        if i >= len(bars) - 1:
            continue
        if i == 0:
            continue
        if fast[i] is None or slow[i] is None or fast[i - 1] is None or slow[i - 1] is None:
            continue

        cross_up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
        cross_dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]

        if cross_up:
            pending_target_pos = 1
            pending_reason = "CROSS"
        elif cross_dn:
            pending_target_pos = -1
            pending_reason = "CROSS"

    open_position: Optional[OpenPosition] = None

    # Close at end using LAST close (best-effort). TradingView can force close
    # in tester, but the exact fill depends on settings. We'll close at last close.
    if close_at_end and pos != 0:
        last_i = len(bars) - 1
        exit_side = -pos
        exit_px = _apply_slippage(c[last_i], exit_side, slippage_bps)
        gross_pnl = position_usd * pos * (exit_px - entry_px) / entry_px
        exit_fee = _fee_for_notional(position_usd, fee_rate)
        realized += gross_pnl - exit_fee

        # allocate fees to the closing trade (entry+exit)
        trade_entry_fee = entry_fee_paid
        trade_exit_fee = exit_fee
        trades.append(
            Trade(
                side="LONG" if pos == 1 else "SHORT",
                entry_ts=entry_ts,
                entry_price=entry_px,
                exit_ts=ts[last_i],
                exit_price=exit_px,
                pnl=gross_pnl - (trade_entry_fee + trade_exit_fee),
                fees=trade_entry_fee + trade_exit_fee,
                exit_reason="EOD",
            )
        )
        if equity:
            equity[-1] = realized
        pos = 0
        entry_fee_paid = 0.0

    if not close_at_end and pos != 0:
        last_i = len(bars) - 1
        unreal = position_usd * pos * (c[last_i] - entry_px) / entry_px
        est_exit_fee = _fee_for_notional(position_usd, fee_rate)
        open_position = OpenPosition(
            side="LONG" if pos == 1 else "SHORT",
            entry_ts=entry_ts,
            entry_price=entry_px,
            current_ts=ts[last_i],
            current_price=c[last_i],
            unrealized_pnl=unreal - est_exit_fee,
        )

    return BacktestResult(trades, equity_ts, equity, fast, slow, open_position)

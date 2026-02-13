from __future__ import annotations

"""TradingView-like backtest for Strategy 3.

This mirrors the provided Pine Script:
  "Always-in-Market L/S v2 (Railway): Supertrend + ADX NoTrade + Reverse Cooldown"

Main alignment points vs TradingView default strategy execution:
  - Signals are evaluated on bar close.
  - Orders (close/entry/reverse) are filled on NEXT bar OPEN
    (i.e. Pine default when process_orders_on_close = false).
  - Commission is percent of notional per fill.
  - Slippage is in ticks (syminfo.mintick * ticks).
  - Position sizing uses percent-of-equity (like strategy.percent_of_equity).

Limitations (shared by most bar-based backtests):
  - For stop orders we only know that high/low crossed the stop within the bar;
    we fill at stop price +/- slippage ticks (worst direction).
  - TradingView's internal broker emulator has additional nuances; this is a
    close practical match for comparing trade lists and performance.
"""

from dataclasses import dataclass
from datetime import datetime
from math import floor, log10
from typing import List, Optional, Tuple, Literal


Bar = Tuple[int, float, float, float, float, float]  # (ts_ms, o, h, l, c, v) where ts_ms is bar OPEN time
Side = Literal["LONG", "SHORT"]


# -----------------------------------------------------------------------------
# Indicators (kept consistent with app/backtest/strategy3_backtest.py)
# -----------------------------------------------------------------------------

def _rma(values: List[float], n: int) -> List[Optional[float]]:
    """Wilder's RMA (TradingView ta.rma) with SMA initialization.

    TradingView/Pine initializes rma with the SMA of the first `n` values,
    producing `na` for the first (n-1) bars.
    """
    out: List[Optional[float]] = [None] * len(values)
    if n <= 0 or not values:
        return out
    if len(values) < n:
        return out

    # init with SMA on bar (n-1)
    sma0 = sum(values[:n]) / n
    out[n - 1] = sma0
    prev = sma0
    for i in range(n, len(values)):
        prev = (prev * (n - 1) + values[i]) / n
        out[i] = prev
    return out


def atr(high: List[float], low: List[float], close: List[float], n: int) -> List[Optional[float]]:
    tr: List[float] = [0.0] * len(close)
    for i in range(len(close)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
    return _rma(tr, n)


def adx(
    high: List[float],
    low: List[float],
    close: List[float],
    di_len: int,
    adx_smooth: int,
) -> List[Optional[float]]:
    """TradingView-compatible DMI/ADX: DI uses di_len, ADX uses adx_smooth."""
    n = len(close)
    if n == 0:
        return []

    up_move: List[float] = [0.0] * n
    down_move: List[float] = [0.0] * n
    tr: List[float] = [0.0] * n

    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        up_move[i] = up if (up > dn and up > 0) else 0.0
        down_move[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr[0] = high[0] - low[0]

    tr_rma = _rma(tr, di_len)
    plus_rma = _rma(up_move, di_len)
    minus_rma = _rma(down_move, di_len)

    dx: List[float] = [0.0] * n
    for i in range(n):
        if tr_rma[i] in (None, 0.0):
            dx[i] = 0.0
            continue
        trv = tr_rma[i] or 0.0
        pdi = 100.0 * (plus_rma[i] or 0.0) / trv
        mdi = 100.0 * (minus_rma[i] or 0.0) / trv
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom

    return _rma(dx, adx_smooth)


def supertrend(
    high: List[float],
    low: List[float],
    close: List[float],
    factor: float,
    atr_len: int,
) -> Tuple[List[Optional[float]], List[Optional[int]]]:
    """Return (st_line, st_dir) where st_dir is +1 (up) or -1 (down)."""
    n = len(close)
    st_line: List[Optional[float]] = [None] * n
    st_dir: List[Optional[int]] = [None] * n
    atr_v = atr(high, low, close, atr_len)

    final_upper: List[Optional[float]] = [None] * n
    final_lower: List[Optional[float]] = [None] * n

    for i in range(n):
        if atr_v[i] is None:
            continue
        mid = (high[i] + low[i]) / 2.0
        basic_upper = mid + factor * atr_v[i]
        basic_lower = mid - factor * atr_v[i]

        if i == 0 or final_upper[i - 1] is None or final_lower[i - 1] is None:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
        else:
            prev_fu = final_upper[i - 1]
            prev_fl = final_lower[i - 1]
            prev_close = close[i - 1]

            # TV-style band "stickiness"
            final_upper[i] = basic_upper if (basic_upper < prev_fu or prev_close > prev_fu) else prev_fu
            final_lower[i] = basic_lower if (basic_lower > prev_fl or prev_close < prev_fl) else prev_fl

        # direction (TradingView-like):
        # In uptrend (dir=+1) the stop line is the LOWER band; trend flips to -1 when close crosses BELOW it.
        # In downtrend (dir=-1) the stop line is the UPPER band; trend flips to +1 when close crosses ABOVE it.
        if i == 0 or st_dir[i - 1] is None:
            st_dir[i] = 1
        else:
            prev_dir = st_dir[i - 1]
            fu = final_upper[i]
            fl = final_lower[i]
            if prev_dir == 1 and fl is not None and close[i] < fl:
                st_dir[i] = -1
            elif prev_dir == -1 and fu is not None and close[i] > fu:
                st_dir[i] = 1
            else:
                st_dir[i] = prev_dir

        # line
        if st_dir[i] == 1:
            st_line[i] = final_lower[i]
        elif st_dir[i] == -1:
            st_line[i] = final_upper[i]

    return st_line, st_dir


# -----------------------------------------------------------------------------
# Backtest structures
# -----------------------------------------------------------------------------


@dataclass
class Trade:
    side: Side
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float  # net PnL incl. fees (USD)
    exit_reason: str
    cum_pnl: float


@dataclass
class BacktestResult:
    trades: List[Trade]  # closed trades only
    open_position: Optional[dict]
    equity_ts: List[int]
    equity: List[float]
    st_line: List[Optional[float]]
    st_dir: List[Optional[int]]
    adx: List[Optional[float]]
    no_trade: List[bool]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _estimate_tick_size(prices: List[float]) -> Optional[float]:
    """Best-effort tick size estimation from a price series.

    TradingView uses syminfo.mintick. In Python we don't have instrument meta,
    so we estimate. You can pass tick_size explicitly to be exact.
    """
    diffs: List[float] = []
    for i in range(1, len(prices)):
        d = abs(prices[i] - prices[i - 1])
        if d > 0:
            diffs.append(d)
    if not diffs:
        return None

    dmin = min(diffs)

    # Reduce floating noise: round to a reasonable number of decimals based on magnitude.
    if dmin <= 0:
        return None
    mag = 10 ** (-floor(log10(dmin)))
    # try rounding to 0..10 decimals and pick the first stable value
    for dec in range(0, 11):
        r = round(dmin, dec)
        if r > 0 and abs(r - dmin) < (1.0 / mag) * 1e-9:
            return r
    return dmin


def _slip(px: float, side: int, slippage_ticks: int, tick_size: float) -> float:
    """Apply adverse slippage.

    side: +1 for buy fills (long entry / short exit), -1 for sell fills.
    """
    if slippage_ticks <= 0 or tick_size <= 0:
        return px
    return px + side * slippage_ticks * tick_size


# -----------------------------------------------------------------------------
# TradingView-like backtest
# -----------------------------------------------------------------------------


def backtest_strategy3_tv_like(
    bars: List[Bar],
    *,
    initial_capital: float = 10000.0,
    percent_of_equity: float = 50.0,
    commission_percent: float = 0.10,  # TradingView commission_value (percent)
    slippage_ticks: int = 2,
    tick_size: Optional[float] = None,
    use_no_trade: bool = True,
    adx_len: int = 14,
    adx_smooth: int = 14,
    adx_no_trade_below: float = 14.0,
    st_atr_len: int = 14,
    st_factor: float = 4.0,
    use_rev_cooldown: bool = True,
    rev_cooldown_hrs: int = 8,
    use_flip_limit: bool = False,
    max_flips_per_day: int = 6,
    use_emergency_sl: bool = True,
    atr_len: int = 14,
    atr_mult: float = 3.0,
    close_at_end: bool = False,
) -> BacktestResult:
    if not bars:
        return BacktestResult([], None, [], [], [], [], [], [])

    ts = [b[0] for b in bars]
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]

    if tick_size is None:
        tick_size = _estimate_tick_size(c) or 0.0001

    fee_rate = max(0.0, commission_percent / 100.0)

    st_line, st_dir = supertrend(h, l, c, st_factor, st_atr_len)
    adx_v = adx(h, l, c, adx_len, adx_smooth)
    atr_v = atr(h, l, c, atr_len)

    # calc_on_order_fills (Pine) means stops exist immediately after an entry fills on bar OPEN.
    # At that moment, the current bar has just opened (high=low=close=open), so ta.atr() is based
    # on an 'open-tick' TR, not the final bar range.
    # We approximate this by building an ATR series on a synthetic OHLC where H=L=C=O for each bar.


    no_trade: List[bool] = [False] * len(bars)
    for i in range(len(bars)):
        if not use_no_trade:
            no_trade[i] = False
        else:
            av = adx_v[i]
            no_trade[i] = (av is not None) and (av < adx_no_trade_below)

    # State
    trades: List[Trade] = []
    equity_ts: List[int] = []
    equity: List[float] = []

    pos = 0  # +1 long, -1 short, 0 flat
    qty = 0.0  # base units
    entry_ts = 0
    entry_px = 0.0  # fill price
    entry_bar_i = -1  # bar index where position was opened (for TV-like stop timing)
    entry_fee = 0.0

    realized_equity = initial_capital
    cum_pnl = 0.0

    last_flip_time: Optional[int] = None
    cooldown_ms = int(rev_cooldown_hrs * 60 * 60 * 1000)

    flips_today = 0
    cur_day = datetime.utcfromtimestamp(ts[0] / 1000).date()

    # Pending target position to execute at NEXT bar open
    pending_target: Optional[int] = None
    pending_reason: str = ""

    def m2m(i: int) -> float:
        if pos == 0:
            return realized_equity
        return realized_equity + qty * pos * (c[i] - entry_px)

    def close_position(fill_px: float, fill_ts: int, reason: str) -> None:
        nonlocal pos, qty, entry_ts, entry_px, entry_bar_i, entry_fee, realized_equity, cum_pnl
        if pos == 0:
            return
        notional_exit = qty * fill_px
        fee_exit = notional_exit * fee_rate
        pnl_gross = qty * pos * (fill_px - entry_px)
        # realized_equity already includes entry_fee deduction
        realized_equity += pnl_gross - fee_exit
        pnl_net = pnl_gross - entry_fee - fee_exit
        cum_pnl += pnl_net
        trades.append(
            Trade(
                side="LONG" if pos == 1 else "SHORT",
                entry_ts=entry_ts,
                entry_price=entry_px,
                exit_ts=fill_ts,
                exit_price=fill_px,
                pnl=pnl_net,
                exit_reason=reason,
                cum_pnl=cum_pnl,
            )
        )
        pos = 0
        qty = 0.0
        entry_ts = 0
        entry_px = 0.0
        entry_bar_i = -1
        entry_fee = 0.0

    def open_position(new_pos: int, fill_px: float, fill_ts: int, bar_i: int) -> None:
        nonlocal pos, qty, entry_ts, entry_px, entry_bar_i, entry_fee, realized_equity
        if new_pos == 0:
            return
        # Size: percent of current realized equity (TV-like with cash sizing)
        notional = max(0.0, realized_equity * (percent_of_equity / 100.0))
        if notional <= 0:
            return
        if fill_px is None or fill_px <= 0:
            # Bad / missing price (can happen with sparse or dirty feeds). Skip opening.
            return
        qty = notional / fill_px
        fee_ent = notional * fee_rate
        realized_equity -= fee_ent
        pos = new_pos
        entry_ts = fill_ts
        entry_px = fill_px
        entry_bar_i = bar_i
        entry_fee = fee_ent

    # Iterate bars with "open -> intrabar stop -> close (schedule)" structure
    for i in range(len(bars)):
        # New day resets flip counter (UTC-based)
        d = datetime.utcfromtimestamp(ts[i] / 1000).date()
        if d != cur_day:
            cur_day = d
            flips_today = 0

        # 1) Execute pending actions at bar OPEN (orders from previous close)
        if i > 0 and pending_target is not None:
            target = pending_target
            reason = pending_reason
            pending_target = None
            pending_reason = ""

            # Close if needed
            if target == 0 and pos != 0:
                # Sell to close long / buy to close short at open with slippage
                fill_px = _slip(o[i], side=-1 if pos == 1 else +1, slippage_ticks=slippage_ticks, tick_size=tick_size)
                close_position(fill_px, ts[i], reason)

            # Reverse / enter
            if target != 0 and target != pos:
                # Close opposite first
                if pos != 0:
                    fill_px = _slip(o[i], side=-1 if pos == 1 else +1, slippage_ticks=slippage_ticks, tick_size=tick_size)
                    close_position(fill_px, ts[i], reason)

                # Open new
                if pos == 0:
                    fill_px = _slip(o[i], side=+1 if target == 1 else -1, slippage_ticks=slippage_ticks, tick_size=tick_size)
                    open_position(target, fill_px, ts[i], i)

        # 2) Emergency ATR stop (intrabar).
        # TradingView places/updates strategy.exit on bar close; with process_orders_on_close=false
        # the earliest realistic stop fill is the bar AFTER entry (not the same entry-open bar).
        if use_emergency_sl and pos != 0 and atr_v[i] is not None:
            if entry_bar_i != -1 and i == entry_bar_i:
                pass  # do not allow emergency stop on the same bar as entry
            else:
                atr_for_stop = atr_v[i]
                if pos == 1:
                    stop_px = entry_px - atr_for_stop * atr_mult
                    if l[i] <= stop_px:
                        # stop is a SELL
                        fill_px = _slip(stop_px, side=-1, slippage_ticks=slippage_ticks, tick_size=tick_size)
                        close_position(fill_px, ts[i], "STOP_LONG")
                else:
                    stop_px = entry_px + atr_for_stop * atr_mult
                    if h[i] >= stop_px:
                        # stop is a BUY
                        fill_px = _slip(stop_px, side=+1, slippage_ticks=slippage_ticks, tick_size=tick_size)
                        close_position(fill_px, ts[i], "STOP_SHORT")
# 3) Record equity at bar CLOSE (after any intrabar stops)
        equity_ts.append(ts[i])
        equity.append(m2m(i))

        # 4) At bar CLOSE, decide what to do NEXT bar OPEN
        if i == len(bars) - 1:
            break  # no next bar

        if st_dir[i] is None:
            continue

        # Pine: noTrade = useNoTrade and adxVal < threshold
        if no_trade[i]:
            if pos != 0:
                pending_target = 0
                pending_reason = "NO_TRADE"
            continue

        want_long = st_dir[i] == 1
        want_short = st_dir[i] == -1

        flip_to_long = want_long and (pos <= 0)
        flip_to_short = want_short and (pos >= 0)
        flip_request = flip_to_long or flip_to_short

        if not flip_request:
            continue

        # Cooldown and flip limit follow Pine semantics (based on bar OPEN time `time`)
        rev_ok = (
            (not use_rev_cooldown)
            or rev_cooldown_hrs == 0
            or last_flip_time is None
            or (ts[i] - last_flip_time >= cooldown_ms)
        )
        flip_limit_ok = (not use_flip_limit) or (flips_today < max_flips_per_day)

        if not (rev_ok and flip_limit_ok):
            continue

        pending_target = 1 if flip_to_long else -1
        pending_reason = "ST_FLIP"
        last_flip_time = ts[i]
        flips_today += 1

    # Optional: close at end (at last bar CLOSE is not TV-like; TV keeps it open).
    open_position: Optional[dict] = None
    if pos != 0:
        last_i = len(bars) - 1
        cur_px = c[last_i]
        unreal = qty * pos * (cur_px - entry_px)
        if close_at_end:
            # close at last bar CLOSE (approx; TV usually leaves it open)
            fill_px = _slip(cur_px, side=-1 if pos == 1 else +1, slippage_ticks=slippage_ticks, tick_size=tick_size)
            close_position(fill_px, ts[last_i], "END")
            if equity:
                equity[-1] = realized_equity
        else:
            open_position = {
                "side": "LONG" if pos == 1 else "SHORT",
                "entry_ts": entry_ts,
                "entry_price": entry_px,
                "current_ts": ts[last_i],
                "current_price": cur_px,
                "unrealized_pnl": unreal,
                "reason": "OPEN",
            }
            if equity:
                equity[-1] = realized_equity + unreal

    return BacktestResult(trades, open_position, equity_ts, equity, st_line, st_dir, adx_v, no_trade)

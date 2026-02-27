from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from typing import List, Tuple, Optional, Literal


Bar = Tuple[int, float, float, float, float, float]  # (ts_ms,o,h,l,c,v)
Side = Literal["LONG", "SHORT"]


def _round_px_worst(px: float, tick: float, side: str) -> float:
    """Round price to tick size in a direction that is worst for the trader.
    side: "BUY" or "SELL".
    """
    if tick <= 0:
        return px
    q = px / tick
    if side == "BUY":
        return math.ceil(q) * tick
    else:
        return math.floor(q) * tick


def _apply_fill_px(px: float, side: str, slippage_ticks: int, spread_ticks: float, tick: float) -> float:
    """Apply spread (half on each side) + slippage (per side) and round to tick.
    side: "BUY" or "SELL".
    """
    if tick <= 0:
        return px
    half_spread = max(0.0, float(spread_ticks)) / 2.0
    slip = max(0.0, int(slippage_ticks))
    ticks = half_spread + slip
    if side == "BUY":
        adj = px + ticks * tick
    else:
        adj = px - ticks * tick
    return _round_px_worst(adj, tick, side)


def _fee_usd(notional_usd: float, fee_percent: float) -> float:
    return abs(notional_usd) * (max(0.0, float(fee_percent)) / 100.0)


def _rma(values: List[float], n: int) -> List[Optional[float]]:
    """Wilder's RMA (TradingView ta.rma)."""
    out: List[Optional[float]] = [None] * len(values)
    if n <= 0 or not values:
        return out
    alpha = 1.0 / n
    prev: Optional[float] = None
    for i, v in enumerate(values):
        if prev is None:
            prev = v
        else:
            prev = prev + alpha * (v - prev)
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


@dataclass
class Trade:
    side: Side
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float  # USD
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


def backtest_strategy3(
    bars: List[Bar],
    position_usd: float = 1000.0,
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
    min_hold_bars: int = 0,
    # Execution frictions (used in /chart only; optimizer leaves defaults)
    slippage_ticks: int = 0,
    tick_size: float = 0.0001,
    fee_percent: float = 0.0,          # per side, in percent (e.g. 0.06)
    spread_ticks: float = 0.0,         # full bid-ask spread in ticks
    funding_8h_percent: float = 0.0,   # funding per 8h in percent (perp)
    sl_atr_len: Optional[int] = None,    # backward compat
    sl_atr_mult: Optional[float] = None, # backward compat
    close_at_end: bool = False,
    leverage_mult: Optional[float] = None,
    trade_from_current_capital: bool = False,
    capital_usd: float = 10000.0,
    # Realism option:
    # If True, signals are confirmed on candle close (t-1) and executed on next candle open (t).
    confirm_on_close: bool = False,
) -> BacktestResult:
    if not bars:
        return BacktestResult([], None, [], [], [], [], [], [])

    ts = [b[0] for b in bars]
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]

    st_line, st_dir = supertrend(h, l, c, st_factor, st_atr_len)
    adx_v = adx(h, l, c, adx_len, adx_smooth)
    if sl_atr_len is not None:
        atr_len = sl_atr_len
    if sl_atr_mult is not None:
        atr_mult = sl_atr_mult

    atr_v = atr(h, l, c, atr_len)

    no_trade: List[bool] = [False] * len(bars)
    for i in range(len(bars)):
        if not use_no_trade:
            no_trade[i] = False
        else:
            av = adx_v[i]
            no_trade[i] = (av is not None) and (av < adx_no_trade_below)

    trades: List[Trade] = []
    equity_ts: List[int] = []
    equity: List[float] = []

    FUNDING_MS = 8 * 60 * 60 * 1000

    pos = 0  # 1 long, -1 short, 0 flat
    entry_ts = 0
    entry_price = 0.0
    entry_fee = 0.0
    realized = 0.0
    cum = 0.0


    # Position sizing
    # - Legacy mode: fixed `position_usd` (when leverage_mult is None)
    # - Leverage mode: position_usd = base_capital * leverage_mult
    #   where base_capital is either initial capital_usd or (capital_usd + realized) if trade_from_current_capital.
    entry_size_usd = float(position_usd)
    entry_i = -1

    def _calc_entry_size_usd() -> float:
        if leverage_mult is None:
            return float(position_usd)
        try:
            lm = float(leverage_mult)
        except Exception:
            return float(position_usd)
        if lm < 0:
            lm = 0.0
        base = float(capital_usd)
        if trade_from_current_capital:
            base = max(0.0, float(capital_usd) + float(realized))
        return base * lm
    last_flip_time: Optional[int] = None
    cooldown_ms = int(rev_cooldown_hrs * 60 * 60 * 1000)

    flips_today = 0
    cur_day = datetime.utcfromtimestamp(ts[0] / 1000).date()

    def mark_to_market(i: int) -> float:
        # Be defensive: some data sources may occasionally provide zero/invalid prices.
        # In that case, avoid division by zero and treat position as not markable.
        if pos == 0 or entry_price <= 0:
            return realized
        if c[i] <= 0:
            return realized
        unreal = entry_size_usd * pos * (c[i] - entry_price) / entry_price
        return realized + unreal

    for i in range(len(bars)):
        # new day resets flip counter
        d = datetime.utcfromtimestamp(ts[i] / 1000).date()
        if d != cur_day:
            cur_day = d
            flips_today = 0

        # Funding (perp) every 8 hours (00:00/08:00/16:00 UTC buckets)
        if pos != 0 and funding_8h_percent != 0.0 and i > 0:
            prev_bucket = ts[i - 1] // FUNDING_MS
            cur_bucket = ts[i] // FUNDING_MS
            if cur_bucket > prev_bucket:
                # apply funding for each boundary crossed
                periods = int(cur_bucket - prev_bucket)
                rate = float(funding_8h_percent) / 100.0
                # long pays when rate>0, short receives
                sign = -1.0 if pos == 1 else 1.0
                funding_pnl = sign * entry_size_usd * rate * periods
                realized += funding_pnl
                cum += funding_pnl

        equity_ts.append(ts[i])
        equity.append(mark_to_market(i))

        # In "confirm_on_close" mode we only act on information from the previous candle.
        # Signals are evaluated on (i-1) and executed on candle i open.
        if confirm_on_close:
            if i == 0:
                continue
            sig_i = i - 1
            px_exec = o[i]
        else:
            sig_i = i
            px_exec = c[i]

        # 1) Emergency stop (intrabar). If triggered, we exit and stay flat for this bar.
        if use_emergency_sl and pos != 0 and atr_v[sig_i] is not None:
            # If entry price is invalid, flatten without blowing up.
            if entry_price <= 0:
                pos = 0
                equity[-1] = realized
                continue
            if pos == 1:
                stop_px = entry_price - atr_v[sig_i] * atr_mult
                if l[i] <= stop_px:
                    exit_side = "SELL"
                    exit_px = _apply_fill_px(stop_px, exit_side, slippage_ticks, spread_ticks, tick_size)
                    pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
                    exit_fee = _fee_usd(entry_size_usd, fee_percent)
                    pnl_net = pnl - exit_fee
                    realized += pnl_net
                    cum += pnl_net
                    trades.append(
                        Trade("LONG", entry_ts, entry_price, ts[i], exit_px, pnl_net, "STOP_LONG", cum)
                    )
                    pos = 0
                    entry_i = -1
                    # update equity point after forced close
                    equity[-1] = realized
                    continue
            else:  # pos == -1
                stop_px = entry_price + atr_v[sig_i] * atr_mult
                if h[i] >= stop_px:
                    exit_side = "BUY"
                    exit_px = _apply_fill_px(stop_px, exit_side, slippage_ticks, spread_ticks, tick_size)
                    pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
                    exit_fee = _fee_usd(entry_size_usd, fee_percent)
                    pnl_net = pnl - exit_fee
                    realized += pnl_net
                    cum += pnl_net
                    trades.append(
                        Trade("SHORT", entry_ts, entry_price, ts[i], exit_px, pnl_net, "STOP_SHORT", cum)
                    )
                    pos = 0
                    entry_i = -1
                    equity[-1] = realized
                    continue

        # 2) NO-TRADE closes any position and stays flat
        if no_trade[sig_i]:
            if pos != 0:
                px = px_exec
                if entry_price <= 0 or px <= 0:
                    # Bad tick; just flatten.
                    pos = 0
                    equity[-1] = realized
                    continue
                exit_side = "SELL" if pos == 1 else "BUY"
                exit_px = _apply_fill_px(px, exit_side, slippage_ticks, spread_ticks, tick_size)
                pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
                exit_fee = _fee_usd(entry_size_usd, fee_percent)
                pnl_net = pnl - exit_fee
                realized += pnl_net
                cum += pnl_net
                trades.append(
                    Trade("LONG" if pos == 1 else "SHORT", entry_ts, entry_price, ts[i], exit_px, pnl_net, "NO_TRADE", cum)
                )
                pos = 0
                entry_i = -1
                equity[-1] = realized
            continue

        # 3) Entries / reversals by Supertrend direction
        if st_dir[sig_i] is None:
            continue

        want_long = st_dir[sig_i] == 1
        want_short = st_dir[sig_i] == -1

        flip_to_long = want_long and (pos <= 0)
        flip_to_short = want_short and (pos >= 0)
        flip_request = flip_to_long or flip_to_short

        if not flip_request:
            continue

        rev_ok = (not use_rev_cooldown) or cooldown_ms == 0 or last_flip_time is None or (ts[i] - last_flip_time >= cooldown_ms)
        flip_limit_ok = (not use_flip_limit) or (flips_today < max_flips_per_day)
        if not (rev_ok and flip_limit_ok):
            continue

        px = px_exec
        # Never open/close on an invalid price.
        if px <= 0:
            continue

        # close opposite if needed
        hold_ok = (pos == 0) or (entry_i < 0) or ((i - entry_i) >= max(0, int(min_hold_bars)))

        if flip_to_long and pos == -1:
            if not hold_ok:
                continue
            if entry_price <= 0:
                pos = 0
                entry_i = -1
                equity[-1] = realized
                continue
            exit_side = "BUY"
            exit_px = _apply_fill_px(px, exit_side, slippage_ticks, spread_ticks, tick_size)
            pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
            exit_fee = _fee_usd(entry_size_usd, fee_percent)
            pnl_net = pnl - exit_fee
            realized += pnl_net
            cum += pnl_net
            trades.append(Trade("SHORT", entry_ts, entry_price, ts[i], exit_px, pnl_net, "ST_FLIP", cum))
            pos = 0
            entry_i = -1
            equity[-1] = realized

        if flip_to_short and pos == 1:
            if not hold_ok:
                continue
            if entry_price <= 0:
                pos = 0
                entry_i = -1
                equity[-1] = realized
                continue
            exit_side = "SELL"
            exit_px = _apply_fill_px(px, exit_side, slippage_ticks, spread_ticks, tick_size)
            pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
            exit_fee = _fee_usd(entry_size_usd, fee_percent)
            pnl_net = pnl - exit_fee
            realized += pnl_net
            cum += pnl_net
            trades.append(Trade("LONG", entry_ts, entry_price, ts[i], exit_px, pnl_net, "ST_FLIP", cum))
            pos = 0
            entry_i = -1
            equity[-1] = realized

        # open new
        if pos == 0:
            entry_size_usd = _calc_entry_size_usd()
            pos = 1 if flip_to_long else -1
            entry_ts = ts[i]
            # Entry fill with spread+slippage
            entry_side = "BUY" if pos == 1 else "SELL"
            entry_price = _apply_fill_px(px, entry_side, slippage_ticks, spread_ticks, tick_size)
            # Entry fee charged immediately
            entry_fee = _fee_usd(entry_size_usd, fee_percent)
            realized -= entry_fee
            cum -= entry_fee
            entry_i = i
            last_flip_time = ts[i]
            flips_today += 1

    open_position: Optional[dict] = None
    if pos != 0:
        last_i = len(bars) - 1
        cur_px = c[last_i]

        # If we somehow ended up with an invalid entry/price, just report flat.
        if entry_price <= 0 or cur_px <= 0:
            pos = 0
            open_position = None
            if equity:
                equity[-1] = realized
            return BacktestResult(trades, open_position, equity_ts, equity, st_line, st_dir, adx_v, no_trade)

        if close_at_end:
            # Force-close the position on the last candle (useful for pure backtests).
            exit_side = "SELL" if pos == 1 else "BUY"
            exit_px = _apply_fill_px(cur_px, exit_side, slippage_ticks, spread_ticks, tick_size)
            pnl = entry_size_usd * pos * (exit_px - entry_price) / entry_price
            exit_fee = _fee_usd(entry_size_usd, fee_percent)
            pnl_net = pnl - exit_fee
            realized += pnl_net
            cum += pnl_net
            trades.append(
                Trade(
                    "LONG" if pos == 1 else "SHORT",
                    entry_ts,
                    entry_price,
                    ts[last_i],
                    exit_px,
                    pnl_net,
                    "END",
                    cum,
                )
            )
            pos = 0
            entry_i = -1
            if equity:
                equity[-1] = realized
        else:
            # Keep it as an open position (to match TradingView "open" row).
            unreal = entry_size_usd * pos * (cur_px - entry_price) / entry_price
            open_position = {
                "side": "LONG" if pos == 1 else "SHORT",
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "current_ts": ts[last_i],
                "current_price": cur_px,
                "unrealized_pnl": unreal,
                "reason": "OPEN",
            }
            if equity:
                equity[-1] = realized + unreal

    return BacktestResult(trades, open_position, equity_ts, equity, st_line, st_dir, adx_v, no_trade)

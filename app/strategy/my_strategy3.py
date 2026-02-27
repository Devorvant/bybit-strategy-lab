# app/strategy/my_strategy3.py

"""Always-in-market L/S with Supertrend + ADX NO-TRADE + reverse cooldown + ATR emergency stop.

Python adaptation of the user-provided TradingView Pine strategy:
"Always-in-Market L/S v2 (Railway): Supertrend + ADX NoTrade + Reverse Cooldown".

This module is primarily used by the chart/backtest selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


Bar = Tuple[int, float, float, float, float, float]  # (ts_ms,o,h,l,c,v)


@dataclass
class SignalResult:
    ts: int
    signal: str  # "LONG" / "SHORT" / "FLAT"
    note: str = ""


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


def _atr(high: List[float], low: List[float], close: List[float], n: int) -> List[Optional[float]]:
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


def _adx(
    high: List[float],
    low: List[float],
    close: List[float],
    di_len: int,
    adx_smooth: int,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """TradingView-like DMI/ADX: ta.dmi(di_len, adx_smooth)."""
    n = len(close)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr_rma = _rma(tr, di_len)
    plus_rma = _rma(plus_dm, di_len)
    minus_rma = _rma(minus_dm, di_len)

    di_plus: List[Optional[float]] = [None] * n
    di_minus: List[Optional[float]] = [None] * n
    dx: List[float] = [0.0] * n

    for i in range(n):
        a = atr_rma[i]
        if a is None or a == 0:
            continue
        p = (plus_rma[i] or 0.0) * 100.0 / a
        m = (minus_rma[i] or 0.0) * 100.0 / a
        di_plus[i] = p
        di_minus[i] = m
        denom = p + m
        dx[i] = 0.0 if denom == 0 else 100.0 * abs(p - m) / denom

    adx = _rma(dx, adx_smooth)
    return di_plus, di_minus, adx


def _supertrend(
    high: List[float],
    low: List[float],
    close: List[float],
    atr_len: int,
    factor: float,
) -> Tuple[List[Optional[float]], List[Optional[int]]]:
    """Classic Supertrend.

    Returns (st_line, st_dir) where st_dir is +1 (up) / -1 (down).
    """
    n = len(close)
    atr = _atr(high, low, close, atr_len)

    st_line: List[Optional[float]] = [None] * n
    st_dir: List[Optional[int]] = [None] * n
    fub: List[Optional[float]] = [None] * n
    flb: List[Optional[float]] = [None] * n

    for i in range(n):
        if atr[i] is None:
            continue
        hl2 = (high[i] + low[i]) / 2.0
        ub = hl2 + factor * atr[i]
        lb = hl2 - factor * atr[i]
        if i == 0:
            fub[i] = ub
            flb[i] = lb
            st_dir[i] = 1
            st_line[i] = lb
            continue

        # final upper/lower band
        prev_fub = fub[i - 1] if fub[i - 1] is not None else ub
        prev_flb = flb[i - 1] if flb[i - 1] is not None else lb

        fub[i] = ub if (ub < prev_fub or close[i - 1] > prev_fub) else prev_fub
        flb[i] = lb if (lb > prev_flb or close[i - 1] < prev_flb) else prev_flb

        prev_dir = st_dir[i - 1] if st_dir[i - 1] is not None else 1
        if prev_dir == 1:
            st_dir[i] = -1 if close[i] < (flb[i] or lb) else 1
        else:
            st_dir[i] = 1 if close[i] > (fub[i] or ub) else -1

        st_line[i] = flb[i] if st_dir[i] == 1 else fub[i]

    return st_line, st_dir


def _generate_signal_legacy(
    bars: List[Bar],
    *,
    use_no_trade: bool = True,
    adx_len: int = 14,
    adx_smooth: int = 14,
    adx_no_trade_below: float = 14.0,
    st_atr_len: int = 14,
    st_factor: float = 4.0,
    use_rev_cooldown: bool = True,
    rev_cooldown_hrs: int = 8,
) -> Optional[SignalResult]:
    """Compute the latest signal.

    Returns FLAT if NO-TRADE or cooldown blocks entries; otherwise LONG/SHORT
    based on current Supertrend direction.
    """
    if len(bars) < max(100, st_atr_len * 3, adx_len * 3):
        return None

    ts = [b[0] for b in bars]
    high = [b[2] for b in bars]
    low = [b[3] for b in bars]
    close = [b[4] for b in bars]

    _di_p, _di_m, adx = _adx(high, low, close, adx_len, adx_smooth)
    _st_line, st_dir = _supertrend(high, low, close, st_atr_len, st_factor)

    i = len(bars) - 1
    if st_dir[i] is None:
        return None

    no_trade = bool(use_no_trade and (adx[i] is not None and adx[i] < adx_no_trade_below))
    if no_trade:
        return SignalResult(ts=ts[i], signal="FLAT", note=f"NO_TRADE (ADX<{adx_no_trade_below})")

    # Cooldown can't be fully inferred from bars without trade state.
    # For chart/backtest we enforce it in backtest. Here we conservatively ignore.
    if use_rev_cooldown and rev_cooldown_hrs > 0:
        # We don't track lastFlipTime here; return directional bias.
        pass

    return SignalResult(
        ts=ts[i],
        signal="LONG" if st_dir[i] == 1 else "SHORT",
        note="Supertrend direction",
    )



def generate_signal_via_backtest(
    bars: List[Bar],
    *,
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
    confirm_on_close: bool = False,
) -> Optional[SignalResult]:
    """Compute latest signal via strategy3 backtest core.

    This uses the same stateful logic as optimizer/backtest and returns the
    current directional state as LONG/SHORT/FLAT.
    """
    if len(bars) < max(100, st_atr_len * 3, adx_len * 3):
        return None

    from app.backtest.strategy3_backtest import backtest_strategy3  # local import to avoid cycles

    bt = backtest_strategy3(
        bars,
        position_usd=1000.0,
        use_no_trade=use_no_trade,
        adx_len=adx_len,
        adx_smooth=adx_smooth,
        adx_no_trade_below=adx_no_trade_below,
        st_atr_len=st_atr_len,
        st_factor=st_factor,
        use_rev_cooldown=use_rev_cooldown,
        rev_cooldown_hrs=rev_cooldown_hrs,
        use_flip_limit=use_flip_limit,
        max_flips_per_day=max_flips_per_day,
        use_emergency_sl=use_emergency_sl,
        atr_len=atr_len,
        atr_mult=atr_mult,
        min_hold_bars=min_hold_bars,
        close_at_end=False,
        confirm_on_close=confirm_on_close,
    )

    pos = getattr(bt, 'open_position', None)
    last_ts = bars[-1][0]
    if not pos:
        note = 'Backtest core: FLAT'
        trades = list(getattr(bt, 'trades', []) or [])
        if trades:
            try:
                note = f"Backtest core: last exit {getattr(trades[-1], 'exit_reason', 'FLAT')}"
            except Exception:
                pass
        return SignalResult(ts=last_ts, signal='FLAT', note=note)

    side = str(pos.get('side') if isinstance(pos, dict) else getattr(pos, 'side', '')).upper()
    if side in ('LONG', 'BUY'):
        return SignalResult(ts=last_ts, signal='LONG', note='Backtest core open_position')
    if side in ('SHORT', 'SELL'):
        return SignalResult(ts=last_ts, signal='SHORT', note='Backtest core open_position')
    return SignalResult(ts=last_ts, signal='FLAT', note='Backtest core unknown side')



def generate_signal(*args, **kwargs):
    """Primary signal entrypoint: use unified backtest-core logic."""
    return generate_signal_via_backtest(*args, **kwargs)

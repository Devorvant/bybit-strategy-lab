from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from app.backtest.sma_backtest import BacktestResult, Bar, Trade, sma


def _adx(
    bars: Sequence[Bar],
    n: int = 14,
) -> List[Optional[float]]:
    """Wilder's ADX.

    Returns list of length len(bars) with None until enough data.
    """

    out: List[Optional[float]] = [None] * len(bars)
    if n <= 0 or len(bars) < n + 1:
        return out

    high = [b[2] for b in bars]
    low = [b[3] for b in bars]
    close = [b[4] for b in bars]

    tr: List[float] = [0.0] * len(bars)
    plus_dm: List[float] = [0.0] * len(bars)
    minus_dm: List[float] = [0.0] * len(bars)

    for i in range(1, len(bars)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder smoothing
    tr_s: List[Optional[float]] = [None] * len(bars)
    plus_s: List[Optional[float]] = [None] * len(bars)
    minus_s: List[Optional[float]] = [None] * len(bars)

    tr_sum = sum(tr[1 : n + 1])
    plus_sum = sum(plus_dm[1 : n + 1])
    minus_sum = sum(minus_dm[1 : n + 1])

    tr_s[n] = tr_sum
    plus_s[n] = plus_sum
    minus_s[n] = minus_sum

    for i in range(n + 1, len(bars)):
        tr_sum = tr_sum - (tr_sum / n) + tr[i]
        plus_sum = plus_sum - (plus_sum / n) + plus_dm[i]
        minus_sum = minus_sum - (minus_sum / n) + minus_dm[i]

        tr_s[i] = tr_sum
        plus_s[i] = plus_sum
        minus_s[i] = minus_sum

    # DI and DX
    dx: List[Optional[float]] = [None] * len(bars)
    for i in range(n, len(bars)):
        if tr_s[i] is None or tr_s[i] == 0:
            continue
        pdi = 100.0 * (plus_s[i] or 0.0) / tr_s[i]
        mdi = 100.0 * (minus_s[i] or 0.0) / tr_s[i]
        denom = pdi + mdi
        if denom == 0:
            continue
        dx[i] = 100.0 * abs(pdi - mdi) / denom

    # ADX (smoothed DX)
    # First ADX is the average of DX over the next n values
    first_adx_idx = n * 2
    if len(bars) <= first_adx_idx:
        return out

    first_vals = [v for v in dx[n : first_adx_idx + 1] if v is not None]
    if len(first_vals) < n:
        return out

    adx_val = sum(first_vals) / n
    out[first_adx_idx] = adx_val

    for i in range(first_adx_idx + 1, len(bars)):
        if dx[i] is None:
            continue
        adx_val = ((adx_val * (n - 1)) + dx[i]) / n
        out[i] = adx_val

    return out


def backtest_sma_adx_filter(
    bars: List[Bar],
    *,
    position_usd: float = 1000.0,
    fast_n: int = 20,
    slow_n: int = 50,
    adx_n: int = 14,
    adx_enter: float = 20.0,
    adx_exit: float = 18.0,
    stop_pct: float = 0.03,
    close_at_end: bool = True,
) -> BacktestResult:
    """SMA-cross + ADX filter.

    - Вход только если ADX >= adx_enter
    - Выход в кэш (flat) если ADX < adx_exit
    - Стоп фиксированный в % от входа
    """

    if not bars:
        return BacktestResult([], [], [], [], [])

    ts = [b[0] for b in bars]
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]

    fast = sma(c, fast_n)
    slow = sma(c, slow_n)
    adx = _adx(bars, adx_n)

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
        unreal = position_usd * pos * (c[i] - entry_price) / entry_price
        return realized + unreal

    def stop_price() -> float:
        if pos == 1:
            return entry_price * (1.0 - stop_pct)
        return entry_price * (1.0 + stop_pct)

    for i in range(len(bars)):
        equity_ts.append(ts[i])
        equity.append(m2m(i))

        if i == 0:
            continue

        # --- STOP ---
        if pos != 0 and stop_pct > 0:
            sp = stop_price()
            hit = (pos == 1 and l[i] <= sp) or (pos == -1 and h[i] >= sp)
            if hit:
                exit_px = sp
                t = ts[i]
                pnl = position_usd * pos * (exit_px - entry_price) / entry_price
                realized += pnl
                trades.append(
                    Trade(
                        "LONG" if pos == 1 else "SHORT",
                        entry_ts,
                        entry_price,
                        t,
                        exit_px,
                        pnl,
                        exit_reason="STOP",
                    )
                )
                pos = 0
                if equity:
                    equity[-1] = realized
                # после стопа не входим в ту же свечу
                continue

        # --- FILTER EXIT ---
        if pos != 0 and adx[i] is not None and adx[i] < adx_exit:
            exit_px = c[i]
            t = ts[i]
            pnl = position_usd * pos * (exit_px - entry_price) / entry_price
            realized += pnl
            trades.append(
                Trade(
                    "LONG" if pos == 1 else "SHORT",
                    entry_ts,
                    entry_price,
                    t,
                    exit_px,
                    pnl,
                    exit_reason="FILTER",
                )
            )
            pos = 0
            if equity:
                equity[-1] = realized

        # --- SIGNALS ---
        if fast[i] is None or slow[i] is None or fast[i - 1] is None or slow[i - 1] is None:
            continue

        cross_up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
        cross_dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]

        can_enter = adx[i] is not None and adx[i] >= adx_enter

        px = c[i]
        t = ts[i]

        if cross_up:
            if pos == -1:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(Trade("SHORT", entry_ts, entry_price, t, px, pnl, exit_reason="CROSS"))
                pos = 0
            if pos == 0 and can_enter:
                pos = 1
                entry_ts = t
                entry_price = px

        elif cross_dn:
            if pos == 1:
                pnl = position_usd * pos * (px - entry_price) / entry_price
                realized += pnl
                trades.append(Trade("LONG", entry_ts, entry_price, t, px, pnl, exit_reason="CROSS"))
                pos = 0
            if pos == 0 and can_enter:
                pos = -1
                entry_ts = t
                entry_price = px

        if equity:
            equity[-1] = m2m(i)

    if close_at_end and pos != 0:
        px = c[-1]
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

    return BacktestResult(trades, equity_ts, equity, fast, slow)

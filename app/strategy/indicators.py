import numpy as np


def rma(x, length: int):
    """Wilder's RMA (smoothed moving average).

    This is the smoothing used in ATR/ADX in most charting platforms.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if length <= 0 or len(x) == 0:
        return out

    if len(x) >= length:
        # initialize with the first SMA
        out[length - 1] = np.nanmean(x[:length])
        for i in range(length, len(x)):
            out[i] = (out[i - 1] * (length - 1) + x[i]) / length
    return out

def atr(high, low, close, length=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    out = np.full_like(close, np.nan, dtype=float)
    if len(close) >= length:
        out[length - 1] = np.mean(tr[:length])
        for i in range(length, len(tr)):
            out[i] = (out[i - 1] * (length - 1) + tr[i]) / length
    return out


def adx(high, low, close, length: int = 14):
    """Average Directional Index (ADX) + DI+/DI-.

    Returns (adx, plus_di, minus_di) as np.ndarrays.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)

    if n == 0:
        empty = np.asarray([])
        return empty, empty, empty

    prev_high = np.r_[high[0], high[:-1]]
    prev_low = np.r_[low[0], low[:-1]]
    prev_close = np.r_[close[0], close[:-1]]

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    tr_rma = rma(tr, length)
    plus_dm_rma = rma(plus_dm, length)
    minus_dm_rma = rma(minus_dm, length)

    plus_di = 100.0 * (plus_dm_rma / tr_rma)
    minus_di = 100.0 * (minus_dm_rma / tr_rma)

    denom = plus_di + minus_di
    dx = 100.0 * np.where(denom == 0, 0.0, np.abs(plus_di - minus_di) / denom)
    adx_ = rma(dx, length)
    return adx_, plus_di, minus_di


def supertrend(high, low, close, atr_length: int = 10, factor: float = 3.0):
    """Supertrend indicator.

    Returns (supertrend_line, direction), where direction is 1 (bull) or -1 (bear).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    st = np.full(n, np.nan, dtype=float)
    dir_ = np.full(n, np.nan, dtype=float)

    if n == 0:
        return st, dir_

    atr_ = atr(high, low, close, length=atr_length)
    hl2 = (high + low) / 2.0
    upper = hl2 + factor * atr_
    lower = hl2 - factor * atr_

    f_upper = np.full(n, np.nan, dtype=float)
    f_lower = np.full(n, np.nan, dtype=float)

    # start after ATR is available
    start = atr_length - 1
    if start < 0 or start >= n:
        return st, dir_

    f_upper[start] = upper[start]
    f_lower[start] = lower[start]
    dir_[start] = 1.0
    st[start] = f_lower[start]

    for i in range(start + 1, n):
        # final upper band
        if np.isnan(f_upper[i - 1]) or np.isnan(upper[i]):
            f_upper[i] = upper[i]
        else:
            f_upper[i] = upper[i] if (upper[i] < f_upper[i - 1] or close[i - 1] > f_upper[i - 1]) else f_upper[i - 1]

        # final lower band
        if np.isnan(f_lower[i - 1]) or np.isnan(lower[i]):
            f_lower[i] = lower[i]
        else:
            f_lower[i] = lower[i] if (lower[i] > f_lower[i - 1] or close[i - 1] < f_lower[i - 1]) else f_lower[i - 1]

        # direction switch
        # Match TradingView ta.supertrend logic:
        # - If we were bearish (-1), flip to bullish (+1) only when
        #   close crosses ABOVE the *previous* final upper band.
        # - If we were bullish (+1), flip to bearish (-1) only when
        #   close crosses BELOW the *previous* final lower band.
        prev_dir = dir_[i - 1]
        if prev_dir == -1.0 and close[i] > f_upper[i - 1]:
            dir_[i] = 1.0
        elif prev_dir == 1.0 and close[i] < f_lower[i - 1]:
            dir_[i] = -1.0
        else:
            dir_[i] = prev_dir

        st[i] = f_lower[i] if dir_[i] == 1.0 else f_upper[i]

    return st, dir_

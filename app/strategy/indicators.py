import numpy as np

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

# ADX / Supertrend добавим следующим шагом.

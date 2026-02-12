from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query

from app.storage.db import load_bars

router = APIRouter()


# ---------------- TradingView-like math (Wilder RMA init via SMA) ----------------
def _rma_tv(values: List[float], length: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if length <= 0 or len(values) < length:
        return out

    sma = sum(values[:length]) / length
    out[length - 1] = sma
    prev = sma
    for i in range(length, len(values)):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def _atr_tv(h: List[float], l: List[float], c: List[float], length: int) -> List[Optional[float]]:
    tr: List[float] = []
    for i in range(len(c)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return _rma_tv(tr, length)


def _dmi_tv(h: List[float], l: List[float], c: List[float], di_len: int, adx_smooth: int):
    n = len(c)
    up_move = [0.0] * n
    down_move = [0.0] * n
    tr = [0.0] * n

    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            up = h[i] - h[i - 1]
            dn = l[i - 1] - l[i]
            up_move[i] = up if (up > dn and up > 0) else 0.0
            down_move[i] = dn if (dn > up and dn > 0) else 0.0
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    tr_rma = _rma_tv(tr, di_len)
    plus_rma = _rma_tv(up_move, di_len)
    minus_rma = _rma_tv(down_move, di_len)

    di_plus: List[Optional[float]] = [None] * n
    di_minus: List[Optional[float]] = [None] * n
    dx = [0.0] * n

    for i in range(n):
        if tr_rma[i] is None or tr_rma[i] == 0 or plus_rma[i] is None or minus_rma[i] is None:
            continue
        p = 100.0 * (plus_rma[i] / tr_rma[i])
        m = 100.0 * (minus_rma[i] / tr_rma[i])
        di_plus[i] = p
        di_minus[i] = m
        denom = p + m
        dx[i] = 0.0 if denom == 0 else (100.0 * abs(p - m) / denom)

    adx = _rma_tv(dx, adx_smooth)
    return di_plus, di_minus, adx


def _supertrend_tv(h: List[float], l: List[float], c: List[float], factor: float, atr_len: int):
    n = len(c)
    atr = _atr_tv(h, l, c, atr_len)

    final_upper: List[Optional[float]] = [None] * n
    final_lower: List[Optional[float]] = [None] * n
    st_dir: List[Optional[int]] = [None] * n
    st_line: List[Optional[float]] = [None] * n

    for i in range(n):
        if atr[i] is None:
            continue
        hl2 = (h[i] + l[i]) / 2.0
        basic_upper = hl2 + factor * atr[i]
        basic_lower = hl2 - factor * atr[i]

        if i == 0 or final_upper[i - 1] is None or final_lower[i - 1] is None:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
        else:
            # TV band finalization
            final_upper[i] = basic_upper if (basic_upper < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
            final_lower[i] = basic_lower if (basic_lower > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]) else final_lower[i - 1]

        # TV flip uses PREVIOUS finalized bands
        if i == 0 or st_dir[i - 1] is None:
            st_dir[i] = 1
        else:
            prev_dir = st_dir[i - 1]
            prev_fu = final_upper[i - 1]
            prev_fl = final_lower[i - 1]
            if prev_dir == 1 and prev_fl is not None and c[i] < prev_fl:
                st_dir[i] = -1
            elif prev_dir == -1 and prev_fu is not None and c[i] > prev_fu:
                st_dir[i] = 1
            else:
                st_dir[i] = prev_dir

        st_line[i] = final_lower[i] if st_dir[i] == 1 else final_upper[i]
    return st_line, st_dir


# ---------------- FastAPI endpoint ----------------
@router.get("/tv_debug")
def tv_debug(
    symbol: str = Query("APTUSDT"),
    tf: str = Query("120"),
    limit: int = Query(200, ge=50, le=50000),
    adx_len: int = Query(14, ge=1),
    adx_smooth: int = Query(14, ge=1),
    adx_no_trade_below: float = Query(14.0),
    st_atr_len: int = Query(14, ge=1),
    st_factor: float = Query(4.0),
):
    """
    Returns TradingView-comparable series for debugging:
    time (OPEN ms), OHLC, ATR, DI+/DI-, ADX, Supertrend dir/line, noTrade flag.
    Uses the bars stored in DB (bars.ts is candle OPEN time in ms).
    """
    # main.py defines global 'conn'
    from app.main import conn  # noqa: WPS433 (simple local import)

    rows = load_bars(conn, symbol.upper(), str(tf), limit=limit)
    ts = [int(r[0]) for r in rows]
    o = [float(r[1]) for r in rows]
    h = [float(r[2]) for r in rows]
    l = [float(r[3]) for r in rows]
    c = [float(r[4]) for r in rows]

    atr = _atr_tv(h, l, c, st_atr_len)
    di_p, di_m, adx = _dmi_tv(h, l, c, adx_len, adx_smooth)
    st_line, st_dir = _supertrend_tv(h, l, c, st_factor, st_atr_len)

    out = []
    for i in range(len(ts)):
        adx_i = adx[i]
        out.append(
            {
                "time": ts[i],     # candle OPEN time (ms) - should match Pine `time`
                "o": o[i],
                "h": h[i],
                "l": l[i],
                "c": c[i],
                "atr": atr[i],
                "diPlus": di_p[i],
                "diMinus": di_m[i],
                "adx": adx_i,
                "stDir": st_dir[i],
                "stLine": st_line[i],
                "noTrade": (adx_i is not None and adx_i < adx_no_trade_below),
            }
        )
    return {"symbol": symbol.upper(), "tf": str(tf), "count": len(out), "rows": out}

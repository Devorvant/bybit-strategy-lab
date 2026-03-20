from __future__ import annotations

from typing import Any


def make_strategy_trade_key(symbol: str, tf: str, opt_id: Any, bar_ts: Any) -> str:
    sym = str(symbol or "").upper().strip()
    tfv = str(tf or "").strip()
    optv = str(opt_id or "").strip()
    ts = int(bar_ts or 0)
    return f"{sym}:{tfv}:{optv}:{ts}"

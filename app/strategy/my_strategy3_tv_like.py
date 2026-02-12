# app/strategy/my_strategy3_tv_like.py

"""
TradingView-like wrapper for Strategy 3 defaults.

This module exists mainly so the strategy selector can reference a real file,
and to keep the "Pine defaults" in one place.

Pine (your script):
- percent_of_equity = 50
- commission = 0.10%
- slippage = 2 ticks
- syminfo.mintick = 0.0001
"""

from __future__ import annotations

from typing import List, Optional

from app.strategy.my_strategy3 import Bar, SignalResult, generate_signal as _generate_signal

# ✅ from TradingView: syminfo.mintick
MINTICK: float = 0.0001

# Pine defaults for params (for reference / UI)
PERCENT_OF_EQUITY: float = 50.0
COMMISSION_PERCENT: float = 0.10
SLIPPAGE_TICKS: int = 2


def generate_signal(
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
    # Delegate to core implementation from my_strategy3.py
    return _generate_signal(
        bars,
        use_no_trade=use_no_trade,
        adx_len=adx_len,
        adx_smooth=adx_smooth,
        adx_no_trade_below=adx_no_trade_below,
        st_atr_len=st_atr_len,
        st_factor=st_factor,
        use_rev_cooldown=use_rev_cooldown,
        rev_cooldown_hrs=rev_cooldown_hrs,
    )

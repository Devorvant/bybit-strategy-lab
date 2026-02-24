
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict

from fastapi import HTTPException

from .bybit_client import BybitClient
from .sizing import calc_qty_from_position_usd

DEFAULT_LEVERAGE = int(os.getenv('TRADE_LEVERAGE', os.getenv('LEVERAGE', '3')))
DEFAULT_POSITION_USD = float(os.getenv('TRADE_POSITION_USD', '1000'))


@dataclass
class ExecutionResult:
    ok: bool
    action: str
    symbol: str
    qty: float | None = None
    position_before: float | None = None
    response: Dict[str, Any] | None = None
    note: str | None = None
    error: str | None = None
    leverage: int | None = None
    tp_price: float | None = None
    sl_price: float | None = None
    stop_response: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def execute_action(
    bybit: BybitClient,
    symbol: str,
    action: str,
    position_usd: float | None = None,
    leverage: int | None = None,
    sl_percent: float | None = None,
    tp_percent: float | None = None,
    ensure_isolated: bool = True,
) -> Dict[str, Any]:
    symbol = bybit.clean_symbol(symbol)
    action = (action or '').upper().strip()
    if action not in {'LONG', 'SHORT', 'CLOSE'}:
        raise HTTPException(status_code=400, detail=f'unsupported action: {action}')

    if not bybit.is_ready():
        return ExecutionResult(ok=False, action=action, symbol=symbol, note='bybit not configured').to_dict()

    lev = int(leverage or DEFAULT_LEVERAGE)
    pos_usd = float(position_usd or DEFAULT_POSITION_USD)
    sl_pct = float(sl_percent) if sl_percent not in (None, '') else None
    tp_pct = float(tp_percent) if tp_percent not in (None, '') else None

    try:
        if ensure_isolated:
            bybit.set_isolated_margin_mode()
        bybit.set_leverage(symbol, lev)

        pos = bybit.get_position_size(symbol)
        if action == 'CLOSE':
            resp = bybit.close_full_position(symbol)
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, response=resp, leverage=lev).to_dict()

        # Flip if needed
        if action == 'LONG' and pos < 0:
            bybit.close_full_position(symbol)
            time.sleep(0.2)
        elif action == 'SHORT' and pos > 0:
            bybit.close_full_position(symbol)
            time.sleep(0.2)
        elif action == 'LONG' and pos > 0:
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, note='already long', leverage=lev).to_dict()
        elif action == 'SHORT' and pos < 0:
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, note='already short', leverage=lev).to_dict()

        price = bybit.get_last_price(symbol)
        rules = bybit.get_qty_rules(symbol)
        qty = calc_qty_from_position_usd(pos_usd, price, rules)
        side = 'Buy' if action == 'LONG' else 'Sell'
        resp = bybit.place_market(symbol, side, qty, reduce_only=False)

        tp_price = None
        sl_price = None
        stop_resp = None
        if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0):
            entry_price = bybit.get_last_price(symbol) or price
            if action == 'LONG':
                if tp_pct is not None and tp_pct > 0:
                    tp_price = entry_price * (1 + tp_pct / 100.0)
                if sl_pct is not None and sl_pct > 0:
                    sl_price = entry_price * (1 - sl_pct / 100.0)
            else:
                if tp_pct is not None and tp_pct > 0:
                    tp_price = entry_price * (1 - tp_pct / 100.0)
                if sl_pct is not None and sl_pct > 0:
                    sl_price = entry_price * (1 + sl_pct / 100.0)
            stop_resp = bybit.set_trading_stop(symbol=symbol, take_profit=tp_price, stop_loss=sl_price)

        return ExecutionResult(
            ok=True,
            action=action,
            symbol=symbol,
            qty=qty,
            position_before=pos,
            response=resp,
            leverage=lev,
            tp_price=tp_price,
            sl_price=sl_price,
            stop_response=stop_resp,
        ).to_dict()
    except HTTPException:
        raise
    except Exception as e:
        return ExecutionResult(ok=False, action=action, symbol=symbol, error=repr(e)).to_dict()

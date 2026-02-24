
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def execute_action(
    bybit: BybitClient,
    symbol: str,
    action: str,
    position_usd: float | None = None,
    leverage: int | None = None,
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

    try:
        if ensure_isolated:
            bybit.set_isolated_margin_mode()
        bybit.set_leverage(symbol, lev)

        pos = bybit.get_position_size(symbol)
        if action == 'CLOSE':
            resp = bybit.close_full_position(symbol)
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, response=resp).to_dict()

        # Flip if needed
        if action == 'LONG' and pos < 0:
            bybit.close_full_position(symbol)
            time.sleep(0.2)
        elif action == 'SHORT' and pos > 0:
            bybit.close_full_position(symbol)
            time.sleep(0.2)
        elif action == 'LONG' and pos > 0:
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, note='already long').to_dict()
        elif action == 'SHORT' and pos < 0:
            return ExecutionResult(ok=True, action=action, symbol=symbol, position_before=pos, note='already short').to_dict()

        price = bybit.get_last_price(symbol)
        rules = bybit.get_qty_rules(symbol)
        qty = calc_qty_from_position_usd(pos_usd, price, rules)
        side = 'Buy' if action == 'LONG' else 'Sell'
        resp = bybit.place_market(symbol, side, qty, reduce_only=False)
        return ExecutionResult(ok=True, action=action, symbol=symbol, qty=qty, position_before=pos, response=resp).to_dict()
    except HTTPException:
        raise
    except Exception as e:
        return ExecutionResult(ok=False, action=action, symbol=symbol, error=repr(e)).to_dict()

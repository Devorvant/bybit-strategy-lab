
import math
from dataclasses import dataclass


@dataclass
class QtyRules:
    min_qty: float
    max_qty: float
    step: float


def round_down_to_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def fmt_number(x: float) -> str:
    s = f"{x:.10f}".rstrip('0').rstrip('.')
    return s if s else '0'


def calc_qty_from_position_usd(position_usd: float, price: float, rules: QtyRules) -> float:
    if position_usd <= 0 or price <= 0:
        raise ValueError(f'invalid position_usd/price: {position_usd}/{price}')
    raw_qty = position_usd / price
    qty = round_down_to_step(raw_qty, rules.step)
    if rules.max_qty > 0:
        qty = min(qty, rules.max_qty)
    if qty < rules.min_qty:
        raise ValueError(
            f'qty too small after rounding: raw={raw_qty}, step={rules.step}, rounded={qty}, minQty={rules.min_qty}'
        )
    return qty


def calc_qty_from_equity(equity: float, price: float, risk_fraction: float, leverage: int, rules: QtyRules) -> float:
    notional = equity * risk_fraction * leverage
    return calc_qty_from_position_usd(notional, price, rules)

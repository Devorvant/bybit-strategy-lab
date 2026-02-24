
import os
from typing import Any, Dict, Optional
from fastapi import HTTPException

try:
    from pybit.unified_trading import HTTP
except Exception:
    HTTP = None  # type: ignore

from .sizing import QtyRules, fmt_number

CATEGORY = os.getenv('BYBIT_CATEGORY', 'linear')
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', '1') != '0'
BYBIT_KEY = os.getenv('BYBIT_KEY', '')
BYBIT_SECRET = os.getenv('BYBIT_SECRET', '')


def _make_client() -> Optional['HTTP']:
    if HTTP is None or not (BYBIT_KEY and BYBIT_SECRET):
        return None
    return HTTP(testnet=BYBIT_TESTNET, api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)


class BybitClient:
    def __init__(self):
        self.client = _make_client()

    def is_ready(self) -> bool:
        return self.client is not None

    def ensure_ready(self):
        if self.client is None:
            raise HTTPException(status_code=500, detail='Bybit API not configured (set BYBIT_KEY/BYBIT_SECRET and install pybit)')

    def clean_symbol(self, sym: str) -> str:
        s = (sym or '').strip().upper()
        for suf in ['.P', '.PERP', 'PERP']:
            if s.endswith(suf):
                s = s[:-len(suf)]
        return s

    def set_isolated_margin_mode(self):
        self.ensure_ready()
        try:
            self.client.set_margin_mode(setMarginMode='ISOLATED_MARGIN')
        except Exception:
            # Safe to ignore if not supported / already set.
            pass

    def set_leverage(self, symbol: str, lev: int):
        self.ensure_ready()
        try:
            self.client.set_leverage(category=CATEGORY, symbol=symbol, buyLeverage=str(lev), sellLeverage=str(lev))
        except Exception:
            pass

    def get_equity_usdt(self) -> float:
        self.ensure_ready()
        resp = self.client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
        lst = resp.get('result', {}).get('list', [])
        if not lst:
            return 0.0
        for c in lst[0].get('coin', []):
            if c.get('coin') == 'USDT':
                return float(c.get('equity', 0) or 0)
        return 0.0

    def get_last_price(self, symbol: str) -> float:
        self.ensure_ready()
        resp = self.client.get_tickers(category=CATEGORY, symbol=symbol)
        lst = resp.get('result', {}).get('list', []) or []
        if not lst:
            return 0.0
        return float(lst[0].get('lastPrice', 0) or 0)

    def get_position_size(self, symbol: str) -> float:
        self.ensure_ready()
        resp = self.client.get_positions(category=CATEGORY, symbol=symbol)
        pos_list = resp.get('result', {}).get('list', []) or []
        if not pos_list:
            return 0.0
        p = pos_list[0]
        size = float(p.get('size', 0) or 0)
        side = (p.get('side', '') or '').lower()
        return -size if side == 'sell' else size

    def get_qty_rules(self, symbol: str) -> QtyRules:
        self.ensure_ready()
        resp = self.client.get_instruments_info(category=CATEGORY, symbol=symbol)
        lst = resp.get('result', {}).get('list', []) or []
        if not lst:
            raise HTTPException(status_code=400, detail=f'symbol not found in instruments_info: {symbol}')
        lot = lst[0].get('lotSizeFilter', {}) or {}
        min_qty = float(lot.get('minOrderQty', 0) or 0)
        max_qty = float(lot.get('maxOrderQty', 0) or 0)
        step = float(lot.get('qtyStep', 0) or 0)
        if step <= 0:
            step = 1.0
        return QtyRules(min_qty=min_qty, max_qty=max_qty, step=step)

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]:
        self.ensure_ready()
        return self.client.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=side,
            orderType='Market',
            qty=fmt_number(qty),
            reduceOnly=reduce_only,
            timeInForce='GTC',
        )

    def close_full_position(self, symbol: str) -> Dict[str, Any]:
        pos = self.get_position_size(symbol)
        if pos == 0:
            return {'ok': True, 'note': 'no position to close'}
        qty = abs(pos)
        if pos > 0:
            return self.place_market(symbol, 'Sell', qty, reduce_only=True)
        return self.place_market(symbol, 'Buy', qty, reduce_only=True)

    def status(self, symbol: str) -> Dict[str, Any]:
        symbol = self.clean_symbol(symbol)
        if not self.is_ready():
            return {'ready': False, 'testnet': BYBIT_TESTNET, 'symbol': symbol}
        return {
            'ready': True,
            'testnet': BYBIT_TESTNET,
            'category': CATEGORY,
            'symbol': symbol,
            'equity_usdt': self.get_equity_usdt(),
            'last_price': self.get_last_price(symbol),
            'position_size': self.get_position_size(symbol),
        }

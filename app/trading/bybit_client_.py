import os
from typing import Any, Dict, Optional
from fastapi import HTTPException

try:
    from pybit.unified_trading import HTTP
except Exception:
    HTTP = None  # type: ignore

from app.config import settings
from .sizing import QtyRules, fmt_number


CATEGORY = os.getenv('BYBIT_CATEGORY', 'linear')
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', '1') != '0'
BYBIT_KEY = os.getenv('BYBIT_KEY', '')
BYBIT_SECRET = os.getenv('BYBIT_SECRET', '')


def _make_client() -> Optional['HTTP']:
    if HTTP is None:
        return None

    mode = getattr(settings, "BYBIT_MODE", "testnet").strip().lower()

    if mode == "demo":
        key = getattr(settings, "BYBIT_DEMO_KEY", "") or os.getenv("BYBIT_DEMO_KEY", "") or BYBIT_KEY
        secret = getattr(settings, "BYBIT_DEMO_SECRET", "") or os.getenv("BYBIT_DEMO_SECRET", "") or BYBIT_SECRET
        if not (key and secret):
            return None
        return HTTP(
            testnet=False,
            demo=True,
            api_key=key,
            api_secret=secret,
        )

    if mode == "mainnet":
        if not (BYBIT_KEY and BYBIT_SECRET):
            return None
        return HTTP(
            testnet=False,
            api_key=BYBIT_KEY,
            api_secret=BYBIT_SECRET,
        )

    # default: testnet
    if not (BYBIT_KEY and BYBIT_SECRET):
        return None
    return HTTP(
        testnet=True,
        api_key=BYBIT_KEY,
        api_secret=BYBIT_SECRET,
    )


class BybitClient:
    def __init__(self):
        self.client = _make_client()

    def is_ready(self) -> bool:
        return self.client is not None

    def ensure_ready(self):
        if self.client is None:
            raise HTTPException(status_code=500, detail='Bybit API not configured (set API keys and install pybit)')

    @property
    def ready(self) -> bool:
        return self.client is not None

    def clean_symbol(self, sym: str) -> str:
        s = (sym or '').strip().upper()
        for suf in ['.P', '.PERP', 'PERP']:
            if s.endswith(suf):
                s = s[:-len(suf)]
        if s.endswith('_R'):
            s = s[:-2]
        return s

    def set_isolated_margin_mode(self):
        self.ensure_ready()
        try:
            self.client.set_margin_mode(setMarginMode='ISOLATED_MARGIN')
        except Exception:
            pass

    def set_leverage(self, symbol: str, lev: int):
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
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
        symbol = self.clean_symbol(symbol)
        resp = self.client.get_tickers(category=CATEGORY, symbol=symbol)
        lst = resp.get('result', {}).get('list', []) or []
        if not lst:
            return 0.0
        return float(lst[0].get('lastPrice', 0) or 0)

    def get_position_size(self, symbol: str) -> float:
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
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
        symbol = self.clean_symbol(symbol)
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

    def _get_price_rules(self, symbol: str) -> tuple[float, float]:
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
        resp = self.client.get_instruments_info(category=CATEGORY, symbol=symbol)
        lst = resp.get('result', {}).get('list', []) or []
        if not lst:
            raise HTTPException(status_code=400, detail=f'symbol not found in instruments_info: {symbol}')
        pf = lst[0].get('priceFilter', {}) or {}
        tick = float(pf.get('tickSize', 0) or 0)
        min_price = float(pf.get('minPrice', 0) or 0)
        if tick <= 0:
            tick = 0.01
        return tick, min_price

    def _round_price_to_tick(self, price: float, tick: float) -> float:
        if tick <= 0:
            return price
        return round(round(price / tick) * tick, 10)

    def set_trading_stop(self, symbol: str, take_profit: float | None = None, stop_loss: float | None = None) -> Dict[str, Any]:
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
        if take_profit is None and stop_loss is None:
            return {'ok': True, 'note': 'no tp/sl requested'}
        tick, min_price = self._get_price_rules(symbol)
        payload: Dict[str, Any] = {'category': CATEGORY, 'symbol': symbol, 'tpslMode': 'Full'}
        if take_profit is not None and take_profit > 0:
            tp = max(self._round_price_to_tick(take_profit, tick), min_price or 0)
            payload['takeProfit'] = str(tp)
            payload['tpTriggerBy'] = 'MarkPrice'
        if stop_loss is not None and stop_loss > 0:
            sl = max(self._round_price_to_tick(stop_loss, tick), min_price or 0)
            payload['stopLoss'] = str(sl)
            payload['slTriggerBy'] = 'MarkPrice'
        return self.client.set_trading_stop(**payload)

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool = False, order_link_id: str | None = None) -> Dict[str, Any]:
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
        payload = {
            'category': CATEGORY,
            'symbol': symbol,
            'side': side,
            'orderType': 'Market',
            'qty': fmt_number(qty),
            'reduceOnly': reduce_only,
            'timeInForce': 'GTC',
        }
        if order_link_id:
            payload['orderLinkId'] = order_link_id
        return self.client.place_order(**payload)

    def close_full_position(self, symbol: str, order_link_id: str | None = None) -> Dict[str, Any]:
        pos = self.get_position_size(symbol)
        if pos == 0:
            return {'ok': True, 'note': 'no position to close'}
        qty = abs(pos)
        if pos > 0:
            return self.place_market(symbol, 'Sell', qty, reduce_only=True, order_link_id=order_link_id)
        return self.place_market(symbol, 'Buy', qty, reduce_only=True, order_link_id=order_link_id)

    def get_position_snapshot(self, symbol: str) -> Dict[str, Any]:
        symbol = self.clean_symbol(symbol)
        if not self.is_ready():
            return {'ready': False, 'symbol': symbol}
        self.ensure_ready()
        resp = self.client.get_positions(category=CATEGORY, symbol=symbol)
        pos_list = resp.get('result', {}).get('list', []) or []
        chosen = None
        for p in pos_list:
            try:
                if float(p.get('size', 0) or 0) > 0:
                    chosen = p
                    break
            except Exception:
                pass
        if chosen is None and pos_list:
            chosen = pos_list[0]
        if not chosen:
            return {'ready': True, 'symbol': symbol, 'has_position': False, 'position': None}

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        side_raw = str(chosen.get('side', '') or '')
        size = _to_float(chosen.get('size')) or 0.0
        return {
            'ready': True,
            'symbol': symbol,
            'has_position': size > 0,
            'position': {
                'side': side_raw,
                'size': size,
                'avgPrice': _to_float(chosen.get('avgPrice')),
                'markPrice': _to_float(chosen.get('markPrice')),
                'unrealisedPnl': _to_float(chosen.get('unrealisedPnl')),
                'leverage': str(chosen.get('leverage', '') or ''),
                'liqPrice': _to_float(chosen.get('liqPrice')),
                'takeProfit': _to_float(chosen.get('takeProfit')),
                'stopLoss': _to_float(chosen.get('stopLoss')),
                'positionIdx': chosen.get('positionIdx'),
                'tradeMode': chosen.get('tradeMode'),
            },
            'raw_count': len(pos_list),
        }

    def get_account_snapshot(self) -> Dict[str, Any]:
        if not self.is_ready():
            return {'ready': False}
        self.ensure_ready()
        try:
            resp = self.client.get_wallet_balance(accountType='UNIFIED', coin='USDT')
            result = resp.get('result', {}) or {}
            lst = result.get('list', []) or []
            acct = lst[0] if lst else {}
            coin_list = acct.get('coin', []) or []
            usdt = None
            for c in coin_list:
                if str(c.get('coin', '')).upper() == 'USDT':
                    usdt = c
                    break

            def _f(v):
                try:
                    return float(v)
                except Exception:
                    return None

            equity = _f(acct.get('totalEquity'))
            wallet = _f(acct.get('totalWalletBalance'))
            available = _f(acct.get('totalAvailableBalance'))
            unreal = _f(acct.get('totalPerpUPL'))
            if usdt:
                equity = equity if equity is not None else _f(usdt.get('equity'))
                wallet = wallet if wallet is not None else _f(usdt.get('walletBalance'))
                available = available if available is not None else _f(usdt.get('availableToWithdraw') or usdt.get('availableBalance'))
                unreal = unreal if unreal is not None else _f(usdt.get('unrealisedPnl'))
            return {
                'ready': True,
                'accountType': 'UNIFIED',
                'equity': equity,
                'walletBalance': wallet,
                'availableBalance': available,
                'unrealisedPnl': unreal,
                'raw': resp,
            }
        except Exception as e:
            return {'ready': False, 'error': repr(e)}


__all__ = ["BybitClient", "BYBIT_TESTNET", "CATEGORY"]

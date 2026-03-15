import os
from typing import Any, Dict, Optional, List
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


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


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

    # -------- NEW: execution / fill history --------

    def get_execution_history(
        self,
        symbol: str,
        limit: int = 50,
        start_ms: int | None = None,
        end_ms: int | None = None,
        order_id: str | None = None,
        order_link_id: str | None = None,
    ) -> Dict[str, Any]:
        self.ensure_ready()
        symbol = self.clean_symbol(symbol)
        payload: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "limit": int(limit),
        }
        if start_ms is not None:
            payload["startTime"] = int(start_ms)
        if end_ms is not None:
            payload["endTime"] = int(end_ms)
        if order_id:
            payload["orderId"] = order_id
        if order_link_id:
            payload["orderLinkId"] = order_link_id
        return self.client.get_executions(**payload)

    def _normalize_execution_fill(self, item: Dict[str, Any]) -> Dict[str, Any]:
        exec_price = _safe_float(item.get("execPrice") or item.get("price"))
        exec_qty = _safe_float(item.get("execQty") or item.get("exec_qty") or item.get("qty"))
        exec_fee = _safe_float(item.get("execFee") or item.get("exec_fee") or item.get("fee"))
        exec_time = item.get("execTime") or item.get("exec_time") or item.get("tradeTime")
        return {
            "symbol": item.get("symbol"),
            "side": item.get("side"),
            "orderId": item.get("orderId"),
            "orderLinkId": item.get("orderLinkId"),
            "execId": item.get("execId"),
            "execTime": int(exec_time) if str(exec_time).isdigit() else exec_time,
            "execPrice": exec_price,
            "execQty": exec_qty,
            "execFee": exec_fee,
            "feeCurrency": item.get("feeCurrency") or item.get("execFeeCurrency"),
            "isMaker": item.get("isMaker"),
            "raw": item,
        }

    def find_fills_by_order_link_id(
        self,
        symbol: str,
        order_link_id: str,
        limit: int = 100,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not order_link_id:
            return []
        resp = self.get_execution_history(
            symbol=symbol,
            limit=limit,
            start_ms=start_ms,
            end_ms=end_ms,
            order_link_id=order_link_id,
        )
        items = resp.get("result", {}).get("list", []) or []
        fills = [self._normalize_execution_fill(x) for x in items]
        # extra safeguard if api ignores orderLinkId filter
        fills = [f for f in fills if str(f.get("orderLinkId") or "") == str(order_link_id)]
        fills.sort(key=lambda x: x.get("execTime") or 0)
        return fills

    def summarize_fills(self, fills: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not fills:
            return {
                "qty": None,
                "avg_price": None,
                "fee": None,
                "fee_currency": None,
                "first_exec_time": None,
                "last_exec_time": None,
            }
        total_qty = 0.0
        total_notional = 0.0
        total_fee = 0.0
        fee_currency = None
        first_exec_time = None
        last_exec_time = None

        for f in fills:
            q = _safe_float(f.get("execQty")) or 0.0
            p = _safe_float(f.get("execPrice")) or 0.0
            fee = _safe_float(f.get("execFee")) or 0.0
            total_qty += q
            total_notional += q * p
            total_fee += fee
            fee_currency = fee_currency or f.get("feeCurrency")
            t = f.get("execTime")
            if first_exec_time is None or (t is not None and t < first_exec_time):
                first_exec_time = t
            if last_exec_time is None or (t is not None and t > last_exec_time):
                last_exec_time = t

        return {
            "qty": total_qty if total_qty > 0 else None,
            "avg_price": (total_notional / total_qty) if total_qty > 0 else None,
            "fee": total_fee if total_fee > 0 else 0.0,
            "fee_currency": fee_currency,
            "first_exec_time": first_exec_time,
            "last_exec_time": last_exec_time,
        }


__all__ = ["BybitClient", "BYBIT_TESTNET", "CATEGORY"]

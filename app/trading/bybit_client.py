import os
from typing import Any, Dict, Optional

try:
    from pybit.unified_trading import HTTP
except Exception:
    HTTP = None

from app.config import settings

# Backward-compatible exports expected by older routes.py
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "1") != "0"
CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")


def _make_client() -> Optional["HTTP"]:
    if HTTP is None:
        return None

    mode = getattr(settings, "BYBIT_MODE", "testnet").strip().lower()

    if mode == "demo":
        key = getattr(settings, "BYBIT_DEMO_KEY", "") or os.getenv("BYBIT_DEMO_KEY", "")
        secret = getattr(settings, "BYBIT_DEMO_SECRET", "") or os.getenv("BYBIT_DEMO_SECRET", "")
        if not (key and secret):
            return None
        return HTTP(
            testnet=False,
            api_key=key,
            api_secret=secret,
            domain="api-demo.bybit.com",
        )

    if mode == "mainnet":
        key = os.getenv("BYBIT_KEY", "")
        secret = os.getenv("BYBIT_SECRET", "")
        if not (key and secret):
            return None
        return HTTP(
            testnet=False,
            api_key=key,
            api_secret=secret,
        )

    # default: testnet
    key = os.getenv("BYBIT_KEY", "")
    secret = os.getenv("BYBIT_SECRET", "")
    if not (key and secret):
        return None
    return HTTP(
        testnet=True,
        api_key=key,
        api_secret=secret,
    )


class BybitClient:
    def __init__(self):
        self.client = _make_client()

    @property
    def ready(self) -> bool:
        return self.client is not None

    def _safe_call(self, fn_name: str, **kwargs) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "error": "Bybit client is not configured"}
        try:
            fn = getattr(self.client, fn_name)
            resp = fn(**kwargs)
            return {"ok": True, "response": resp}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_positions(self, symbol: str) -> Dict[str, Any]:
        return self._safe_call("get_positions", category=CATEGORY, symbol=symbol)

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        return self._safe_call("get_wallet_balance", accountType=account_type)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str = "Market",
        reduce_only: bool = False,
        position_idx: int = 0,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "reduceOnly": reduce_only,
            "positionIdx": position_idx,
        }
        payload.update(extra)
        return self._safe_call("place_order", **payload)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        return self._safe_call("cancel_all_orders", category=CATEGORY, symbol=symbol)

    def set_trading_stop(
        self,
        symbol: str,
        position_idx: int = 0,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "positionIdx": position_idx,
        }
        if take_profit is not None:
            payload["takeProfit"] = take_profit
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        payload.update(extra)
        return self._safe_call("set_trading_stop", **payload)


__all__ = ["BybitClient", "BYBIT_TESTNET", "CATEGORY"]

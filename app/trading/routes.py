
from typing import Any, Dict
import time

from fastapi import APIRouter, Body, Depends, Query

from .auth import require_trade_token
from .bybit_client import BybitClient, BYBIT_TESTNET, CATEGORY
from .executor import execute_action

router = APIRouter(prefix='/trade', tags=['trade'])

_last_cmd: Dict[str, Any] | None = None
_last_result: Dict[str, Any] | None = None


def _client() -> BybitClient:
    return BybitClient()


@router.get('/health')
def trade_health():
    c = _client()
    return {
        'ok': True,
        'ready': c.is_ready(),
        'testnet': BYBIT_TESTNET,
        'category': CATEGORY,
        'has_keys': bool(c.is_ready()),
    }


@router.get('/last')
def trade_last():
    return {'ok': True, 'last_cmd': _last_cmd, 'last_result': _last_result}


@router.get('/status')
def trade_status(symbol: str = Query('APTUSDT')):
    c = _client()
    return {'ok': True, 'status': c.status(symbol)}


@router.post('/execute')
def trade_execute(
    payload: Dict[str, Any] = Body(...),
    _auth=Depends(require_trade_token),
):
    global _last_cmd, _last_result
    symbol = str(payload.get('symbol', 'APTUSDT'))
    action = str(payload.get('action', '')).upper()
    position_usd = payload.get('position_usd')
    leverage = payload.get('leverage')
    sl_percent = payload.get('sl_percent')
    tp_percent = payload.get('tp_percent')
    _last_cmd = {
        'ts': time.time(),
        'symbol': symbol,
        'action': action,
        'position_usd': position_usd,
        'leverage': leverage,
        'sl_percent': sl_percent,
        'tp_percent': tp_percent,
    }
    c = _client()
    _last_result = execute_action(c, symbol=symbol, action=action, position_usd=position_usd, leverage=leverage, sl_percent=sl_percent, tp_percent=tp_percent)
    return {'ok': True, 'result': _last_result}

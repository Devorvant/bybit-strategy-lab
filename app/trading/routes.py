
from typing import Any, Dict
import time

from fastapi import APIRouter, Body, Depends, Query

from .auth import require_trade_token
from .bybit_client import BybitClient, BYBIT_TESTNET, CATEGORY
from .executor import execute_action
from .auto_trader import get_auto_config, get_last_auto_events, process_symbol

router = APIRouter(prefix='/trade', tags=['trade'])

_last_cmd: Dict[str, Any] | None = None
_last_result: Dict[str, Any] | None = None


def _client() -> BybitClient:
    return BybitClient()




@router.get('/auto/config')
def trade_auto_config():
    return {'ok': True, 'config': get_auto_config()}


@router.get('/auto/last')
def trade_auto_last(limit: int = Query(20, ge=1, le=100)):
    return get_last_auto_events(limit=limit)


@router.get('/auto/preview')
def trade_auto_preview(
    symbol: str = Query('APTUSDT'),
    tf: str = Query('120'),
):
    return {'ok': True, 'result': process_symbol(symbol=symbol, tf=tf, execute=False)}


@router.post('/auto/process')
def trade_auto_process(
    payload: Dict[str, Any] = Body(...),
    _auth=Depends(require_trade_token),
):
    symbol = str(payload.get('symbol', 'APTUSDT'))
    tf = str(payload.get('tf', '120'))
    execute = bool(payload.get('execute', False))
    force = bool(payload.get('force', False))
    strategy_params = payload.get('strategy_params') if isinstance(payload.get('strategy_params'), dict) else {}
    trade_params = payload.get('trade_params') if isinstance(payload.get('trade_params'), dict) else {}
    return {
        'ok': True,
        'result': process_symbol(
            symbol=symbol,
            tf=tf,
            execute=execute,
            force=force,
            strategy_params=strategy_params,
            trade_params=trade_params,
        )
    }
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




@router.get('/position')
def trade_position(symbol: str = Query('APTUSDT')):
    c = _client()
    return {'ok': True, 'position': c.get_position_snapshot(symbol)}


@router.get('/account')
def trade_account():
    c = _client()
    return {'ok': True, 'account': c.get_account_snapshot()}

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

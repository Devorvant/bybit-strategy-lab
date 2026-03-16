
from typing import Any, Dict
import time

from fastapi import APIRouter, Body, Depends, Query

from .auth import require_trade_token
from .bybit_client import BybitClient, BYBIT_TESTNET, CATEGORY
from .executor import execute_action
from .auto_trader import get_auto_config, get_auto_loop_status, get_last_auto_events, process_symbol, start_auto_loop, stop_auto_loop
from .strategy_registry import ensure_opt_auto_columns, get_opt_strategy_entry, set_opt_auto_status, list_risk_profiles, get_risk_profile, set_opt_risk_profile
from app.storage.db import get_conn

from app.trading.analyzer import analyze_recent_events, analyze_exchange_side_closes

try:
    from .analyzer import analyze_recent_events
except Exception:
    def analyze_recent_events(*args, **kwargs):
        return []

try:
    from .journal import (
        log_execution_event,
        log_exchange_snapshot,
        snapshot_from_exchange,
        log_execution_fills,
        upsert_execution_fill_summary,
    )
except Exception:
    def log_execution_event(*args, **kwargs):
        pass
    def log_exchange_snapshot(*args, **kwargs):
        pass
    def snapshot_from_exchange(*args, **kwargs):
        return {}
    def log_execution_fills(*args, **kwargs):
        pass
    def upsert_execution_fill_summary(*args, **kwargs):
        pass

router = APIRouter(prefix='/trade', tags=['trade'])

_last_cmd: Dict[str, Any] | None = None
_last_result: Dict[str, Any] | None = None


def _client() -> BybitClient:
    return BybitClient()


def _extract_order_ids(resp: Dict[str, Any] | None) -> tuple[str | None, str | None]:
    try:
        result = ((resp or {}).get('response') or {}).get('result') or {}
        order_id = result.get('orderId')
        order_link_id = result.get('orderLinkId')
        return order_id, order_link_id
    except Exception:
        return None, None


def _make_manual_order_link_id(action: str) -> str:
    a = str(action or '').strip().lower() or 'unknown'
    return f"manual-{a}-{int(time.time() * 1000)}"


def _persist_fills_for_order(conn, client: BybitClient, *, symbol: str, tf: str | None, order_link_id: str | None) -> int:
    if not order_link_id:
        return 0
    try:
        fills = client.find_fills_by_order_link_id(symbol=symbol, order_link_id=order_link_id, limit=100)
        if not fills:
            return 0
        log_execution_fills(conn, symbol=symbol, tf=tf, fills=fills)
        upsert_execution_fill_summary(conn, symbol=symbol, tf=tf, order_link_id=order_link_id)
        return len(fills)
    except Exception:
        return 0

@router.get('/strategy3/auto-status')
def trade_strategy3_auto_status(opt_id: int = Query(..., ge=1)):
    ensure_opt_auto_columns()
    row = get_opt_strategy_entry(opt_id)
    if not row:
        return {'ok': False, 'error': 'strategy preset not found'}
    return {'ok': True, 'strategy': row}


@router.post('/strategy3/auto-status')
def trade_strategy3_auto_status_set(
    payload: Dict[str, Any] = Body(...),
):
    opt_id = int(payload.get('opt_id') or 0)
    enabled = bool(payload.get('auto_trade_enabled', False))
    allowed_symbols = payload.get('allowed_symbols') if isinstance(payload.get('allowed_symbols'), list) else None
    allowed_timeframes = payload.get('allowed_timeframes') if isinstance(payload.get('allowed_timeframes'), list) else None
    if opt_id <= 0:
        return {'ok': False, 'error': 'opt_id is required'}
    ensure_opt_auto_columns()
    row = set_opt_auto_status(opt_id, enabled, allowed_symbols=allowed_symbols, allowed_timeframes=allowed_timeframes)
    if not row:
        return {'ok': False, 'error': 'strategy preset not found'}
    return {'ok': True, 'strategy': row}



@router.get('/risk-profiles')
def trade_risk_profiles():
    return {'ok': True, 'profiles': list_risk_profiles()}


@router.get('/strategy3/risk-profile')
def trade_strategy3_risk_profile(opt_id: int = Query(..., ge=1)):
    ensure_opt_auto_columns()
    row = get_opt_strategy_entry(opt_id)
    if not row:
        return {'ok': False, 'error': 'strategy preset not found'}
    return {
        'ok': True,
        'opt_id': opt_id,
        'risk_profile_id': row.get('risk_profile_id'),
        'risk_profile': row.get('risk_profile'),
    }


@router.post('/strategy3/risk-profile')
def trade_strategy3_risk_profile_set(
    payload: Dict[str, Any] = Body(...),
):
    opt_id = int(payload.get('opt_id') or 0)
    risk_profile_id = payload.get('risk_profile_id')
    if opt_id <= 0:
        return {'ok': False, 'error': 'opt_id is required'}
    try:
        row = set_opt_risk_profile(opt_id, risk_profile_id)
    except ValueError as e:
        return {'ok': False, 'error': str(e)}
    if not row:
        return {'ok': False, 'error': 'strategy preset not found'}
    return {
        'ok': True,
        'opt_id': opt_id,
        'risk_profile_id': row.get('risk_profile_id'),
        'risk_profile': row.get('risk_profile'),
        'strategy': row,
    }

@router.get('/auto/config')
def trade_auto_config():
    return {'ok': True, 'config': get_auto_config()}


@router.get('/auto/last')
def trade_auto_last(limit: int = Query(20, ge=1, le=100)):
    return get_last_auto_events(limit=limit)


@router.get('/auto/loop/status')
def trade_auto_loop_status():
    return {'ok': True, 'loop': get_auto_loop_status()}


@router.post('/auto/loop/start')
def trade_auto_loop_start(
    payload: Dict[str, Any] = Body(...),
    _auth=Depends(require_trade_token),
):
    symbol = str(payload.get('symbol', 'APTUSDT'))
    tf = str(payload.get('tf', '120'))
    opt_id = int(payload.get('opt_id')) if str(payload.get('opt_id') or '').strip().isdigit() else 0
    poll_sec = float(payload.get('poll_sec') or 0) or None
    trade_params = payload.get('trade_params') if isinstance(payload.get('trade_params'), dict) else {}
    return start_auto_loop(symbol=symbol, tf=tf, opt_id=opt_id, poll_sec=poll_sec, trade_params=trade_params)


@router.post('/auto/loop/stop')
def trade_auto_loop_stop(
    _auth=Depends(require_trade_token),
):
    return stop_auto_loop()



@router.get('/auto/preview')
def trade_auto_preview(
    symbol: str = Query('APTUSDT'),
    tf: str = Query('120'),
    opt_id: int | None = Query(None),
):
    return {'ok': True, 'result': process_symbol(symbol=symbol, tf=tf, opt_id=opt_id, execute=False)}


@router.post('/auto/process')
def trade_auto_process(
    payload: Dict[str, Any] = Body(...),
    _auth=Depends(require_trade_token),
):
    symbol = str(payload.get('symbol', 'APTUSDT'))
    tf = str(payload.get('tf', '120'))
    execute = bool(payload.get('execute', False))
    force = bool(payload.get('force', False))
    opt_id = int(payload.get('opt_id')) if str(payload.get('opt_id') or '').strip().isdigit() else None
    strategy_params = payload.get('strategy_params') if isinstance(payload.get('strategy_params'), dict) else {}
    trade_params = payload.get('trade_params') if isinstance(payload.get('trade_params'), dict) else {}
    return {
        'ok': True,
        'result': process_symbol(
            symbol=symbol,
            tf=tf,
            execute=execute,
            force=force,
            opt_id=opt_id,
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
    pos = c.get_position_snapshot(symbol)

    try:
        conn = get_conn()
        acc = c.get_account_snapshot()
        log_exchange_snapshot(conn, snapshot_from_exchange(
            symbol,
            pos,
            acc,
            source='refresh',
            run_id=None,
            tf=None,
            bar_ts=None,
        ))
        conn.commit()
    except Exception:
        pass

    return {'ok': True, 'position': pos}


@router.get('/account')
def trade_account(symbol: str = Query('APTUSDT')):
    c = _client()
    acc = c.get_account_snapshot()

    try:
        conn = get_conn()
        pos = c.get_position_snapshot(symbol)
        log_exchange_snapshot(conn, snapshot_from_exchange(
            symbol,
            pos,
            acc,
            source='refresh',
            run_id=None,
            tf=None,
            bar_ts=None,
        ))
        conn.commit()
    except Exception:
        pass

    return {'ok': True, 'account': acc}

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

    order_link_id = _make_manual_order_link_id(action)

    _last_cmd = {
        'ts': time.time(),
        'symbol': symbol,
        'action': action,
        'position_usd': position_usd,
        'leverage': leverage,
        'sl_percent': sl_percent,
        'tp_percent': tp_percent,
        'order_link_id': order_link_id,
    }

    conn = None
    try:
        conn = get_conn()
        log_execution_event(conn, {
            'run_id': None,
            'symbol': symbol,
            'tf': None,
            'bar_ts': None,
            'event_type': 'manual_execute_requested',
            'requested_action': action,
            'side': None,
            'qty': None,
            'price': None,
            'reduce_only': (action == 'CLOSE'),
            'ok': True,
            'error': None,
            'order_id': None,
            'order_link_id': order_link_id,
            'response': {
                'source': 'manual_trade',
                'position_usd': position_usd,
                'leverage': leverage,
                'sl_percent': sl_percent,
                'tp_percent': tp_percent,
            },
        })
        conn.commit()
    except Exception:
        pass

    c = _client()
    _last_result = execute_action(
        c,
        symbol=symbol,
        action=action,
        position_usd=position_usd,
        leverage=leverage,
        sl_percent=sl_percent,
        tp_percent=tp_percent,
        order_link_id=order_link_id,
    )

    try:
        conn = get_conn()
        order_id, result_order_link_id = _extract_order_ids(_last_result)
        log_execution_event(conn, {
            'run_id': None,
            'symbol': symbol,
            'tf': None,
            'bar_ts': None,
            'event_type': 'manual_execute_result',
            'requested_action': action,
            'side': None,
            'qty': None,
            'price': None,
            'reduce_only': (action == 'CLOSE'),
            'ok': bool((_last_result or {}).get('ok', False)),
            'error': (_last_result or {}).get('error'),
            'order_id': order_id,
            'order_link_id': result_order_link_id or order_link_id,
            'response': _last_result,
        })
        try:
            pos = c.get_position_snapshot(symbol)
            acc = c.get_account_snapshot()
            log_exchange_snapshot(conn, snapshot_from_exchange(
                symbol,
                pos,
                acc,
                source='manual_after',
                run_id=None,
                tf=None,
                bar_ts=None,
            ))
        except Exception:
            pass
        try:
            _persist_fills_for_order(
                conn,
                c,
                symbol=symbol,
                tf=None,
                order_link_id=(result_order_link_id or order_link_id),
            )
        except Exception:
            pass
        conn.commit()
    except Exception:
        pass

    return {'ok': True, 'result': _last_result}


@router.get('/analyze/recent')
def trade_analyze_recent(limit: int = 50, _auth=Depends(require_trade_token)):
    conn = get_conn()
    rows = analyze_recent_events(conn, limit=limit)
    return {
        'ok': True,
        'count': len(rows),
        'rows': rows,
    }


@router.get('/analyze/mismatch')
def trade_analyze_mismatch(limit: int = 100, _auth=Depends(require_trade_token)):
    conn = get_conn()
    rows = analyze_recent_events(conn, limit=limit)
    bad = [
        r for r in rows
        if r.get('status') in ('mismatch', 'no_effect', 'error', 'unknown')
    ]
    return {
        'ok': True,
        'count': len(bad),
        'rows': bad,
    }


@router.get('/analyze/exchange-closes')
def trade_analyze_exchange_closes(
    symbol: str | None = None,
    limit: int = 200,
    _auth=Depends(require_trade_token),
):
    conn = get_conn()
    rows = analyze_exchange_side_closes(conn, symbol=symbol, limit=limit)
    return {
        "ok": True,
        "count": len(rows),
        "rows": rows,
    }

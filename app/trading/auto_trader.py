
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.storage.db import init_db, load_bars
from app.strategy.my_strategy3 import generate_signal, generate_signal_via_backtest
from .bybit_client import BYBIT_TESTNET, BybitClient
from .executor import DEFAULT_LEVERAGE, DEFAULT_POSITION_USD, execute_action
from .strategy_registry import get_opt_strategy_entry


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str, fallback: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw and raw.strip():
        return [s.strip().upper() for s in raw.split(',') if s.strip()]
    return [str(x).strip().upper() for x in fallback if str(x).strip()]


AUTO_TRADE_ENABLED = _bool_env('AUTO_TRADE_ENABLED', False)
AUTO_TRADE_TESTNET_ONLY = _bool_env('AUTO_TRADE_TESTNET_ONLY', True)
AUTO_TRADE_CLOSE_ON_FLAT = _bool_env('AUTO_TRADE_CLOSE_ON_FLAT', False)
AUTO_TRADE_COOLDOWN_SEC = int(os.getenv('AUTO_TRADE_COOLDOWN_SEC', '10'))
AUTO_TRADE_LOOKBACK = int(os.getenv('AUTO_TRADE_LOOKBACK', str(max(300, int(getattr(settings, 'LOOKBACK', 500) or 500)))))
AUTO_TRADE_ALLOWED_SYMBOLS = _csv_env('AUTO_TRADE_ALLOWED_SYMBOLS', list(getattr(settings, 'SYMBOLS', []) or []))
AUTO_TRADE_ALLOWED_TFS = _csv_env('AUTO_TRADE_ALLOWED_TFS', list(getattr(settings, 'TFS', []) or [getattr(settings, 'TF', '120')]))
AUTO_TRADE_LOG_PATH = os.getenv('AUTO_TRADE_LOG_PATH', 'auto_trade_log.jsonl')
AUTO_TRADE_MAX_EVENTS = int(os.getenv('AUTO_TRADE_MAX_EVENTS', '100'))
AUTO_TRADE_POLL_SEC = float(os.getenv('AUTO_TRADE_POLL_SEC', '5'))
AUTO_DEFAULT_LEVERAGE = max(1, int(os.getenv('AUTO_DEFAULT_LEVERAGE', '1')))
AUTO_MAX_LEVERAGE = max(1, int(os.getenv('AUTO_MAX_LEVERAGE', '2')))
AUTO_POSITION_MODE = (os.getenv('AUTO_POSITION_MODE', 'fixed_usd') or 'fixed_usd').strip().lower()
AUTO_POSITION_USD = max(0.0, float(os.getenv('AUTO_POSITION_USD', '25')))
AUTO_MAX_MARGIN_USDT = max(0.0, float(os.getenv('AUTO_MAX_MARGIN_USDT', str(AUTO_POSITION_USD or 25))))
AUTO_REQUIRE_SL = _bool_env('AUTO_REQUIRE_SL', True)
AUTO_REQUIRE_TP = _bool_env('AUTO_REQUIRE_TP', False)
AUTO_SL_PERCENT = max(0.0, float(os.getenv('AUTO_SL_PERCENT', '1.0')))
AUTO_TP_PERCENT = max(0.0, float(os.getenv('AUTO_TP_PERCENT', '0')))
AUTO_LOOP_WAIT_NEXT_BAR_ON_START = _bool_env('AUTO_LOOP_WAIT_NEXT_BAR_ON_START', True)
AUTO_BLOCK_ON_SL_FAILURE = _bool_env('AUTO_BLOCK_ON_SL_FAILURE', True)
AUTO_CLOSE_ON_SL_FAILURE = _bool_env('AUTO_CLOSE_ON_SL_FAILURE', True)

_CONN = None
_RUNTIME: Dict[str, Dict[str, Any]] = {}
_EVENTS: deque[dict] = deque(maxlen=max(10, AUTO_TRADE_MAX_EVENTS))
_LOOP_LOCK = threading.Lock()
_LOOP_THREAD: threading.Thread | None = None
_LOOP_STOP = threading.Event()
_LOOP_STATE: Dict[str, Any] = {
    'running': False,
    'symbol': None,
    'tf': None,
    'opt_id': None,
    'poll_sec': AUTO_TRADE_POLL_SEC,
    'started_at': None,
    'last_tick_at': None,
    'last_result': None,
    'last_error': None,
    'trade_params': None,
    'state': 'STOPPED',
    'armed_bar_ts': None,
    'blocked_reason': None,
}


@dataclass
class DecisionView:
    ok: bool
    symbol: str
    tf: str
    strategy: str
    bar_ts: int | None = None
    signal: str | None = None
    signal_note: str | None = None
    exchange_side: str | None = None
    exchange_size: float | None = None
    action: str | None = None
    reason: str | None = None
    execute_requested: bool = False
    executed: bool = False
    blocked: bool = False
    cooldown_active: bool = False
    cooldown_until: float | None = None
    bybit_ready: bool | None = None
    auto_enabled: bool | None = None
    testnet: bool | None = None
    state: Dict[str, Any] | None = None
    position: Dict[str, Any] | None = None
    execution: Dict[str, Any] | None = None
    strategy_params: Dict[str, Any] | None = None
    trade_params: Dict[str, Any] | None = None
    error: str | None = None
    opt_id: int | None = None
    strategy_name: str | None = None
    strategy_auto_trade_enabled: bool | None = None
    fatal_block: bool = False
    safety_state: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _get_conn():
    global _CONN
    if _CONN is None:
        _CONN = init_db()
    return _CONN



def _log_event(event: Dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault('logged_at', time.time())
    _EVENTS.appendleft(event)
    try:
        path = Path(AUTO_TRADE_LOG_PATH)
        if path.parent and str(path.parent) not in ('', '.'):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception:
        pass



def _runtime_key(symbol: str, tf: str) -> str:
    return f'{symbol.upper()}:{str(tf)}'



def _get_runtime_state(symbol: str, tf: str) -> Dict[str, Any]:
    key = _runtime_key(symbol, tf)
    st = _RUNTIME.get(key)
    if st is None:
        st = {
            'last_bar_ts': None,
            'last_signal': None,
            'last_action': None,
            'last_reason': None,
            'last_run_at': None,
            'cooldown_until': 0.0,
            'last_execution_ok': None,
            'last_execute_bar_ts': None,
            'last_execute_action': None,
        }
        _RUNTIME[key] = st
    return st



def _snapshot_runtime(symbol: str, tf: str) -> Dict[str, Any]:
    return dict(_get_runtime_state(symbol, tf))



def _normalize_position_side(position_snapshot: Optional[Dict[str, Any]]) -> tuple[str, float]:
    if not position_snapshot or not position_snapshot.get('has_position'):
        return 'FLAT', 0.0
    pos = position_snapshot.get('position') or {}
    side = str(pos.get('side', '') or '').upper()
    size = float(pos.get('size', 0) or 0)
    if side == 'BUY':
        return 'LONG', size
    if side == 'SELL':
        return 'SHORT', size
    return ('FLAT', 0.0) if size <= 0 else (side or 'UNKNOWN', size)



def _filter_strategy_params(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = dict(raw or {})
    sig = inspect.signature(generate_signal_via_backtest)
    allowed = set(sig.parameters.keys()) - {'bars'}
    out: Dict[str, Any] = {}
    for k, v in src.items():
        if k in allowed and v is not None:
            out[k] = v
    return out


def _resolve_auto_trade_params(_raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    leverage = min(AUTO_DEFAULT_LEVERAGE, AUTO_MAX_LEVERAGE)
    position_usd = min(AUTO_POSITION_USD, AUTO_MAX_MARGIN_USDT)
    sl_percent = AUTO_SL_PERCENT if AUTO_REQUIRE_SL else (AUTO_SL_PERCENT or None)
    tp_percent = AUTO_TP_PERCENT if (AUTO_REQUIRE_TP or AUTO_TP_PERCENT > 0) else None
    return {
        'mode': AUTO_POSITION_MODE,
        'position_usd': position_usd,
        'max_margin_usdt': AUTO_MAX_MARGIN_USDT,
        'leverage': leverage,
        'max_leverage': AUTO_MAX_LEVERAGE,
        'sl_percent': sl_percent,
        'tp_percent': tp_percent,
        'require_sl': AUTO_REQUIRE_SL,
        'require_tp': AUTO_REQUIRE_TP,
    }


def _verify_position_protection(bybit: BybitClient, symbol: str, *, require_sl: bool = True) -> tuple[bool, Dict[str, Any], str | None]:
    snapshot = bybit.get_position_snapshot(symbol)
    if not snapshot.get('ready'):
        return False, snapshot, 'position snapshot unavailable'
    if not snapshot.get('has_position'):
        return False, snapshot, 'position missing after entry'
    pos = snapshot.get('position') or {}
    stop_loss = pos.get('stopLoss')
    try:
        stop_loss_val = float(stop_loss or 0)
    except Exception:
        stop_loss_val = 0.0
    if require_sl and stop_loss_val <= 0:
        return False, snapshot, 'stop loss was not confirmed after entry'
    return True, snapshot, None



def get_auto_config() -> Dict[str, Any]:
    return {
        'strategy': 'my_strategy3',
        'auto_enabled': AUTO_TRADE_ENABLED,
        'testnet_only': AUTO_TRADE_TESTNET_ONLY,
        'close_on_flat': AUTO_TRADE_CLOSE_ON_FLAT,
        'cooldown_sec': AUTO_TRADE_COOLDOWN_SEC,
        'lookback': AUTO_TRADE_LOOKBACK,
        'allowed_symbols': AUTO_TRADE_ALLOWED_SYMBOLS,
        'allowed_tfs': AUTO_TRADE_ALLOWED_TFS,
        'default_position_usd': AUTO_POSITION_USD,
        'default_leverage': AUTO_DEFAULT_LEVERAGE,
        'max_leverage': AUTO_MAX_LEVERAGE,
        'max_margin_usdt': AUTO_MAX_MARGIN_USDT,
        'require_sl': AUTO_REQUIRE_SL,
        'require_tp': AUTO_REQUIRE_TP,
        'sl_percent': AUTO_SL_PERCENT,
        'tp_percent': AUTO_TP_PERCENT,
        'wait_next_bar_on_start': AUTO_LOOP_WAIT_NEXT_BAR_ON_START,
        'bybit_testnet': BYBIT_TESTNET,
        'execute_requires_saved_preset': True,
        'poll_sec': AUTO_TRADE_POLL_SEC,
        'loop': get_auto_loop_status(),
    }



def _compute_action(signal: str, exchange_side: str, close_on_flat: bool) -> tuple[str, str]:
    sig = (signal or 'FLAT').upper()
    side = (exchange_side or 'FLAT').upper()
    if sig == 'LONG':
        if side == 'LONG':
            return 'HOLD', 'already long'
        if side == 'SHORT':
            return 'CLOSE_SHORT', 'opposite signal, close short first'
        return 'OPEN_LONG', 'no position + long signal'
    if sig == 'SHORT':
        if side == 'SHORT':
            return 'HOLD', 'already short'
        if side == 'LONG':
            return 'CLOSE_LONG', 'opposite signal, close long first'
        return 'OPEN_SHORT', 'no position + short signal'
    # FLAT
    if close_on_flat and side == 'LONG':
        return 'CLOSE_LONG', 'flat signal requests closing long'
    if close_on_flat and side == 'SHORT':
        return 'CLOSE_SHORT', 'flat signal requests closing short'
    return 'HOLD', 'flat / no action'



def process_symbol(
    symbol: str,
    tf: str,
    *,
    execute: bool = False,
    force: bool = False,
    opt_id: int | None = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    trade_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = (symbol or '').strip().upper()
    tf = str(tf or getattr(settings, 'TF', '120')).strip()
    requested_trade_params = dict(trade_params or {})
    trade_params = _resolve_auto_trade_params(requested_trade_params)
    state = _get_runtime_state(symbol, tf)

    opt_entry = None
    strategy_name = None
    strategy_auto_enabled = None
    strategy_kwargs = _filter_strategy_params(strategy_params)
    if opt_id is not None:
        try:
            opt_entry = get_opt_strategy_entry(int(opt_id))
        except Exception as e:
            return DecisionView(ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', opt_id=opt_id, error=f'failed to load opt strategy: {e!r}').to_dict()
        if not opt_entry:
            return DecisionView(ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', opt_id=opt_id, error='strategy preset not found').to_dict()
        if str(opt_entry.get('strategy') or '') != 'strategy3':
            return DecisionView(ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', opt_id=opt_id, error='preset is not a strategy3 record').to_dict()
        strategy_name = f"#{int(opt_entry['id'])}"
        strategy_auto_enabled = bool(opt_entry.get('auto_trade_enabled'))
        strategy_kwargs = _filter_strategy_params(opt_entry.get('best_params') or {})

    try:
        conn = _get_conn()
        bars = load_bars(conn, symbol, tf, limit=AUTO_TRADE_LOOKBACK)
    except Exception as e:
        return DecisionView(ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', error=f'failed to load bars: {e!r}').to_dict()

    if len(bars) < 50:
        return DecisionView(ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', error=f'not enough bars for strategy3: {len(bars)}').to_dict()

    bar_ts = int(bars[-1][0])
    bybit = BybitClient()
    position_snapshot = bybit.get_position_snapshot(symbol)
    exchange_side, exchange_size = _normalize_position_side(position_snapshot)

    try:
        sig_out = generate_signal(bars, **strategy_kwargs)
    except Exception as e:
        return DecisionView(
            ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', bar_ts=bar_ts,
            error=f'strategy3 failed: {e!r}', state=_snapshot_runtime(symbol, tf), position=position_snapshot,
            strategy_params=strategy_kwargs, trade_params=trade_params,
        ).to_dict()

    if not sig_out:
        return DecisionView(
            ok=False, symbol=symbol, tf=tf, strategy='my_strategy3', bar_ts=bar_ts,
            error='strategy3 returned no signal', state=_snapshot_runtime(symbol, tf), position=position_snapshot,
            strategy_params=strategy_kwargs, trade_params=trade_params,
        ).to_dict()

    signal = str(getattr(sig_out, 'signal', 'FLAT') or 'FLAT').upper()
    signal_note = str(getattr(sig_out, 'note', '') or '')
    action, reason = _compute_action(signal, exchange_side, AUTO_TRADE_CLOSE_ON_FLAT)

    now = time.time()
    cooldown_until = float(state.get('cooldown_until') or 0.0)
    cooldown_active = cooldown_until > now
    blocked_reason = None
    fatal_block = False
    entry_allowed_symbols = [str(x).strip().upper() for x in ((opt_entry or {}).get('allowed_symbols') or []) if str(x).strip()]
    entry_allowed_tfs = [str(x).strip().upper() for x in ((opt_entry or {}).get('allowed_timeframes') or []) if str(x).strip()]

    if symbol not in AUTO_TRADE_ALLOWED_SYMBOLS:
        blocked_reason = f'symbol not allowed: {symbol}'
    elif tf.upper() not in AUTO_TRADE_ALLOWED_TFS:
        blocked_reason = f'tf not allowed: {tf}'
    elif entry_allowed_symbols and symbol not in entry_allowed_symbols:
        blocked_reason = f'symbol not allowed for preset #{int(opt_id)}: {symbol}'
    elif entry_allowed_tfs and tf.upper() not in entry_allowed_tfs:
        blocked_reason = f'tf not allowed for preset #{int(opt_id)}: {tf}'
    elif AUTO_TRADE_TESTNET_ONLY and not BYBIT_TESTNET:
        blocked_reason = 'testnet_only enabled, but Bybit client is not on testnet'
    elif execute and opt_id is None:
        blocked_reason = 'select saved strategy number (opt_id) first'
    elif execute and not strategy_auto_enabled:
        blocked_reason = 'selected strategy is not allowed for auto trading'
    elif execute and not AUTO_TRADE_ENABLED:
        blocked_reason = 'AUTO_TRADE_ENABLED is false'
    elif execute and not bybit.is_ready():
        blocked_reason = 'Bybit is not configured'
        fatal_block = True
    elif execute and float(trade_params.get('leverage') or 0) > float(trade_params.get('max_leverage') or 0):
        blocked_reason = 'leverage_limit_exceeded'
        fatal_block = True
    elif execute and float(trade_params.get('position_usd') or 0) > float(trade_params.get('max_margin_usdt') or 0):
        blocked_reason = 'position_usd_limit_exceeded'
        fatal_block = True
    elif execute and action in {'OPEN_LONG', 'OPEN_SHORT'} and bool(trade_params.get('require_sl')) and float(trade_params.get('sl_percent') or 0) <= 0:
        blocked_reason = 'sl_required_but_missing'
        fatal_block = True
    elif execute and cooldown_active:
        blocked_reason = f'cooldown active until {cooldown_until:.0f}'
    elif execute and not force and state.get('last_execute_bar_ts') == bar_ts and state.get('last_execute_action') == action and action != 'HOLD':
        blocked_reason = 'same bar/action already processed'

    execution_result = None
    executed = False
    blocked = blocked_reason is not None

    if execute and not blocked and action in {'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'}:
        action_for_executor = 'LONG' if action == 'OPEN_LONG' else 'SHORT' if action == 'OPEN_SHORT' else 'CLOSE'
        execution_result = execute_action(
            bybit,
            symbol=symbol,
            action=action_for_executor,
            position_usd=trade_params.get('position_usd'),
            leverage=trade_params.get('leverage'),
            sl_percent=trade_params.get('sl_percent'),
            tp_percent=trade_params.get('tp_percent'),
        )
        executed = bool(execution_result.get('ok'))
        state['cooldown_until'] = now + max(0, AUTO_TRADE_COOLDOWN_SEC)
        state['last_execution_ok'] = executed
        if executed:
            state['last_execute_bar_ts'] = bar_ts
            state['last_execute_action'] = action
            if action in {'OPEN_LONG', 'OPEN_SHORT'} and bool(trade_params.get('require_sl')):
                time.sleep(0.6)
                sl_ok, protection_snapshot, protection_error = _verify_position_protection(bybit, symbol, require_sl=True)
                execution_result['protection_snapshot'] = protection_snapshot
                execution_result['sl_verified'] = sl_ok
                if not sl_ok:
                    if AUTO_CLOSE_ON_SL_FAILURE:
                        try:
                            execution_result['emergency_close'] = bybit.close_full_position(symbol)
                        except Exception as e:
                            execution_result['emergency_close_error'] = repr(e)
                    blocked = True
                    fatal_block = AUTO_BLOCK_ON_SL_FAILURE
                    executed = False
                    blocked_reason = protection_error or 'stop loss was not confirmed after entry'
        if not executed and not blocked_reason:
            blocked = True
            blocked_reason = execution_result.get('error') or execution_result.get('note') or 'execution failed'
            fatal_block = True
    elif execute and not blocked and action == 'HOLD':
        reason = f'{reason}; nothing to execute'

    state['last_bar_ts'] = bar_ts
    state['last_signal'] = signal
    state['last_action'] = action
    state['last_reason'] = blocked_reason or reason
    state['last_run_at'] = now

    payload = DecisionView(
        ok=True,
        symbol=symbol,
        tf=tf,
        strategy='my_strategy3',
        opt_id=opt_id,
        strategy_name=strategy_name,
        strategy_auto_trade_enabled=strategy_auto_enabled,
        bar_ts=bar_ts,
        signal=signal,
        signal_note=signal_note,
        exchange_side=exchange_side,
        exchange_size=exchange_size,
        action=action,
        reason=blocked_reason or reason,
        execute_requested=execute,
        executed=executed,
        blocked=blocked,
        cooldown_active=cooldown_active,
        cooldown_until=state.get('cooldown_until'),
        bybit_ready=bybit.is_ready(),
        auto_enabled=AUTO_TRADE_ENABLED,
        testnet=BYBIT_TESTNET,
        state=_snapshot_runtime(symbol, tf),
        position=position_snapshot,
        execution=execution_result,
        strategy_params=strategy_kwargs,
        trade_params=trade_params,
        fatal_block=fatal_block,
        safety_state='BLOCKED' if fatal_block else ('READY' if execute else 'PREVIEW'),
    ).to_dict()
    _log_event(payload)
    return payload




def get_auto_loop_status() -> Dict[str, Any]:
    with _LOOP_LOCK:
        return dict(_LOOP_STATE)


def _loop_runner() -> None:
    while not _LOOP_STOP.is_set():
        cfg = get_auto_loop_status()
        poll_sec = float(cfg.get('poll_sec') or AUTO_TRADE_POLL_SEC or 5)
        loop_state = str(cfg.get('state') or ('RUNNING' if cfg.get('running') else 'STOPPED'))
        try:
            if loop_state == 'BLOCKED':
                with _LOOP_LOCK:
                    _LOOP_STATE['last_tick_at'] = time.time()
                _LOOP_STOP.wait(max(1.0, poll_sec))
                continue
            preview = process_symbol(
                symbol=str(cfg.get('symbol') or ''),
                tf=str(cfg.get('tf') or ''),
                execute=False,
                opt_id=cfg.get('opt_id'),
                trade_params=dict(cfg.get('trade_params') or {}),
            )
            with _LOOP_LOCK:
                _LOOP_STATE['last_tick_at'] = time.time()
                _LOOP_STATE['last_result'] = preview
                _LOOP_STATE['last_error'] = preview.get('error') if isinstance(preview, dict) else None
            if not isinstance(preview, dict) or not preview.get('ok'):
                _LOOP_STOP.wait(max(1.0, poll_sec))
                continue
            if loop_state == 'ARMED_WAIT_NEXT_BAR':
                armed_bar_ts = int(cfg.get('armed_bar_ts') or 0)
                current_bar_ts = int(preview.get('bar_ts') or 0)
                if current_bar_ts <= armed_bar_ts:
                    preview = dict(preview)
                    preview['reason'] = f'armed: waiting for next bar after {armed_bar_ts}'
                    preview['mode'] = 'armed_wait_next_bar'
                    with _LOOP_LOCK:
                        _LOOP_STATE['last_result'] = preview
                    _LOOP_STOP.wait(max(1.0, poll_sec))
                    continue
                with _LOOP_LOCK:
                    _LOOP_STATE['state'] = 'RUNNING'
                    _LOOP_STATE['armed_bar_ts'] = current_bar_ts
            result = process_symbol(
                symbol=str(cfg.get('symbol') or ''),
                tf=str(cfg.get('tf') or ''),
                execute=True,
                opt_id=cfg.get('opt_id'),
                trade_params=dict(cfg.get('trade_params') or {}),
            )
            with _LOOP_LOCK:
                _LOOP_STATE['last_tick_at'] = time.time()
                _LOOP_STATE['last_result'] = result
                _LOOP_STATE['last_error'] = result.get('error') if isinstance(result, dict) else None
                if isinstance(result, dict) and result.get('fatal_block'):
                    _LOOP_STATE['state'] = 'BLOCKED'
                    _LOOP_STATE['blocked_reason'] = result.get('reason') or result.get('error')
        except Exception as e:
            with _LOOP_LOCK:
                _LOOP_STATE['last_tick_at'] = time.time()
                _LOOP_STATE['last_error'] = repr(e)
                _LOOP_STATE['state'] = 'BLOCKED'
                _LOOP_STATE['blocked_reason'] = repr(e)
            _log_event({'ok': False, 'mode': 'loop', 'error': repr(e)})
        _LOOP_STOP.wait(max(1.0, poll_sec))


def start_auto_loop(symbol: str, tf: str, *, opt_id: int, poll_sec: float | None = None, trade_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _LOOP_THREAD
    symbol = (symbol or '').strip().upper()
    tf = str(tf or '').strip()
    if not symbol or not tf or not opt_id:
        return {'ok': False, 'error': 'symbol, tf, opt_id are required'}
    if not AUTO_TRADE_ENABLED:
        return {'ok': False, 'error': 'AUTO_TRADE_ENABLED is false'}
    if AUTO_TRADE_TESTNET_ONLY and not BYBIT_TESTNET:
        return {'ok': False, 'error': 'testnet_only enabled, but Bybit client is not on testnet'}
    probe = process_symbol(symbol=symbol, tf=tf, execute=False, opt_id=int(opt_id), trade_params=dict(trade_params or {}))
    if not probe.get('ok'):
        return {'ok': False, 'error': probe.get('error') or 'strategy probe failed', 'probe': probe}
    if not bool(probe.get('strategy_auto_trade_enabled')):
        return {'ok': False, 'error': 'selected strategy is not allowed for auto trading', 'probe': probe}
    initial_state = 'ARMED_WAIT_NEXT_BAR' if AUTO_LOOP_WAIT_NEXT_BAR_ON_START else 'RUNNING'
    with _LOOP_LOCK:
        if _LOOP_STATE.get('running'):
            return {'ok': False, 'error': 'auto loop already running', 'loop': dict(_LOOP_STATE)}
        _LOOP_STOP.clear()
        _LOOP_STATE.update({
            'running': True,
            'symbol': symbol,
            'tf': tf,
            'opt_id': int(opt_id),
            'poll_sec': float(poll_sec or AUTO_TRADE_POLL_SEC or 5),
            'started_at': time.time(),
            'last_tick_at': None,
            'last_result': probe,
            'last_error': None,
            'trade_params': _resolve_auto_trade_params(dict(trade_params or {})),
            'state': initial_state,
            'armed_bar_ts': int(probe.get('bar_ts') or 0),
            'blocked_reason': None,
        })
        _LOOP_THREAD = threading.Thread(target=_loop_runner, name='strategy3-auto-loop', daemon=True)
        _LOOP_THREAD.start()
        payload = dict(_LOOP_STATE)
    _log_event({'ok': True, 'mode': 'loop_start', 'loop': payload})
    return {'ok': True, 'loop': payload, 'reason': 'armed and waiting next bar' if initial_state == 'ARMED_WAIT_NEXT_BAR' else 'loop started'}


def stop_auto_loop() -> Dict[str, Any]:
    global _LOOP_THREAD
    with _LOOP_LOCK:
        was_running = bool(_LOOP_STATE.get('running'))
        prev = dict(_LOOP_STATE)
        _LOOP_STATE['running'] = False
        _LOOP_STATE['state'] = 'STOPPED'
    _LOOP_STOP.set()
    thr = _LOOP_THREAD
    if thr and thr.is_alive():
        thr.join(timeout=2.0)
    _LOOP_THREAD = None
    with _LOOP_LOCK:
        _LOOP_STATE['running'] = False
        _LOOP_STATE['state'] = 'STOPPED'
    _log_event({'ok': True, 'mode': 'loop_stop', 'was_running': was_running, 'loop': prev})
    return {'ok': True, 'stopped': was_running, 'loop': dict(_LOOP_STATE)}


def get_last_auto_events(limit: int = 20) -> Dict[str, Any]:
    lim = max(1, min(int(limit or 20), len(_EVENTS) or 1))
    return {'ok': True, 'events': list(_EVENTS)[:lim], 'config': get_auto_config()}

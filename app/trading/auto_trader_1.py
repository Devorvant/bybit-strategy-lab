
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
from .strategy_registry import get_opt_strategy_entry, get_risk_profile


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
AUTO_TRADE_POLL_SEC = float(os.getenv('AUTO_TRADE_POLL_SEC', '10'))

AUTO_DEFAULT_LEVERAGE = int(os.getenv('AUTO_DEFAULT_LEVERAGE', str(DEFAULT_LEVERAGE or 1)))
AUTO_MAX_LEVERAGE = int(os.getenv('AUTO_MAX_LEVERAGE', str(max(1, AUTO_DEFAULT_LEVERAGE))))
AUTO_POSITION_MODE = os.getenv('AUTO_POSITION_MODE', 'fixed_usd').strip().lower() or 'fixed_usd'
AUTO_POSITION_USD = float(os.getenv('AUTO_POSITION_USD', str(DEFAULT_POSITION_USD or 25)))
AUTO_MAX_MARGIN_USDT = float(os.getenv('AUTO_MAX_MARGIN_USDT', str(AUTO_POSITION_USD)))
AUTO_REQUIRE_SL = _bool_env('AUTO_REQUIRE_SL', True)
AUTO_REQUIRE_TP = _bool_env('AUTO_REQUIRE_TP', False)
AUTO_SL_PERCENT = float(os.getenv('AUTO_SL_PERCENT', '1.0'))
AUTO_TP_PERCENT = float(os.getenv('AUTO_TP_PERCENT', '0'))
AUTO_CLOSE_ON_SL_FAILURE = _bool_env('AUTO_CLOSE_ON_SL_FAILURE', True)
AUTO_BLOCK_ON_SL_FAILURE = _bool_env('AUTO_BLOCK_ON_SL_FAILURE', True)

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
    risk_profile_id: str | None = None
    risk_profile_name: str | None = None

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



def _safe_trade_params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    src = dict(overrides or {})
    lev = int(src.get('leverage') or AUTO_DEFAULT_LEVERAGE or 1)
    max_lev = int(src.get('max_leverage') or AUTO_MAX_LEVERAGE or lev or 1)
    lev = max(1, min(lev, max_lev))
    position_usd = float(src.get('position_usd') or AUTO_POSITION_USD or 0)
    max_margin = float(src.get('max_margin_usdt') or AUTO_MAX_MARGIN_USDT or position_usd or 0)
    if max_margin > 0 and position_usd > max_margin:
        position_usd = max_margin
    tp_val = src.get('tp_percent')
    tp_percent = float(tp_val) if tp_val not in (None, '', 0, '0') else (AUTO_TP_PERCENT if AUTO_REQUIRE_TP or AUTO_TP_PERCENT > 0 else None)
    sl_val = src.get('sl_percent')
    sl_percent = float(sl_val) if sl_val not in (None, '') else AUTO_SL_PERCENT
    return {
        'mode': src.get('mode') or AUTO_POSITION_MODE,
        'position_usd': position_usd,
        'max_margin_usdt': max_margin,
        'leverage': lev,
        'max_leverage': max_lev,
        'sl_percent': sl_percent,
        'tp_percent': tp_percent,
        'require_sl': bool(src.get('require_sl')) if 'require_sl' in src else AUTO_REQUIRE_SL,
        'require_tp': bool(src.get('require_tp')) if 'require_tp' in src else AUTO_REQUIRE_TP,
    }


def _profile_to_trade_params(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not profile:
        return {}
    return {
        'mode': profile.get('position_mode') or profile.get('mode') or AUTO_POSITION_MODE,
        'position_usd': profile.get('position_usd'),
        'max_margin_usdt': profile.get('max_margin_usdt'),
        'leverage': profile.get('default_leverage'),
        'max_leverage': profile.get('max_leverage'),
        'sl_percent': profile.get('sl_percent'),
        'tp_percent': profile.get('tp_percent'),
        'require_sl': profile.get('require_sl'),
        'require_tp': profile.get('require_tp'),
    }


def _resolve_preset_trade_params(opt_entry: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = None
    if opt_entry:
        profile = opt_entry.get('risk_profile')
        if not profile and opt_entry.get('risk_profile_id'):
            profile = get_risk_profile(opt_entry.get('risk_profile_id'))
    base = _profile_to_trade_params(profile)
    merged = dict(base)
    merged.update(dict(overrides or {}))
    return _safe_trade_params(merged)


def _verify_protection(position_snapshot: Optional[Dict[str, Any]], *, require_sl: bool, require_tp: bool) -> tuple[bool, Optional[str]]:
    if not position_snapshot or not position_snapshot.get('has_position'):
        return False, 'position_missing_after_entry'
    pos = position_snapshot.get('position') or {}
    sl = pos.get('stopLoss')
    tp = pos.get('takeProfit')
    if require_sl and not sl:
        return False, 'stop_loss_missing_after_entry'
    if require_tp and not tp:
        return False, 'take_profit_missing_after_entry'
    return True, None


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
        'default_position_usd': DEFAULT_POSITION_USD,
        'default_leverage': DEFAULT_LEVERAGE,
        'safe_trade_params': _safe_trade_params(),
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
    state = _get_runtime_state(symbol, tf)

    opt_entry = None
    strategy_name = None
    strategy_auto_enabled = None
    risk_profile_id = None
    risk_profile_name = None
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
        risk_profile_id = str(opt_entry.get('risk_profile_id') or '') or None
        risk_profile_name = str((opt_entry.get('risk_profile') or {}).get('name') or '') or None
        strategy_kwargs = _filter_strategy_params(opt_entry.get('best_params') or {})
        trade_params = _resolve_preset_trade_params(opt_entry, trade_params)

    trade_params = _safe_trade_params(trade_params)

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
    elif execute and cooldown_active:
        blocked_reason = f'cooldown active until {cooldown_until:.0f}'
    elif execute and int(trade_params.get('leverage') or 0) > int(trade_params.get('max_leverage') or 0 or AUTO_MAX_LEVERAGE):
        blocked_reason = 'leverage_limit_exceeded'
    elif execute and float(trade_params.get('position_usd') or 0) > float(trade_params.get('max_margin_usdt') or 0 or AUTO_MAX_MARGIN_USDT):
        blocked_reason = 'max_margin_exceeded'
    elif execute and action in {'OPEN_LONG', 'OPEN_SHORT'} and bool(trade_params.get('require_sl')) and float(trade_params.get('sl_percent') or 0) <= 0:
        blocked_reason = 'sl_required_but_missing'
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
            if action in {'OPEN_LONG', 'OPEN_SHORT'}:
                time.sleep(1.0)
                verified_snapshot = bybit.get_position_snapshot(symbol)
                protection_ok, protection_error = _verify_protection(
                    verified_snapshot,
                    require_sl=bool(trade_params.get('require_sl')),
                    require_tp=bool(trade_params.get('require_tp')),
                )
                position_snapshot = verified_snapshot
                exchange_side, exchange_size = _normalize_position_side(position_snapshot)
                if not protection_ok:
                    if AUTO_CLOSE_ON_SL_FAILURE and position_snapshot.get('has_position'):
                        emergency = execute_action(bybit, symbol=symbol, action='CLOSE', leverage=trade_params.get('leverage'))
                        execution_result = {
                            **dict(execution_result or {}),
                            'protection_error': protection_error,
                            'post_entry_position': position_snapshot,
                            'emergency_close': emergency,
                        }
                        position_snapshot = bybit.get_position_snapshot(symbol)
                        exchange_side, exchange_size = _normalize_position_side(position_snapshot)
                    blocked = bool(AUTO_BLOCK_ON_SL_FAILURE)
                    executed = False
                    blocked_reason = protection_error or 'post_entry_protection_failed'
        if not executed and blocked_reason is None:
            blocked = True
            blocked_reason = execution_result.get('error') or execution_result.get('note') or 'execution failed'
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
        try:
            symbol = str(cfg.get('symbol') or '')
            tf = str(cfg.get('tf') or '')
            opt_id = cfg.get('opt_id')
            tp = dict(cfg.get('trade_params') or {})
            loop_state = str(cfg.get('state') or 'RUNNING').upper()
            armed_bar_ts = cfg.get('armed_bar_ts')
            preview = process_symbol(symbol=symbol, tf=tf, execute=False, opt_id=opt_id, trade_params=tp)
            now = time.time()
            result = preview
            if loop_state == 'ARMED_WAIT_NEXT_BAR':
                cur_bar_ts = preview.get('bar_ts')
                if cur_bar_ts is not None and armed_bar_ts is not None and int(cur_bar_ts) > int(armed_bar_ts):
                    with _LOOP_LOCK:
                        _LOOP_STATE['state'] = 'RUNNING'
                        _LOOP_STATE['armed_bar_ts'] = int(cur_bar_ts)
                    result = process_symbol(symbol=symbol, tf=tf, execute=True, opt_id=opt_id, trade_params=tp)
                    _log_event({'ok': True, 'mode': 'loop', 'event': 'new_bar_detected', 'bar_ts': cur_bar_ts, 'symbol': symbol, 'tf': tf})
                else:
                    _log_event({'ok': True, 'mode': 'loop', 'event': 'armed_waiting_same_bar', 'bar_ts': cur_bar_ts, 'armed_bar_ts': armed_bar_ts, 'symbol': symbol, 'tf': tf})
            else:
                result = process_symbol(symbol=symbol, tf=tf, execute=True, opt_id=opt_id, trade_params=tp)
            with _LOOP_LOCK:
                _LOOP_STATE['last_tick_at'] = now
                _LOOP_STATE['last_result'] = result
                _LOOP_STATE['last_error'] = result.get('error') if isinstance(result, dict) else None
                if isinstance(result, dict) and result.get('blocked') and result.get('reason') in {'stop_loss_missing_after_entry', 'take_profit_missing_after_entry', 'position_missing_after_entry'}:
                    _LOOP_STATE['state'] = 'BLOCKED'
                    _LOOP_STATE['blocked_reason'] = result.get('reason')
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
    opt_entry = get_opt_strategy_entry(int(opt_id))
    if not opt_entry:
        return {'ok': False, 'error': 'strategy preset not found'}
    safe_trade_params = _resolve_preset_trade_params(opt_entry, trade_params)
    probe = process_symbol(symbol=symbol, tf=tf, execute=False, opt_id=int(opt_id), trade_params=safe_trade_params)
    if not probe.get('ok'):
        return {'ok': False, 'error': probe.get('error') or 'strategy probe failed', 'probe': probe}
    if not bool(probe.get('strategy_auto_trade_enabled')):
        return {'ok': False, 'error': 'selected strategy is not allowed for auto trading', 'probe': probe}
    state = _get_runtime_state(symbol, tf)
    state['last_execute_bar_ts'] = None
    state['last_execute_action'] = None
    state['last_execution_ok'] = None
    state['cooldown_until'] = 0.0
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
            'trade_params': safe_trade_params,
            'state': 'ARMED_WAIT_NEXT_BAR',
            'armed_bar_ts': probe.get('bar_ts'),
            'blocked_reason': None,
        })
        _LOOP_THREAD = threading.Thread(target=_loop_runner, name='strategy3-auto-loop', daemon=True)
        _LOOP_THREAD.start()
        payload = dict(_LOOP_STATE)
    _log_event({'ok': True, 'mode': 'loop_start', 'loop': payload, 'reason': 'armed and waiting next bar'})
    return {'ok': True, 'loop': payload, 'reason': 'armed and waiting next bar'}


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


from __future__ import annotations

import inspect
import json
import os
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

_CONN = None
_RUNTIME: Dict[str, Dict[str, Any]] = {}
_EVENTS: deque[dict] = deque(maxlen=max(10, AUTO_TRADE_MAX_EVENTS))


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
        'bybit_testnet': BYBIT_TESTNET,
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
    strategy_params: Optional[Dict[str, Any]] = None,
    trade_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = (symbol or '').strip().upper()
    tf = str(tf or getattr(settings, 'TF', '120')).strip()
    strategy_kwargs = _filter_strategy_params(strategy_params)
    trade_params = dict(trade_params or {})
    state = _get_runtime_state(symbol, tf)

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

    if symbol not in AUTO_TRADE_ALLOWED_SYMBOLS:
        blocked_reason = f'symbol not allowed: {symbol}'
    elif tf.upper() not in AUTO_TRADE_ALLOWED_TFS:
        blocked_reason = f'tf not allowed: {tf}'
    elif AUTO_TRADE_TESTNET_ONLY and not BYBIT_TESTNET:
        blocked_reason = 'testnet_only enabled, but Bybit client is not on testnet'
    elif execute and not AUTO_TRADE_ENABLED:
        blocked_reason = 'AUTO_TRADE_ENABLED is false'
    elif execute and not bybit.is_ready():
        blocked_reason = 'Bybit is not configured'
    elif execute and cooldown_active:
        blocked_reason = f'cooldown active until {cooldown_until:.0f}'
    elif execute and not force and state.get('last_bar_ts') == bar_ts and state.get('last_action') == action and action != 'HOLD':
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
        if not executed:
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



def get_last_auto_events(limit: int = 20) -> Dict[str, Any]:
    lim = max(1, min(int(limit or 20), len(_EVENTS) or 1))
    return {'ok': True, 'events': list(_EVENTS)[:lim], 'config': get_auto_config()}

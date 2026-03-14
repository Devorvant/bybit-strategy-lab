from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from app.reporting.chart import _build_fig_and_bt
from app.trading.auto_trader import get_auto_config, get_auto_loop_status
from app.trading.bybit_client import BybitClient

from .chart_repo import TradeChartRepo


LIVE_ENTRY_KINDS = {"manual_long", "manual_short", "auto_long", "auto_short"}
LIVE_CLOSE_KINDS = {"manual_close", "auto_close", "exchange_external_close", "exchange_tp_sl_close"}
_EXEC_RESULT_HINTS = ("_result", "_filled", "_done", "_completed")
_EXEC_REQUEST_HINTS = ("_requested", "_request")


class TradeChartService:
    def __init__(self, repo: TradeChartRepo | None = None):
        self.repo = repo or TradeChartRepo()

    def build_trade_chart_payload(
        self,
        *,
        symbol: str,
        tf: str,
        strategy: str,
        last: int | None = None,
        limit: int = 2000,
        opt_id: int | None = None,
        capital_usd: float = 10000.0,
        signal_mode: str = "legacy",
    ) -> dict[str, Any]:
        warnings: list[str] = []
        symbol = str(symbol or "").upper().strip()
        tf = str(tf or "").strip()
        strategy = str(strategy or "my_strategy3.py").strip()
        limit = max(50, min(int(limit or 2000), 50000))

        bars = self.repo.fetch_bars(symbol=symbol, tf=tf, limit=limit)
        if not bars:
            return {
                "ok": False,
                "error": "no bars for selected symbol/tf",
                "context": {
                    "symbol": symbol,
                    "tf": tf,
                    "strategy": strategy,
                    "opt_id": opt_id,
                    "last": last,
                    "limit": limit,
                    "mode": "strategy_only",
                    "generated_at_ts": _now_ms(),
                },
            }

        strategy_params = self.repo.fetch_opt_params(symbol=symbol, tf=tf, strategy=strategy, opt_id=opt_id)
        fig, bt, err_html = _build_fig_and_bt(
            bars,
            strategy=strategy,
            strategy_params=strategy_params,
            capital_usd=capital_usd,
            signal_mode=signal_mode,
        )
        if err_html or bt is None:
            return {
                "ok": False,
                "error": "strategy backtest/chart build failed",
                "error_html": err_html,
            }

        chart_block = self._build_chart_block(bars=bars, bt=bt, strategy=strategy)
        strategy_trades = self._build_strategy_trades(bt)
        strategy_events = self._build_strategy_events(bt=bt, bars=bars, strategy=strategy)
        strategy_summary = self._build_strategy_summary(bt=bt)

        live_status = self._build_live_status(symbol=symbol, opt_id=opt_id)
        executions = self.repo.fetch_execution_events(symbol=symbol, tf=tf, limit=1000)
        snapshots = self.repo.fetch_exchange_snapshots(symbol=symbol, tf=tf, limit=1000)
        live_events = self._build_live_events(bars=bars, executions=executions, snapshots=snapshots, limit=1000)
        live_summary = self._build_live_summary(live_events)

        journal_available = self.repo.table_exists("execution_events") or self.repo.table_exists("exchange_snapshots")
        has_live = bool(live_events) or bool(executions) or bool(snapshots) or bool(live_status.get("current_trading_strategy"))
        mode = "compare" if has_live else "strategy_only"
        if journal_available and not live_events:
            warnings.append("Trade journal tables exist, but no normalized live events were produced for current symbol/tf")
        elif not has_live:
            warnings.append("Live trading not available for selected symbol/strategy")

        compare_rows = self._build_compare_rows(strategy_trades=strategy_trades, live_events=live_events)
        discrepancies = self._build_discrepancies(compare_rows=compare_rows, live_status=live_status)

        return {
            "ok": True,
            "context": {
                "symbol": symbol,
                "tf": tf,
                "strategy": strategy,
                "opt_id": opt_id,
                "last": last,
                "limit": limit,
                "mode": mode,
                "generated_at_ts": _now_ms(),
            },
            "status": {
                "strategy_signal": live_status.get("strategy_signal", "UNKNOWN"),
                "strategy_action": live_status.get("strategy_action", "UNKNOWN"),
                "exchange_side": live_status.get("exchange_side", "UNKNOWN"),
                "current_trading_strategy": live_status.get("current_trading_strategy"),
                "preset_id": live_status.get("preset_id", opt_id),
                "auto_enabled": live_status.get("auto_enabled", False),
                "loop_state": live_status.get("loop_state", "STOPPED"),
                "risk_profile": live_status.get("risk_profile"),
                "bybit_ready": live_status.get("bybit_ready", False),
                "last_sync_ts": live_status.get("last_sync_ts"),
                "journal_available": journal_available,
                "compare_available": has_live,
                "live_available": has_live,
                "raw_execution_events": len(executions),
                "raw_snapshot_events": len(snapshots),
            },
            "chart": chart_block,
            "strategy": {
                "summary": strategy_summary,
                "events": strategy_events,
                "trades": strategy_trades,
                "events_count": len(strategy_events),
                "trades_count": len(strategy_trades),
            },
            "live": {
                "available": has_live,
                "summary": live_summary,
                "events": live_events,
                "events_count": len(live_events),
                "raw_execution_events": len(executions),
                "raw_snapshot_events": len(snapshots),
            },
            "compare": {
                "available": has_live,
                "summary": self._build_compare_summary(compare_rows),
                "rows": compare_rows,
                "rows_count": len(compare_rows),
            },
            "discrepancies": {
                "available": has_live,
                "rows": discrepancies,
                "rows_count": len(discrepancies),
            },
            "ui": {
                "default_tab": "strategy_trades",
                "available_tabs": ["strategy_trades", "live_trades", "compare", "discrepancies", "debug"] if has_live else ["strategy_trades", "debug"],
                "default_layers": {
                    "price": True,
                    "indicators": True,
                    "strategy_events": True,
                    "strategy_trades": True,
                    "live_events": has_live,
                    "manual_events": has_live,
                    "auto_events": has_live,
                    "external_close_events": has_live,
                    "tp_sl_events": has_live,
                    "equity": True,
                    "discrepancies": False,
                },
                "show_live_by_default": has_live,
                "mode": mode,
            },
            "warnings": warnings,
        }

    def build_trade_events_payload(self, *, symbol: str, tf: str, limit: int = 500) -> dict[str, Any]:
        bars = self.repo.fetch_bars(symbol=symbol, tf=tf, limit=max(200, limit))
        executions = self.repo.fetch_execution_events(symbol=symbol, tf=tf, limit=limit)
        snapshots = self.repo.fetch_exchange_snapshots(symbol=symbol, tf=tf, limit=limit)
        events = self._build_live_events(bars=bars, executions=executions, snapshots=snapshots, limit=limit)
        return {
            "ok": True,
            "symbol": symbol,
            "tf": tf,
            "count": len(events),
            "events": events,
            "raw_execution_events": len(executions),
            "raw_snapshot_events": len(snapshots),
        }

    def build_trade_status_payload(self, *, symbol: str, opt_id: int | None = None) -> dict[str, Any]:
        return {"ok": True, "status": self._build_live_status(symbol=symbol, opt_id=opt_id)}

    def _build_chart_block(self, *, bars: list[tuple[int, float, float, float, float, float]], bt: Any, strategy: str) -> dict[str, Any]:
        out_bars = [
            {"ts": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v)}
            for ts, o, h, l, c, v in bars
        ]
        indicators: dict[str, list[dict[str, Any]]] = {}
        if hasattr(bt, "st_line"):
            indicators["supertrend"] = [
                {"ts": int(bars[i][0]), "value": _float_or_none(v), "dir": _int_or_none(bt.st_dir[i]) if hasattr(bt, "st_dir") else None}
                for i, v in enumerate(list(getattr(bt, "st_line", []) or []))
                if i < len(bars)
            ]
        if hasattr(bt, "adx"):
            indicators["adx"] = [
                {"ts": int(bars[i][0]), "value": _float_or_none(v)}
                for i, v in enumerate(list(getattr(bt, "adx", []) or []))
                if i < len(bars)
            ]
        if hasattr(bt, "no_trade"):
            indicators["no_trade"] = [
                {"ts": int(bars[i][0]), "value": bool(v)}
                for i, v in enumerate(list(getattr(bt, "no_trade", []) or []))
                if i < len(bars) and bool(v)
            ]
        equity = [{"ts": int(ts), "value": float(eq)} for ts, eq in zip(list(getattr(bt, "equity_ts", []) or []), list(getattr(bt, "equity", []) or []))]
        return {"range": {"from_ts": int(bars[0][0]), "to_ts": int(bars[-1][0])}, "bars": out_bars, "indicators": indicators, "equity": equity}

    def _build_strategy_trades(self, bt: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, tr in enumerate(list(getattr(bt, "trades", []) or []), start=1):
            rows.append(
                {
                    "trade_id": idx,
                    "status": "closed",
                    "side": str(getattr(tr, "side", "")).upper() or None,
                    "reason": getattr(tr, "exit_reason", None),
                    "entry_ts": int(getattr(tr, "entry_ts", 0) or 0),
                    "exit_ts": int(getattr(tr, "exit_ts", 0) or 0),
                    "entry_price": _float_or_none(getattr(tr, "entry_price", None)),
                    "exit_price": _float_or_none(getattr(tr, "exit_price", None)),
                    "pnl_usd": _float_or_none(getattr(tr, "pnl", None)),
                    "cum_pnl_usd": _float_or_none(getattr(tr, "cum_pnl", None)),
                }
            )
        rows.sort(key=lambda x: x.get("entry_ts") or 0, reverse=True)
        return rows

    def _build_strategy_events(self, *, bt: Any, bars: list[tuple[int, float, float, float, float, float]], strategy: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ts_to_close = {int(ts): float(c) for ts, _, _, _, c, _ in bars}
        for idx, tr in enumerate(list(getattr(bt, "trades", []) or []), start=1):
            side = str(getattr(tr, "side", "")).upper()
            entry_kind = "strategy_long_entry" if side == "LONG" else "strategy_short_entry"
            exit_side = "Sell" if side == "LONG" else "Buy"
            entry_ts = int(getattr(tr, "entry_ts", 0) or 0)
            exit_ts = int(getattr(tr, "exit_ts", 0) or 0)
            events.append({
                "id": f"st_{idx}_entry",
                "ts": entry_ts,
                "bar_ts": entry_ts,
                "group": "strategy",
                "kind": entry_kind,
                "display_category": "entry",
                "side": "Buy" if side == "LONG" else "Sell",
                "price": _float_or_none(getattr(tr, "entry_price", None)) or ts_to_close.get(entry_ts),
                "reason": getattr(tr, "exit_reason", None),
                "trade_id": idx,
            })
            events.append({
                "id": f"st_{idx}_exit",
                "ts": exit_ts,
                "bar_ts": exit_ts,
                "group": "strategy",
                "kind": "strategy_exit",
                "display_category": "exit",
                "side": exit_side,
                "price": _float_or_none(getattr(tr, "exit_price", None)) or ts_to_close.get(exit_ts),
                "reason": getattr(tr, "exit_reason", None),
                "trade_id": idx,
            })

        if strategy == "my_strategy3.py" and hasattr(bt, "no_trade"):
            nt = list(getattr(bt, "no_trade", []) or [])
            for i, is_no_trade in enumerate(nt):
                if not is_no_trade or i >= len(bars):
                    continue
                ts, _, _, _, c, _ = bars[i]
                events.append({
                    "id": f"st_no_trade_{ts}",
                    "ts": int(ts),
                    "bar_ts": int(ts),
                    "group": "strategy",
                    "kind": "strategy_no_trade",
                    "display_category": "state",
                    "side": None,
                    "price": float(c),
                    "reason": "NO_TRADE",
                    "trade_id": None,
                })

        events.sort(key=lambda x: x["ts"])
        return events

    def _build_strategy_summary(self, *, bt: Any) -> dict[str, Any]:
        trades = list(getattr(bt, "trades", []) or [])
        wins = sum(1 for tr in trades if float(getattr(tr, "pnl", 0.0) or 0.0) > 0)
        eq = [float(x) for x in list(getattr(bt, "equity", []) or [])]
        realized = float(getattr(trades[-1], "cum_pnl", 0.0)) if trades else 0.0
        avg_trade = (realized / len(trades)) if trades else 0.0
        winrate = (wins / len(trades) * 100.0) if trades else 0.0
        max_dd_pct = _max_drawdown_pct(eq)
        profit_pct = _ret_pct(eq)
        return {
            "trades_total": len(trades),
            "winrate": round(winrate, 4),
            "realized_pnl_usd": round(realized, 8),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "avg_trade_pnl_usd": round(avg_trade, 8),
            "profit_pct": round(profit_pct, 4),
            "breakeven_pct": round(100.0 - winrate, 4) if trades else 0.0,
            "volatility_pct": round(_equity_vol_pct(eq), 4),
        }

    def _build_live_status(self, *, symbol: str, opt_id: int | None = None) -> dict[str, Any]:
        status: dict[str, Any] = {}
        try:
            bybit = BybitClient()
            raw = bybit.status(symbol) if bybit.is_ready() else {}
            position = bybit.get_position_snapshot(symbol) if bybit.is_ready() else {}
            status["bybit_ready"] = bool(bybit.is_ready())
            status["exchange_side"] = str((position or {}).get("side") or (raw or {}).get("side") or "FLAT").upper()
        except Exception:
            status["bybit_ready"] = False
            status["exchange_side"] = "UNKNOWN"

        try:
            auto_cfg = get_auto_config()
            loop = get_auto_loop_status()
            status["auto_enabled"] = bool(auto_cfg.get("auto_enabled"))
            status["loop_state"] = str(loop.get("state") or "STOPPED").upper()
            status["strategy_signal"] = str(loop.get("last_signal") or loop.get("signal") or "UNKNOWN").upper()
            status["strategy_action"] = str(loop.get("last_action") or loop.get("action") or "UNKNOWN").upper()
            status["last_sync_ts"] = _millis_from_mixed(loop.get("updated_at") or loop.get("ts"))
        except Exception:
            status.setdefault("auto_enabled", False)
            status.setdefault("loop_state", "STOPPED")
            status.setdefault("strategy_signal", "UNKNOWN")
            status.setdefault("strategy_action", "UNKNOWN")
            status.setdefault("last_sync_ts", None)

        opt_entry = self.repo.fetch_opt_strategy_entry(opt_id) if opt_id else None
        status["preset_id"] = opt_id
        status["current_trading_strategy"] = (opt_entry or {}).get("strategy") or None
        status["risk_profile"] = (opt_entry or {}).get("risk_profile_id") or (opt_entry or {}).get("risk_profile")
        return status

    def _build_live_events(
        self,
        *,
        bars: list[tuple[int, float, float, float, float, float]],
        executions: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        events.extend(_normalize_execution_events(executions, bars))
        events.extend(_derive_snapshot_events(snapshots, bars))
        events.sort(key=lambda x: (x.get("ts") or 0, str(x.get("id") or "")))
        return events[-limit:]

    def _build_live_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        kinds = [e.get("kind") for e in events]
        return {
            "events_total": len(events),
            "manual_count": sum(1 for k in kinds if isinstance(k, str) and k.startswith("manual_")),
            "auto_count": sum(1 for k in kinds if isinstance(k, str) and k.startswith("auto_")),
            "external_close_count": sum(1 for k in kinds if k == "exchange_external_close"),
            "tp_sl_close_count": sum(1 for k in kinds if k == "exchange_tp_sl_close"),
            "opens_count": sum(1 for k in kinds if k in LIVE_ENTRY_KINDS),
            "closes_count": sum(1 for k in kinds if k in LIVE_CLOSE_KINDS),
        }

    def _build_compare_rows(self, *, strategy_trades: list[dict[str, Any]], live_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        live_entries = [e for e in live_events if e.get("kind") in LIVE_ENTRY_KINDS]
        live_closes = [e for e in live_events if e.get("kind") in LIVE_CLOSE_KINDS]
        rows: list[dict[str, Any]] = []
        for tr in strategy_trades:
            expected_side = "Buy" if tr.get("side") == "LONG" else "Sell"
            entry = _find_nearest_entry(tr.get("entry_ts"), expected_side, live_entries)
            exit_ev = _find_next_close(entry.get("ts") if entry else tr.get("entry_ts"), live_closes)
            status = _infer_match_status(entry, exit_ev)
            rows.append(
                {
                    "id": f"cmp_{tr['trade_id']}",
                    "strategy_trade_id": tr["trade_id"],
                    "strategy_side": tr.get("side"),
                    "strategy_entry_ts": tr.get("entry_ts"),
                    "strategy_entry_price": tr.get("entry_price"),
                    "actual_entry_ts": entry.get("ts") if entry else None,
                    "actual_entry_price": entry.get("price") if entry else None,
                    "actual_exit_ts": exit_ev.get("ts") if exit_ev else None,
                    "actual_exit_price": exit_ev.get("price") if exit_ev else None,
                    "expected_result": tr.get("reason"),
                    "actual_result": exit_ev.get("kind") if exit_ev else None,
                    "match_status": status,
                    "comment": _compare_comment(status),
                }
            )
        return rows

    def _build_compare_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "matched": 0,
            "entry_delayed": 0,
            "exit_delayed": 0,
            "missed_execution": 0,
            "external_close": 0,
            "manual_override": 0,
            "state_mismatch": 0,
            "orphan_live_trade": 0,
        }
        for row in rows:
            status = row.get("match_status")
            if status in summary:
                summary[status] += 1
        return summary

    def _build_discrepancies(self, *, compare_rows: list[dict[str, Any]], live_status: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in compare_rows:
            status = row.get("match_status")
            if status in (None, "matched"):
                continue
            ts = row.get("actual_exit_ts") or row.get("strategy_entry_ts")
            out.append(
                {
                    "id": f"disc_{row['strategy_trade_id']}",
                    "ts": ts,
                    "bar_ts": ts,
                    "group": "discrepancy",
                    "kind": _discrepancy_kind(status),
                    "type": status,
                    "severity": "warning",
                    "strategy_state": row.get("strategy_side"),
                    "exchange_state": live_status.get("exchange_side"),
                    "description": row.get("comment"),
                    "related_trade_id": row.get("strategy_trade_id"),
                }
            )
        out.sort(key=lambda x: x.get("ts") or 0, reverse=True)
        return out


def build_trade_chart_payload(**kwargs) -> dict[str, Any]:
    return TradeChartService().build_trade_chart_payload(**kwargs)


def build_trade_events_payload(**kwargs) -> dict[str, Any]:
    return TradeChartService().build_trade_events_payload(**kwargs)


def build_trade_status_payload(**kwargs) -> dict[str, Any]:
    return TradeChartService().build_trade_status_payload(**kwargs)


# ------------------- helpers -------------------

def _normalize_execution_events(rows: list[dict[str, Any]], bars: list[tuple[int, float, float, float, float, float]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}

    for row in rows:
        event_type = str(row.get("event_type") or "").lower()
        req = str(row.get("requested_action") or "").upper().strip()
        ts = _millis_from_mixed(row.get("ts"))
        if ts is None or not req:
            continue
        prefix = _execution_prefix(event_type)
        if not prefix:
            continue
        kind = _action_to_kind(prefix, req)
        if not kind:
            continue
        key = (ts, kind, str(row.get("order_id") or ""))
        grouped.setdefault(key, []).append(row)

    for (ts, kind, _), bucket in grouped.items():
        chosen = _pick_best_execution_row(bucket)
        price = _float_or_none(chosen.get("price"))
        if price is None:
            price = _nearest_bar_price(ts, bars)
        qty = _float_or_none(chosen.get("qty"))
        side = _normalize_side(chosen.get("side"))
        if side is None:
            side = "Buy" if "long" in kind else "Sell" if "short" in kind else _close_side_from_kind(kind)

        normalized.append(
            {
                "id": str(chosen.get("id") or f"exec_{ts}_{kind}"),
                "ts": ts,
                "bar_ts": _nearest_bar_ts(ts, bars),
                "group": "live",
                "kind": kind,
                "display_category": _display_category(kind),
                "source": "execution",
                "side": side,
                "price": price,
                "qty": qty,
                "order_id": chosen.get("order_id"),
                "exec_id": None,
                "note": chosen.get("event_type"),
            }
        )

    normalized.sort(key=lambda x: (x.get("ts") or 0, str(x.get("id") or "")))
    return normalized


def _derive_snapshot_events(rows: list[dict[str, Any]], bars: list[tuple[int, float, float, float, float, float]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda x: _millis_from_mixed(x.get("ts")) or 0)
    out: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for row in ordered:
        ts = _millis_from_mixed(row.get("ts"))
        if ts is None:
            prev = row
            continue
        cur_has = _row_has_position(row)
        prev_has = _row_has_position(prev) if prev else False
        if prev is not None and prev_has and not cur_has:
            kind = "exchange_tp_sl_close" if (prev.get("take_profit") or prev.get("stop_loss")) else "exchange_external_close"
            price = _float_or_none(row.get("avg_price") or row.get("mark_price") or prev.get("mark_price") or prev.get("avg_price"))
            if price is None:
                price = _nearest_bar_price(ts, bars)
            out.append(
                {
                    "id": f"snap_{row.get('id')}_{kind}",
                    "ts": ts,
                    "bar_ts": _nearest_bar_ts(ts, bars),
                    "group": "live",
                    "kind": kind,
                    "display_category": _display_category(kind),
                    "source": str(row.get("source") or "snapshot"),
                    "side": _close_side_from_prev(prev),
                    "price": price,
                    "qty": _float_or_none(prev.get("size")),
                    "order_id": None,
                    "exec_id": None,
                    "note": "position closed on exchange outside expected strategy flow",
                }
            )
        prev = row
    return out


def _pick_best_execution_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(row: dict[str, Any]) -> tuple[int, int, int]:
        event_type = str(row.get("event_type") or "").lower()
        is_result = int(any(h in event_type for h in _EXEC_RESULT_HINTS))
        has_price = int(_float_or_none(row.get("price")) is not None)
        row_id = int(row.get("id") or 0)
        return (is_result, has_price, row_id)

    return max(rows, key=rank)


def _execution_prefix(event_type: str) -> str | None:
    if "manual" in event_type:
        return "manual"
    if event_type.startswith("auto") or event_type.startswith("execute_") or "auto_" in event_type:
        return "auto"
    return None


def _row_has_position(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if _bool_from_mixed(row.get("has_position")):
        return float(row.get("size") or 0) > 0
    return float(row.get("size") or 0) > 0


def _find_nearest_entry(strategy_entry_ts: int | None, expected_side: str, entries: list[dict[str, Any]], max_delta_ms: int = 6 * 60 * 60 * 1000) -> dict[str, Any] | None:
    if strategy_entry_ts is None:
        return None
    candidates = [e for e in entries if e.get("side") == expected_side and abs((e.get("ts") or 0) - strategy_entry_ts) <= max_delta_ms]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs((x.get("ts") or 0) - strategy_entry_ts))


def _find_next_close(after_ts: int | None, closes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if after_ts is None:
        return None
    candidates = [e for e in closes if (e.get("ts") or 0) >= after_ts]
    return min(candidates, key=lambda x: x.get("ts") or 0) if candidates else None


def _infer_match_status(entry: dict[str, Any] | None, exit_ev: dict[str, Any] | None) -> str:
    if not entry:
        return "missed_execution"
    if exit_ev and exit_ev.get("kind") == "exchange_external_close":
        return "external_close"
    if exit_ev and exit_ev.get("kind") == "manual_close":
        return "manual_override"
    return "matched"


def _compare_comment(status: str) -> str:
    return {
        "matched": "Strategy trade matched with live execution flow",
        "missed_execution": "No matching live entry found for strategy trade",
        "external_close": "Entry matched, but exchange closed the position outside strategy flow",
        "manual_override": "Strategy flow diverged due to manual close",
    }.get(status, "Unclassified compare result")


def _discrepancy_kind(status: str) -> str:
    return {
        "missed_execution": "missed_execution",
        "manual_override": "manual_override",
        "external_close": "unexpected_position_change",
        "state_mismatch": "state_mismatch",
    }.get(status, "unexpected_position_change")


def _display_category(kind: str) -> str:
    return {
        "manual_long": "entry",
        "manual_short": "entry",
        "manual_close": "close",
        "auto_long": "entry",
        "auto_short": "entry",
        "auto_close": "close",
        "exchange_external_close": "external",
        "exchange_tp_sl_close": "tp_sl",
        "strategy_long_entry": "entry",
        "strategy_short_entry": "entry",
        "strategy_exit": "exit",
        "strategy_no_trade": "state",
    }.get(kind, "state")


def _action_to_kind(prefix: str, requested_action: str) -> str | None:
    if requested_action in {"LONG", "OPEN_LONG", "BUY"}:
        return f"{prefix}_long"
    if requested_action in {"SHORT", "OPEN_SHORT", "SELL"}:
        return f"{prefix}_short"
    if requested_action in {"CLOSE", "CLOSE_LONG", "CLOSE_SHORT"}:
        return f"{prefix}_close"
    return None


def _close_side_from_prev(prev: dict[str, Any]) -> str | None:
    side = str(prev.get("side") or "").lower()
    if side in {"buy", "long"}:
        return "Sell"
    if side in {"sell", "short"}:
        return "Buy"
    return None


def _close_side_from_kind(kind: str) -> str | None:
    if kind.endswith("_close"):
        return None
    return None


def _nearest_bar_ts(ts: int, bars: list[tuple[int, float, float, float, float, float]]) -> int:
    if not bars:
        return ts
    return min((int(row[0]) for row in bars), key=lambda bar_ts: abs(bar_ts - ts))


def _nearest_bar_price(ts: int, bars: list[tuple[int, float, float, float, float, float]]) -> float | None:
    if not bars:
        return None
    nearest = min(bars, key=lambda row: abs(int(row[0]) - ts))
    return _float_or_none(nearest[4])


def _normalize_side(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"buy", "long"}:
        return "Buy"
    if s in {"sell", "short"}:
        return "Sell"
    return None


def _bool_from_mixed(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _max_drawdown_pct(eq: list[float]) -> float:
    if not eq:
        return 0.0
    peak = eq[0]
    worst = 0.0
    for v in eq:
        if v > peak:
            peak = v
        if peak:
            dd = (v - peak) / peak
            if dd < worst:
                worst = dd
    return abs(worst) * 100.0


def _ret_pct(eq: list[float]) -> float:
    if len(eq) < 2:
        return 0.0
    start = eq[0] if abs(eq[0]) > 1e-9 else 10000.0
    cur = eq[-1] if abs(eq[0]) > 1e-9 else 10000.0 + eq[-1]
    return ((cur - start) / start) * 100.0 if start else 0.0


def _equity_vol_pct(eq: list[float]) -> float:
    if len(eq) < 3:
        return 0.0
    rets: list[float] = []
    for i in range(1, len(eq)):
        prev = float(eq[i - 1])
        cur = float(eq[i])
        if abs(prev) > 1e-9:
            rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return (var ** 0.5) * 100.0


def _float_or_none(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _millis_from_mixed(v: Any) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv > 10_000_000_000 else iv * 1000
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    s = str(v).strip()
    try:
        if s.isdigit():
            iv = int(s)
            return iv if iv > 10_000_000_000 else iv * 1000
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)

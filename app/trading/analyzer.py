from __future__ import annotations

from typing import Any


def _expected_side(action: str | None) -> str | None:
    a = str(action or "").upper()
    if a in ("LONG", "OPEN_LONG", "BUY"):
        return "Buy"
    if a in ("SHORT", "OPEN_SHORT", "SELL"):
        return "Sell"
    if a in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT"):
        return "Flat"
    return None


def _match_confidence(event: dict, snapshot: dict | None) -> str:
    if event.get("order_link_id"):
        return "high"
    if event.get("order_id"):
        return "high"
    if snapshot is not None:
        return "medium"
    return "low"


def classify_execution(event: dict, snapshot: dict | None) -> dict:
    action = str(event.get("requested_action") or "").upper()
    event_type = str(event.get("event_type") or "")
    ok = bool(event.get("ok"))

    if not ok:
        return {
            "status": "error",
            "reason": event.get("error") or "execution failed",
            "match_confidence": _match_confidence(event, snapshot),
        }

    if snapshot is None:
        return {
            "status": "unknown",
            "reason": "no near snapshot within 60s",
            "match_confidence": _match_confidence(event, snapshot),
        }

    expected = _expected_side(action)
    snap_side = snapshot.get("side")
    snap_size = snapshot.get("size") or 0
    has_position = snapshot.get("has_position")

    if expected == "Flat":
        if not has_position or float(snap_size or 0) == 0:
            return {
                "status": "confirmed",
                "reason": "position closed",
                "match_confidence": _match_confidence(event, snapshot),
            }
        return {
            "status": "mismatch",
            "reason": f"expected flat, got side={snap_side} size={snap_size}",
            "match_confidence": _match_confidence(event, snapshot),
        }

    if expected == "Buy":
        if str(snap_side or "").lower() == "buy" and float(snap_size or 0) > 0:
            return {
                "status": "confirmed",
                "reason": "long position confirmed",
                "match_confidence": _match_confidence(event, snapshot),
            }
        if float(snap_size or 0) == 0:
            return {
                "status": "no_effect",
                "reason": "long requested but position still flat",
                "match_confidence": _match_confidence(event, snapshot),
            }
        return {
            "status": "mismatch",
            "reason": f"expected long, got side={snap_side} size={snap_size}",
            "match_confidence": _match_confidence(event, snapshot),
        }

    if expected == "Sell":
        if str(snap_side or "").lower() == "sell" and float(snap_size or 0) > 0:
            return {
                "status": "confirmed",
                "reason": "short position confirmed",
                "match_confidence": _match_confidence(event, snapshot),
            }
        if float(snap_size or 0) == 0:
            return {
                "status": "no_effect",
                "reason": "short requested but position still flat",
                "match_confidence": _match_confidence(event, snapshot),
            }
        return {
            "status": "mismatch",
            "reason": f"expected short, got side={snap_side} size={snap_size}",
            "match_confidence": _match_confidence(event, snapshot),
        }

    return {
        "status": "unknown",
        "reason": f"unsupported action={action} event_type={event_type}",
        "match_confidence": _match_confidence(event, snapshot),
    }


def find_next_snapshot(
    conn,
    symbol: str,
    event_ts,
    source: str | None = None,
    within_seconds: int = 60,
) -> dict | None:
    sql = """
    SELECT
        id, ts, symbol, source, side, size, avg_price, mark_price,
        unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
        has_position, wallet_equity, wallet_balance, available_balance, payload
    FROM exchange_snapshots
    WHERE symbol = %(symbol)s
      AND ts >= %(event_ts)s
      AND ts <= (%(event_ts)s + (%(within_seconds)s || ' seconds')::interval)
      AND (%(source)s IS NULL OR source = %(source)s)
    ORDER BY ts ASC
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "symbol": symbol,
                "event_ts": event_ts,
                "within_seconds": within_seconds,
                "source": source,
            },
        )
        row = cur.fetchone()
        if row is None:
            return None

        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def load_recent_execution_events(conn, limit: int = 100) -> list[dict]:
    sql = """
    SELECT
        id,
        ts,
        run_id,
        symbol,
        tf,
        bar_ts,
        event_type,
        requested_action,
        side,
        qty,
        price,
        reduce_only,
        ok,
        error,
        response,
        order_id,
        order_link_id
    FROM execution_events
    ORDER BY id DESC
    LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"limit": limit})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def analyze_recent_events(conn, limit: int = 100, within_seconds: int = 60) -> list[dict]:
    events = load_recent_execution_events(conn, limit=limit)
    out: list[dict] = []

    for e in events:
        event_type = str(e.get("event_type") or "")

        if event_type.startswith("manual_"):
            source = "manual_after"
        elif event_type.startswith("execute_") or event_type.startswith("auto_"):
            source = "auto_after"
        else:
            source = None

        snap = find_next_snapshot(
            conn,
            e["symbol"],
            e["ts"],
            source=source,
            within_seconds=within_seconds,
        )
        verdict = classify_execution(e, snap)

        out.append(
            {
                "event_id": e.get("id"),
                "event_ts": e.get("ts"),
                "symbol": e.get("symbol"),
                "event_type": e.get("event_type"),
                "requested_action": e.get("requested_action"),
                "ok": e.get("ok"),
                "error": e.get("error"),
                "order_link_id": e.get("order_link_id"),
                "order_id": e.get("order_id"),
                "snapshot_id": snap.get("id") if snap else None,
                "snapshot_ts": snap.get("ts") if snap else None,
                "snapshot_source": snap.get("source") if snap else None,
                "snapshot_side": snap.get("side") if snap else None,
                "snapshot_size": snap.get("size") if snap else None,
                "snapshot_avg_price": snap.get("avg_price") if snap else None,
                "snapshot_tp": snap.get("take_profit") if snap else None,
                "snapshot_sl": snap.get("stop_loss") if snap else None,
                "status": verdict.get("status"),
                "reason": verdict.get("reason"),
                "match_confidence": verdict.get("match_confidence"),
            }
        )

    return out


def load_recent_snapshots(conn, symbol: str | None = None, limit: int = 200) -> list[dict]:
    sql = """
    SELECT
        id,
        ts,
        run_id,
        symbol,
        tf,
        bar_ts,
        source,
        side,
        size,
        avg_price,
        mark_price,
        unrealised_pnl,
        leverage,
        liq_price,
        take_profit,
        stop_loss,
        has_position,
        wallet_equity,
        wallet_balance,
        available_balance,
        payload
    FROM exchange_snapshots
    WHERE (%(symbol)s IS NULL OR symbol = %(symbol)s)
    ORDER BY ts DESC
    LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"symbol": symbol, "limit": limit})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def _is_close_event(e: dict) -> bool:
    action = str(e.get("requested_action") or "").upper()
    event_type = str(e.get("event_type") or "")
    if not bool(e.get("ok")):
        return False
    if action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT"):
        return True
    if event_type in ("manual_execute_result", "execute_result") and action.startswith("CLOSE"):
        return True
    return False


def load_execution_events_between(conn, symbol: str, ts_from, ts_to) -> list[dict]:
    sql = """
    SELECT
        id,
        ts,
        run_id,
        symbol,
        tf,
        bar_ts,
        event_type,
        requested_action,
        side,
        qty,
        price,
        reduce_only,
        ok,
        error,
        response,
        order_id,
        order_link_id
    FROM execution_events
    WHERE symbol = %(symbol)s
      AND ts >= %(ts_from)s
      AND ts <= %(ts_to)s
    ORDER BY ts ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"symbol": symbol, "ts_from": ts_from, "ts_to": ts_to})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def detect_exchange_close(prev_snapshot: dict | None, next_snapshot: dict | None, events_between: list[dict]) -> dict | None:
    if not prev_snapshot or not next_snapshot:
        return None

    prev_size = float(prev_snapshot.get("size") or 0)
    next_size = float(next_snapshot.get("size") or 0)

    prev_has = bool(prev_snapshot.get("has_position")) and prev_size > 0
    next_has = bool(next_snapshot.get("has_position")) and next_size > 0

    # Было открыто -> стало пусто
    if not prev_has:
        return None
    if next_has or next_size > 0:
        return None

    # Если между snapshot было наше закрытие — это не биржевое закрытие
    close_events = [e for e in events_between if _is_close_event(e)]
    if close_events:
        return None

    tp = prev_snapshot.get("take_profit")
    sl = prev_snapshot.get("stop_loss")

    if tp is not None or sl is not None:
        return {
            "status": "exchange_tp_sl_close",
            "reason": "position disappeared without our close; TP/SL existed on previous snapshot",
        }

    return {
        "status": "exchange_external_close",
        "reason": "position disappeared without our close and without TP/SL on previous snapshot",
    }


def analyze_exchange_side_closes(conn, symbol: str | None = None, limit: int = 200) -> list[dict]:
    snaps = load_recent_snapshots(conn, symbol=symbol, limit=limit)

    # Для анализа удобнее идти по времени вперёд
    snaps = list(reversed(snaps))

    out: list[dict] = []

    for i in range(len(snaps) - 1):
        prev_s = snaps[i]
        next_s = snaps[i + 1]

        if prev_s.get("symbol") != next_s.get("symbol"):
            continue

        events_between = load_execution_events_between(
            conn,
            symbol=prev_s["symbol"],
            ts_from=prev_s["ts"],
            ts_to=next_s["ts"],
        )

        verdict = detect_exchange_close(prev_s, next_s, events_between)
        if verdict is None:
            continue

        out.append({
            "symbol": prev_s["symbol"],
            "prev_snapshot_id": prev_s.get("id"),
            "prev_ts": prev_s.get("ts"),
            "prev_side": prev_s.get("side"),
            "prev_size": prev_s.get("size"),
            "prev_avg_price": prev_s.get("avg_price"),
            "prev_tp": prev_s.get("take_profit"),
            "prev_sl": prev_s.get("stop_loss"),
            "next_snapshot_id": next_s.get("id"),
            "next_ts": next_s.get("ts"),
            "next_side": next_s.get("side"),
            "next_size": next_s.get("size"),
            "status": verdict["status"],
            "reason": verdict["reason"],
            "events_between_count": len(events_between),
        })

    # Последние сверху
    return list(reversed(out))

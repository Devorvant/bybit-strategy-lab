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


def classify_execution(event: dict, snapshot: dict | None) -> dict:
    action = str(event.get("requested_action") or "").upper()
    event_type = str(event.get("event_type") or "")
    ok = bool(event.get("ok"))

    if not ok:
        return {
            "status": "error",
            "reason": event.get("error") or "execution failed",
        }

    if snapshot is None:
        return {
            "status": "unknown",
            "reason": "no snapshot found after execution",
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
            }
        return {
            "status": "mismatch",
            "reason": f"expected flat, got side={snap_side} size={snap_size}",
        }

    if expected == "Buy":
        if str(snap_side or "").lower() == "buy" and float(snap_size or 0) > 0:
            return {
                "status": "confirmed",
                "reason": "long position confirmed",
            }
        if float(snap_size or 0) == 0:
            return {
                "status": "no_effect",
                "reason": "long requested but position still flat",
            }
        return {
            "status": "mismatch",
            "reason": f"expected long, got side={snap_side} size={snap_size}",
        }

    if expected == "Sell":
        if str(snap_side or "").lower() == "sell" and float(snap_size or 0) > 0:
            return {
                "status": "confirmed",
                "reason": "short position confirmed",
            }
        if float(snap_size or 0) == 0:
            return {
                "status": "no_effect",
                "reason": "short requested but position still flat",
            }
        return {
            "status": "mismatch",
            "reason": f"expected short, got side={snap_side} size={snap_size}",
        }

    return {
        "status": "unknown",
        "reason": f"unsupported action={action} event_type={event_type}",
    }


def find_next_snapshot(conn, symbol: str, event_ts, source: str | None = None) -> dict | None:
    sql = """
    SELECT
        id, ts, symbol, source, side, size, avg_price, mark_price,
        unrealised_pnl, leverage, liq_price, take_profit, stop_loss,
        has_position, wallet_equity, wallet_balance, available_balance, payload
    FROM exchange_snapshots
    WHERE symbol = %(symbol)s
      AND ts >= %(event_ts)s
      AND (%(source)s IS NULL OR source = %(source)s)
    ORDER BY ts ASC
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "symbol": symbol,
            "event_ts": event_ts,
            "source": source,
        })
        row = cur.fetchone()
        if row is None:
            return None

        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def load_recent_execution_events(conn, limit: int = 100) -> list[dict]:
    sql = """
    SELECT
        id, ts, run_id, symbol, tf, bar_ts,
        event_type, requested_action, side, qty, price,
        reduce_only, ok, error, response
    FROM execution_events
    ORDER BY id DESC
    LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"limit": limit})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def analyze_recent_events(conn, limit: int = 100) -> list[dict]:
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

        snap = find_next_snapshot(conn, e["symbol"], e["ts"], source=source)
        verdict = classify_execution(e, snap)

        out.append({
            "event_id": e.get("id"),
            "event_ts": e.get("ts"),
            "symbol": e.get("symbol"),
            "event_type": e.get("event_type"),
            "requested_action": e.get("requested_action"),
            "ok": e.get("ok"),
            "error": e.get("error"),
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
        })

    return out

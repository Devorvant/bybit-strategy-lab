from __future__ import annotations

import datetime
import html
import json
import hashlib
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from app.backtest.sma_backtest import backtest_sma_cross
from app.backtest.sma_backtest_tv_like import backtest_sma_cross_tv_like
from app.backtest.strategy2_backtest import backtest_sma_adx_filter
from app.backtest.strategy3_backtest import backtest_strategy3, atr
from app.backtest.strategy3_backtest_tv_like import backtest_strategy3_tv_like


# Common timeframe buttons (Bybit intervals)
TF_BUTTONS: List[Tuple[str, str]] = [
    ("1", "1m"),
    ("3", "3m"),
    ("5", "5m"),
    ("15", "15m"),
    ("30", "30m"),
    ("60", "1h"),
    ("120", "2h"),
    ("240", "4h"),
    ("360", "6h"),
    ("720", "12h"),
    ("D", "1D"),
    ("W", "1W"),
]


# --- Lightweight caches to speed up /chart and /api/chart_update ---
# Keyed by (strategy, bars_sig, params_hash, capital_rounded).
_PLOT_CACHE = {}
_PLOT_CACHE_TTL_SEC = 30.0
_TRADES_TABLE_MAX_ROWS = 200

def _bars_sig(bars: Sequence[Tuple[int, float, float, float, float, float]]) -> Tuple[int, int, int]:
    if not bars:
        return (0, 0, 0)
    try:
        return (len(bars), int(bars[0][0]), int(bars[-1][0]))
    except Exception:
        return (len(bars), 0, 0)

def _params_hash(strategy_params: Optional[dict]) -> str:
    if not strategy_params:
        return "none"
    try:
        s = json.dumps(strategy_params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except Exception:
        s = str(sorted(strategy_params.items()))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_get(key):
    rec = _PLOT_CACHE.get(key)
    if not rec:
        return None
    ts0, payload = rec
    if time.monotonic() - ts0 > _PLOT_CACHE_TTL_SEC:
        return None
    return payload

def _cache_set(key, payload):
    _PLOT_CACHE[key] = (time.monotonic(), payload)





# Human-readable explanations shown next to strategy parameters in the HTML.
PARAM_HELP = {
    "position_usd": "Размер позиции (USD)",
    "capital_usd": "Стартовый капитал для % метрик",
    "leverage_mult": "Мультипликатор (плечо). Позиция = база × плечо",
    "trade_from_current_capital": "Если Да — позиция от текущего капитала (компаунд), если Нет — от стартового",
    "fee_percent": "Комиссия биржи на сторону (%)",
    "spread_ticks": "Спред (bid-ask) в тиках (полный спред)",
    "slippage_ticks": "Проскальзывание в тиках на сторону",
    "tick_size": "Размер тика (шаг цены)",
    "funding_8h_percent": "Funding за 8 часов (%) (Perpetual)",
    "use_no_trade": "Фильтр: не торговать при слабом тренде",
    "adx_len": "Период ADX",
    "adx_smooth": "Сглаживание ADX",
    "adx_no_trade_below": "Порог ADX: ниже — NO TRADE",
    "st_atr_len": "ATR период для Supertrend",
    "st_factor": "Коэф Supertrend (множитель ATR)",
    "use_rev_cooldown": "Пауза после разворота",
    "rev_cooldown_hrs": "Длина паузы (часы)",
    "use_flip_limit": "Лимит частых переворотов",
    "max_flips_per_day": "Макс переворотов в сутки",
    "use_emergency_sl": "Аварийный стоп-лосс",
    "atr_len": "ATR период для авар. SL",
    "atr_mult": "Множитель ATR для SL",
    "close_at_end": "Закрыть позицию в конце окна",
    "confirm_on_close": "Сигнал на закрытии свечи; исполнение на следующем open (реалистичнее)",
}

# Params that participate in strategy3 optimization and can be edited on the chart page.
EDITABLE_STRAT3_KEYS = [
"leverage_mult",
    "trade_from_current_capital",
    "slippage_ticks",
    "tick_size",
    "fee_percent",
    "spread_ticks",
    "funding_8h_percent",
    # "Other" group (make editable in chart UI)
    "adx_len",
    "adx_smooth",
    "st_atr_len",
    "use_rev_cooldown",
    "use_emergency_sl",
    "atr_len",
    "close_at_end",
    "use_no_trade",
    "adx_no_trade_below",
    "st_factor",
    "rev_cooldown_hrs",
    "use_flip_limit",
    "max_flips_per_day",
    "atr_mult",
    "confirm_on_close",
]

# Extra UI-only editable key (does not affect backtest, only % metrics)
EDITABLE_UI_STRAT3_KEYS = set(EDITABLE_STRAT3_KEYS + ["capital_usd"])

def _build_fig_and_bt(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    strategy: str,
    strategy_params: Optional[dict] = None,
    capital_usd: float = 10000.0,
) -> tuple[go.Figure | None, object | None, str | None]:
    """Build figure + backtest result.

    Returns (fig, bt, error_html).
    """
    if not bars:
        return None, None, '<div style="padding:12px;font-size:14px;">No bars yet for this symbol/tf.</div>'

    df = pd.DataFrame(bars, columns=["ts", "o", "h", "l", "c", "v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")

    # Indicators (keep in sync with strategy defaults)
    df["sma20"] = df["c"].rolling(20).mean()
    df["sma50"] = df["c"].rolling(50).mean()

    cross_up = (df["sma20"].shift(1) <= df["sma50"].shift(1)) & (df["sma20"] > df["sma50"])
    cross_dn = (df["sma20"].shift(1) >= df["sma50"].shift(1)) & (df["sma20"] < df["sma50"])

    # Backtest to draw trades + equity
    if strategy == "my_strategy3.py":
        params = {
            # Defaults (keep in sync with UI initial state)
            "position_usd": 1000.0,
            # Leverage sizing (UI mode) — used only if leverage_mult is not None
            "capital_usd": float(capital_usd),
            "leverage_mult": None,
            "trade_from_current_capital": False,
            "slippage_ticks": 2,
            "tick_size": 0.0001,
            "fee_percent": 0.06,
            "spread_ticks": 1.0,
            "funding_8h_percent": 0.01,
            "adx_len": 14,
            "adx_smooth": 14,
            "adx_no_trade_below": 14.0,
            "st_atr_len": 14,
            "st_factor": 4.0,
            "use_no_trade": True,
            "use_rev_cooldown": True,
            "rev_cooldown_hrs": 8,
            "use_flip_limit": False,
            "max_flips_per_day": 6,
            "use_emergency_sl": True,
            "atr_len": 14,
            "atr_mult": 3.0,
            "close_at_end": False,
            "confirm_on_close": False,
        }
        if strategy_params:
            # Apply only known backtest params (ignore UI-only keys like capital_usd)
            for kk, vv in dict(strategy_params).items():
                if kk in params:
                    params[kk] = vv

        bt = backtest_strategy3(bars, **params)
        df["st_line"] = pd.Series(bt.st_line)
        df["st_dir"] = pd.Series(bt.st_dir)
        df["adx"] = pd.Series(bt.adx)
        df["no_trade"] = pd.Series(bt.no_trade)

    elif strategy == "my_strategy3_tv_like.py":
        bt = backtest_strategy3_tv_like(
            bars,
            initial_capital=10000.0,
            percent_of_equity=50.0,
            commission_percent=0.10,
            slippage_ticks=2,
            tick_size=0.0001,  # ✅ syminfo.mintick
            use_no_trade=True,
            adx_len=14,
            adx_smooth=14,
            adx_no_trade_below=14.0,
            st_atr_len=14,
            st_factor=4.0,
            use_rev_cooldown=True,
            rev_cooldown_hrs=8,
            use_flip_limit=False,
            max_flips_per_day=6,
            use_emergency_sl=True,
            atr_len=14,
            atr_mult=3.0,
            close_at_end=False,
        )
        df["st_line"] = pd.Series(bt.st_line)
        df["st_dir"] = pd.Series(bt.st_dir)
        df["adx"] = pd.Series(bt.adx)
        df["no_trade"] = pd.Series(bt.no_trade)

    elif strategy == "my_strategy2.py":
        bt = backtest_sma_adx_filter(
            bars,
            position_usd=1000.0,
            fast_n=20,
            slow_n=50,
            adx_n=14,
            adx_enter=20.0,
            adx_exit=15.0,
            close_at_end=False,
        )

    elif strategy == "my_strategy_tv_like.py":
        bt = backtest_sma_cross_tv_like(
            bars,
            position_usd=1000.0,
            fast_n=20,
            slow_n=50,
            fee_rate=0.0,
            slippage_bps=0.0,
            close_at_end=False,
        )

    else:
        bt = backtest_sma_cross(bars, position_usd=1000.0, fast_n=20, slow_n=50, close_at_end=False)

    eq_df = pd.DataFrame({"ts": bt.equity_ts, "equity": bt.equity})
    if not eq_df.empty:
        eq_df["dt"] = pd.to_datetime(eq_df["ts"], unit="ms")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )

    # Row 1: price candles + overlays
    fig.add_trace(
        go.Candlestick(
            name="OHLC",
            x=df["dt"],
            open=df["o"],
            high=df["h"],
            low=df["l"],
            close=df["c"],
        ),
        row=1,
        col=1,
    )

    if strategy not in ("my_strategy3.py", "my_strategy3_tv_like.py"):
        # SMA overlays
        fig.add_trace(
            go.Scatter(
                name="SMA20",
                x=df["dt"],
                y=df["sma20"],
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="SMA50",
                x=df["dt"],
                y=df["sma50"],
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

        # Cross markers (where the signal would flip)
        fig.add_trace(
            go.Scatter(
                name="Cross Up",
                x=df.loc[cross_up, "dt"],
                y=df.loc[cross_up, "c"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Cross Down",
                x=df.loc[cross_dn, "dt"],
                y=df.loc[cross_dn, "c"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10),
            ),
            row=1,
            col=1,
        )
    else:
        # Strategy3 overlays: Supertrend (up/down segments) + optional NO-TRADE markers
        up = df["st_dir"] == 1
        dn = df["st_dir"] == -1
        fig.add_trace(
            go.Scatter(
                name="Supertrend Up",
                x=df.loc[up, "dt"],
                y=df.loc[up, "st_line"],
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Supertrend Down",
                x=df.loc[dn, "dt"],
                y=df.loc[dn, "st_line"],
                mode="lines",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

        if "no_trade" in df.columns:
            nt = df["no_trade"] == True  # noqa: E712
            if nt.any():
                fig.add_trace(
                    go.Scatter(
                        name="NO TRADE",
                        x=df.loc[nt, "dt"],
                        y=df.loc[nt, "h"],
                        mode="markers",
                        marker=dict(symbol="x", size=8),
                    ),
                    row=1,
                    col=1,
                )

    # Trades from backtest (entry/exit markers)
    if bt.trades:
        entry_x, entry_y, entry_text, entry_sym = [], [], [], []
        exit_x, exit_y, exit_text = [], [], []
        ts_to_px = {int(t): float(px) for t, px in zip(df["ts"].tolist(), df["c"].tolist())}

        for tr in bt.trades:
            entry_x.append(pd.to_datetime(tr.entry_ts, unit="ms"))
            entry_y.append(ts_to_px.get(tr.entry_ts, tr.entry_price))
            entry_sym.append("triangle-up" if tr.side == "LONG" else "triangle-down")
            entry_text.append(f"{tr.side} entry<br>px={tr.entry_price:.6g}<br>ts={tr.entry_ts}")

            exit_x.append(pd.to_datetime(tr.exit_ts, unit="ms"))
            exit_y.append(ts_to_px.get(tr.exit_ts, tr.exit_price))
            exit_text.append(f"{tr.side} exit<br>px={tr.exit_price:.6g}<br>pnl={tr.pnl:.2f} USD")

        fig.add_trace(
            go.Scatter(
                name="Trade Entry",
                x=entry_x,
                y=entry_y,
                mode="markers",
                marker=dict(size=12, symbol=entry_sym),
                text=entry_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="Trade Exit",
                x=exit_x,
                y=exit_y,
                mode="markers",
                marker=dict(size=10, symbol="x"),
                text=exit_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Row 2: equity curve
    if not eq_df.empty:
        fig.add_trace(
            go.Scatter(
                name="Equity (USD)",
                x=eq_df["dt"],
                y=eq_df["equity"],
                mode="lines",
                line=dict(width=2),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Equity", row=2, col=1)

    # Styling
    fig.update_layout(
        height=760,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig, bt, None


def build_plotly_payload(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    strategy: str,
    strategy_params: Optional[dict] = None,
    capital_usd: float = 10000.0,
) -> dict:
    """Return Plotly payload to update an existing chart via Plotly.react()."""
    fig, bt, err_html = _build_fig_and_bt(
        bars,
        strategy=strategy,
        strategy_params=strategy_params,
        capital_usd=capital_usd,
    )
    if err_html is not None or fig is None:
        return {"ok": False, "error_html": err_html or "<div>error</div>"}

    # IMPORTANT: fig.to_plotly_json() may contain numpy/pandas objects (datetime64,
    # ndarray, etc.) that FastAPI can't JSON-encode. Convert via Plotly's JSON
    # serializer to guarantee plain JSON types.
    pj = json.loads(pio.to_json(fig, validate=False))
    return {
        "ok": True,
        "data": pj.get("data", []),
        "layout": pj.get("layout", {}),
        # Keep config minimal; plotly.js already loaded on page.
        "config": {"responsive": True, "displayModeBar": False},
        # Also return the trades/metrics table so Playground can update it live.
        "trades_html": _build_trades_table_html(bt, bars=bars, params=strategy_params),
    }


def _build_plot_html(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    strategy: str,
    strategy_params: Optional[dict] = None,
    capital_usd: float = 10000.0,
    div_id: str = "chartPlot",
) -> str:
    """Return Plotly HTML fragment (no outer <html> tag)."""
    fig, bt, err_html = _build_fig_and_bt(
        bars,
        strategy=strategy,
        strategy_params=strategy_params,
        capital_usd=capital_usd,
    )
    if err_html is not None or fig is None:
        return err_html or '<div style="padding:12px;font-size:14px;">No bars yet for this symbol/tf.</div>'
    # Stable div_id is required for JS updates via Plotly.react().
    cfg = {"responsive": True, "displayModeBar": False}
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config=cfg,
    )


def _build_trades_table_html(bt, bars=None, params=None) -> str:
    """TradingView-like trades table, including OPEN position if present."""
    trades = list(getattr(bt, "trades", []) or [])

    def dt_str(ts: int) -> str:
        try:
            ts_int = int(ts)
            secs = (ts_int / 1000.0) if ts_int > 10_000_000_000 else float(ts_int)
            return datetime.datetime.utcfromtimestamp(secs).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    total_trades = len(trades)
    wins = sum(1 for t in trades if float(getattr(t, "pnl", 0.0)) > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0

    # Cumulative PnL for each closed trade:
    # Prefer bt.Trade.cum_pnl (it already includes entry-fees, funding, etc. that may not be present in trade.pnl).
    cum_pnl: List[float] = []
    cum = 0.0
    for tr in trades:
        if hasattr(tr, "cum_pnl"):
            try:
                cum = float(getattr(tr, "cum_pnl"))
            except Exception:
                cum += float(getattr(tr, "pnl", 0.0))
        else:
            cum += float(getattr(tr, "pnl", 0.0))
        cum_pnl.append(cum)

    open_pos = getattr(bt, "open_position", None)
    if isinstance(open_pos, dict):
        open_pnl = float(open_pos.get("unrealized_pnl", 0.0))
    else:
        open_pnl = float(getattr(open_pos, "unrealized_pnl", 0.0)) if open_pos else 0.0

    # Net PnL from the equity curve (includes entry fees, funding, etc.).
    eq_series = list(getattr(bt, "equity", []) or [])
    if eq_series:
        # Some backtests track equity as pure PnL from 0, others as account value.
        if abs(float(eq_series[0])) < 1e-9:
            net_pnl = float(eq_series[-1])
        else:
            net_pnl = float(eq_series[-1]) - float(eq_series[0])
    else:
        net_pnl = (cum_pnl[-1] if cum_pnl else 0.0) + open_pnl

    realized_pnl = net_pnl - open_pnl
    avg_pnl = (realized_pnl / total_trades) if total_trades else 0.0

    # --- Extra summary metrics ---
    # Profit % for the whole period and max drawdown % are computed from the equity curve.
    eq = list(getattr(bt, "equity", []) or [])
    if eq:
        # Some backtests track equity as pure PnL starting from 0.
        # Add a virtual base so that % metrics are meaningful.
        base = float(eq[0])
        if abs(base) < 1e-9:
            base = float((params or {}).get('capital_usd', 10000.0))  # starting capital for % metrics
            eq_adj = [base + float(x) for x in eq]
        else:
            eq_adj = [float(x) for x in eq]

        ret_pct = ((eq_adj[-1] - eq_adj[0]) / eq_adj[0] * 100.0) if abs(eq_adj[0]) > 1e-9 else 0.0

        peak = eq_adj[0]
        max_dd = 0.0  # negative
        for v in eq_adj:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (v - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        max_dd_pct = abs(max_dd) * 100.0
    else:
        ret_pct = 0.0
        max_dd_pct = 0.0

    # Волатильность / "дребезг" equity.
    # - Волатильность: std(доходностей equity) за шаг, в %
    # - Дребезг: средняя абсолютная доходность за шаг (амплитуда колебаний), в %
    vol_pct = 0.0
    jitter_pct = 0.0
    rets: List[float] = []
    try:
        if isinstance(eq_adj, list) and len(eq_adj) >= 3:
            for i in range(1, len(eq_adj)):
                prev = float(eq_adj[i - 1])
                cur = float(eq_adj[i])
                if abs(prev) > 1e-9:
                    rets.append((cur - prev) / prev)
            if len(rets) >= 2:
                mean = sum(rets) / len(rets)
                var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
                vol_pct = (var ** 0.5) * 100.0
                jitter_pct = (sum(abs(r) for r in rets) / len(rets)) * 100.0
    except Exception:
        vol_pct = 0.0
        jitter_pct = 0.0

    # Ulcer Index (%): RMS просадок от локальных максимумов.
    ulcer_pct = 0.0
    try:
        if isinstance(eq_adj, list) and len(eq_adj) >= 2:
            peak = float(eq_adj[0])
            dd_sq = []
            for v in eq_adj:
                fv = float(v)
                if fv > peak:
                    peak = fv
                if peak > 1e-9:
                    dd = max(0.0, (peak - fv) / peak) * 100.0
                    dd_sq.append(dd * dd)
            if dd_sq:
                ulcer_pct = (sum(dd_sq) / len(dd_sq)) ** 0.5
    except Exception:
        ulcer_pct = 0.0


    def row_html(
        trade_no: int,
        trade_type: str,
        side: str,
        reason: str,
        entry_dt: str,
        exit_dt: str,
        entry_px: str,
        exit_px: str,
        pnl: float,
        cum_val: float,
    ) -> str:
        pnl_cls = "pnl-pos" if pnl >= 0 else "pnl-neg"
        cum_cls = "pnl-pos" if cum_val >= 0 else "pnl-neg"
        reason_norm = (reason or "").upper()
        reason_cls = "reason-stop" if reason_norm.startswith("STOP") else "reason"
        return f"""
        <tr>
          <td class="num">{trade_no}</td>
          <td class="type">{html.escape(trade_type)}</td>
          <td class="side">{html.escape(side)}</td>
          <td class="{reason_cls}">{html.escape(reason_norm)}</td>
          <td class="dt">{html.escape(entry_dt)}</td>
          <td class="dt">{html.escape(exit_dt)}</td>
          <td class="px">{html.escape(entry_px)}</td>
          <td class="px">{html.escape(exit_px)}</td>
          <td class="pnl {pnl_cls}">{pnl:,.2f}</td>
          <td class="pnl {cum_cls}">{cum_val:,.2f}</td>
        </tr>
        """

    rows = []

    if open_pos is not None:
        trade_no = total_trades + 1
        if isinstance(open_pos, dict):
            side = str(open_pos.get("side", "LONG"))
            entry_dt = dt_str(open_pos.get("entry_ts", 0))
            exit_dt = dt_str(open_pos.get("current_ts", open_pos.get("entry_ts", 0))) + " (OPEN)"
            entry_px = f"{float(open_pos.get('entry_price', 0.0)):.6g}"
            exit_px = f"{float(open_pos.get('current_price', 0.0)):.6g}"
        else:
            side = str(getattr(open_pos, "side", "LONG"))
            entry_dt = dt_str(getattr(open_pos, "entry_ts", 0))
            exit_dt = dt_str(getattr(open_pos, "current_ts", getattr(open_pos, "entry_ts", 0))) + " (OPEN)"
            entry_px = f"{float(getattr(open_pos, 'entry_price', 0.0)):.6g}"
            exit_px = f"{float(getattr(open_pos, 'current_price', 0.0)):.6g}"

        rows.append(
            row_html(
                trade_no,
                "ОТКРЫТА",
                side,
                "ОТКРЫТА",
                entry_dt,
                exit_dt,
                entry_px,
                exit_px,
                open_pnl,
                net_pnl,
            )
        )

    for trade_no, tr in reversed(list(enumerate(trades, start=1))):
        side = str(getattr(tr, "side", "LONG"))
        entry_dt = dt_str(getattr(tr, "entry_ts", 0))
        exit_dt = dt_str(getattr(tr, "exit_ts", 0))
        entry_px = f"{float(getattr(tr, 'entry_price', 0.0)):.6g}"
        exit_px = f"{float(getattr(tr, 'exit_price', 0.0)):.6g}"
        pnl = float(getattr(tr, "pnl", 0.0))
        cum_val = float(cum_pnl[trade_no - 1]) if cum_pnl else 0.0
        reason = str(getattr(tr, "exit_reason", "CROSS"))
        rows.append(
            row_html(
                trade_no,
                "ЗАКРЫТА",
                side,
                reason,
                entry_dt,
                exit_dt,
                entry_px,
                exit_px,
                pnl,
                cum_val,
            )
        )

    rows_html = "\n".join(rows)

    return f"""
    <div class="trades-wrap" id="tradesWrap">
      <div class="trades-metrics">
        <div><b>Всего сделок:</b> {total_trades}{' (+1 open)' if open_pos is not None else ''}</div>
        <div><b>Винрейт:</b> {win_rate:.1f}%</div>
        <div><b>Прибыль за период:</b> {ret_pct:.2f}%</div>
        <div><b>Макс. просадка:</b> {max_dd_pct:.2f}%</div>
        <div><b title="Ульцер индекс: квадратичное среднее просадок от локальных максимумов. Больше = глубже и дольше просадки.">Ульцер индекс:</b> {ulcer_pct:.2f}%</div>
        <div><b title="Волатильность equity: std(процентных изменений) за шаг. Больше = сильнее колебания вверх/вниз.">Волатильность:</b> {vol_pct:.2f}%</div>
        <div><b title="Дребезг equity: средняя абсолютная доходность за шаг (амплитуда колебаний). Больше = более пилообразная кривая.">Дребезг:</b> {jitter_pct:.2f}%</div>
        <div><b>Реализ. PnL:</b> {realized_pnl:,.2f} USD</div>
        <div><b>Нереализ. PnL:</b> {open_pnl:,.2f} USD</div>
        <div><b>Итоговый PnL:</b> {net_pnl:,.2f} USD</div>
        <div><b>Ср. PnL / сделка:</b> {avg_pnl:,.2f} USD</div>
      </div>

      <div class="trades-table">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Тип</th>
              <th>Сторона</th>
              <th>Причина</th>
              <th>Время входа</th>
              <th>Время выхода</th>
              <th>Цена входа</th>
              <th>Цена выхода</th>
              <th>PnL</th>
              <th>Накопл. PnL</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
    """


def make_chart_html(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    symbol: str = "APTUSDT",
    tf: str = "120",
    limit: int = 5000,
    strategy: str = "my_strategy.py",
    symbols: Optional[Iterable[str]] = None,
    tfs: Optional[Iterable[str]] = None,
    # Optional optimizer integration
    opt_strategy: Optional[str] = None,
    opt_results: Optional[Sequence[tuple]] = None,
    opt_id: Optional[int] = None,
    opt_last: int = 20,
    opt_params: Optional[dict] = None,
    # If 1, apply p_* query parameters as manual overrides (strategy3 only)
    use_overrides: int = 0,
    # UI-only: starting capital for % metrics when equity is tracked as PnL
    capital_usd: float = 10000.0,
) -> str:
    """Render a simple chart page with controls (symbol/tf/limit)."""

    tf_e = html.escape(tf)

    symbols_list = [s for s in (symbols or [])] or [symbol]
    if symbol not in symbols_list:
        symbols_list = [symbol] + [s for s in symbols_list if s != symbol]

    available_strategies = [
        "my_strategy.py",
        "my_strategy2.py",
        "my_strategy3.py",
        "my_strategy_tv_like.py",
        "my_strategy3_tv_like.py",
    ]
    if strategy not in available_strategies:
        strategy = "my_strategy.py"

    plot_html = _build_plot_html(
        bars,
        strategy=strategy,
        strategy_params=opt_params,
        capital_usd=float(capital_usd),
        div_id="chartPlot",
    )

    # Build a compact "current params" panel near the chart.
    # Stage 1: display-only (editable panel comes later).
    def _current_params_for_strategy() -> dict:
        if strategy == "my_strategy3.py":
            base = {
                "position_usd": 1000.0,
                "capital_usd": float(capital_usd),
                "leverage_mult": None,
                "trade_from_current_capital": False,
                # Execution frictions (chart-only; optimizer ignores)
                "fee_percent": 0.06,          # per side (%)
                "spread_ticks": 1.0,          # full bid-ask spread (ticks)
                "slippage_ticks": 2,          # extra slippage (ticks per side)
                "tick_size": 0.0001,          # price tick size
                "funding_8h_percent": 0.01,   # funding per 8h (%), perp only
                "use_no_trade": True,
                "adx_len": 14,
                "adx_smooth": 14,
                "adx_no_trade_below": 14.0,
                "st_atr_len": 14,
                "st_factor": 4.0,
                "use_rev_cooldown": True,
                "rev_cooldown_hrs": 8,
                "use_flip_limit": False,
                "max_flips_per_day": 6,
                "use_emergency_sl": True,
                "atr_len": 14,
                "atr_mult": 3.0,
                "close_at_end": False,
                "confirm_on_close": False,
            }
        elif strategy == "my_strategy3_tv_like.py":
            base = {
                "initial_capital": 10000.0,
                "percent_of_equity": 50.0,
                "commission_percent": 0.10,
                "slippage_ticks": 2,
                "tick_size": 0.0001,
                "use_no_trade": True,
                "adx_len": 14,
                "adx_smooth": 14,
                "adx_no_trade_below": 14.0,
                "st_atr_len": 14,
                "st_factor": 4.0,
                "use_rev_cooldown": True,
                "rev_cooldown_hrs": 8,
                "use_flip_limit": False,
                "max_flips_per_day": 6,
                "use_emergency_sl": True,
                "atr_len": 14,
                "atr_mult": 3.0,
                "close_at_end": False,
            }
        elif strategy == "my_strategy2.py":
            base = {
                "position_usd": 1000.0,
                "fast_n": 20,
                "slow_n": 50,
                "adx_n": 14,
                "adx_enter": 20.0,
                "adx_exit": 15.0,
                "close_at_end": False,
            }
        elif strategy == "my_strategy_tv_like.py":
            base = {
                "position_usd": 1000.0,
                "fast_n": 20,
                "slow_n": 50,
                "fee_rate": 0.0,
                "slippage_bps": 0.0,
                "close_at_end": False,
            }
        else:
            base = {
                "position_usd": 1000.0,
                "fast_n": 20,
                "slow_n": 50,
                "close_at_end": False,
            }

        # Only strategy3 currently accepts opt_params overrides in the chart.
        if opt_params and strategy == "my_strategy3.py":
            try:
                base.update({k: opt_params[k] for k in opt_params.keys()})
            except Exception:
                pass
        return base

    cur_params = _current_params_for_strategy()
    # For the chart UI we expose leverage-based sizing instead of raw position_usd.
    if strategy == "my_strategy3.py":
        try:
            cap0 = float(capital_usd) if capital_usd is not None else float(cur_params.get('capital_usd', 10000.0))
        except Exception:
            cap0 = float(cur_params.get('capital_usd', 10000.0) or 10000.0)
        if cur_params.get('leverage_mult', None) is None:
            try:
                pu = float(cur_params.get('position_usd', 1000.0))
                cur_params['leverage_mult'] = (pu / cap0) if cap0 > 0 else 0.0
            except Exception:
                cur_params['leverage_mult'] = 0.0
        if 'trade_from_current_capital' not in cur_params:
            cur_params['trade_from_current_capital'] = False
        cur_params.pop('position_usd', None)
    src_lbl = "defaults"
    if opt_id:
        src_lbl = f"optimized #{int(opt_id)}"
    elif opt_params:
        src_lbl = "overrides"

    # If user edited any coefficients, reflect that in the label.
    if int(use_overrides or 0) == 1 and strategy == "my_strategy3.py":
        if opt_id:
            src_lbl = f"{src_lbl} + overrides"
        else:
            src_lbl = "overrides"
    editable_keys = (set(EDITABLE_UI_STRAT3_KEYS) if strategy == "my_strategy3.py" else set())

    def _render_value_cell(k: str, v) -> str:
        k = str(k)
        if k not in editable_keys:
            return f"<div class='p-v'>{html.escape(str(v))}</div>"

        # capital_usd is UI-only (does not affect backtest params), so we don't prefix it with p_.
        if k == 'capital_usd':
            try:
                val = float(v)
            except Exception:
                val = 10000.0
            return (
                "<div class='p-v'>"
                f"<input id='capital_usd' class='p-in' form='chartForm' name='capital_usd' type='number' step='100' "
                f"value='{html.escape(str(val))}'/>"
                "</div>"
            )

        name = f"p_{k}"
        js_mark = "document.getElementById('use_overrides').value='1'; if (window.__enableOverrides) window.__enableOverrides();"

        if k in (
            "use_no_trade",
            "use_flip_limit",
            "trade_from_current_capital",
            "confirm_on_close",
            "use_rev_cooldown",
            "use_emergency_sl",
            "close_at_end",
        ):
            val = "true" if bool(v) else "false"
            sel_true = "selected" if val == "true" else ""
            sel_false = "selected" if val == "false" else ""
            return (
                "<div class='p-v'>"
                f"<select id='p_{html.escape(k)}' class='p-in' data-ov='1' form='chartForm' name='{html.escape(name)}' onchange=\"{js_mark}\">"
                f"<option value='true' {sel_true}>Да</option>"
                f"<option value='false' {sel_false}>Нет</option>"
                "</select></div>"
            )

        is_int = k in (
            "rev_cooldown_hrs",
            "max_flips_per_day",
            "slippage_ticks",
            "adx_len",
            "adx_smooth",
            "st_atr_len",
            "atr_len",
        )
        if k == "tick_size":
            step = "0.0001"
        elif k in ("fee_percent", "funding_8h_percent"):
            step = "0.01"
        elif k == "spread_ticks":
            step = "0.5"
        else:
            step = "1" if is_int else "0.1"
        try:
            value = str(int(v)) if is_int else str(float(v))
        except Exception:
            value = str(v)

        return (
            "<div class='p-v'>"
            f"<input id='p_{html.escape(k)}' class='p-in' data-ov='1' form='chartForm' name='{html.escape(name)}' type='number' step='{step}' "
            f"value='{html.escape(value)}' oninput=\"{js_mark}\"/>"
            "</div>"
        )

    
    def _iter_params_grouped():
        # For strategy3: show groups with colored headers.
        if strategy == "my_strategy3.py":
            ordered_keys = [str(k) for k in cur_params.keys()]
            used = set()

            def take(keys):
                out = []
                for k in keys:
                    if k in cur_params and k not in used:
                        out.append(k)
                        used.add(k)
                return out

            groups = [
                ("Исходные данные", "g-base", take(["capital_usd", "leverage_mult", "trade_from_current_capital"])),
                ("Комиссии", "g-costs", take(["fee_percent", "spread_ticks", "slippage_ticks", "tick_size", "funding_8h_percent"])),
                ("Коэф. модели", "g-model", take([
                    "use_no_trade",
                    "adx_no_trade_below",
                    "st_factor",
                    "rev_cooldown_hrs",
                    "use_flip_limit",
                    "max_flips_per_day",
                    "atr_mult",
                    "confirm_on_close",
                ])),
            ]
            rest = [k for k in ordered_keys if k not in used]
            if rest:
                groups.append(("Прочее", "g-other", rest))

            for title, cls, keys in groups:
                if not keys:
                    continue
                yield f"<div class='p-gh {cls}'>{html.escape(title)}</div>"
                for k in keys:
                    v = cur_params.get(k)
                    yield (
                        f"<div class='p-k'>{html.escape(str(k))}</div>"
                        f"{_render_value_cell(str(k), v)}"
                        f"<div class='p-d'>{html.escape(PARAM_HELP.get(str(k), ''))}</div>"
                    )
        else:
            for k, v in cur_params.items():
                yield (
                    f"<div class='p-k'>{html.escape(str(k))}</div>"
                    f"{_render_value_cell(str(k), v)}"
                    f"<div class='p-d'>{html.escape(PARAM_HELP.get(str(k), ''))}</div>"
                )

    params_items = "".join(_iter_params_grouped())

    params_html = f"""
    <div class=\"params-card\">
      <div class=\"params-head\"><b>Params</b><span class=\"muted\">({html.escape(src_lbl)})</span></div>
      <div class=\"params-grid\">{params_items}</div>
    </div>
    """

    # Build trades table
    try:
        params_for_metrics = None
        if strategy == "my_strategy2.py":
            bt = backtest_sma_adx_filter(bars, close_at_end=False)
        elif strategy == "my_strategy3.py":
            # Keep trades table in sync with the chart parameters.
            params = {
                "position_usd": 1000.0,
                "capital_usd": float(capital_usd),
                "leverage_mult": None,
                "trade_from_current_capital": False,
                # Execution frictions (chart-only; optimizer ignores)
                "fee_percent": 0.06,
                "spread_ticks": 1.0,
                "slippage_ticks": 2,
                "tick_size": 0.0001,
                "funding_8h_percent": 0.01,
                "adx_len": 14,
                "adx_smooth": 14,
                "adx_no_trade_below": 14.0,
                "st_atr_len": 14,
                "st_factor": 4.0,
                "use_no_trade": True,
                "use_rev_cooldown": True,
                "rev_cooldown_hrs": 8,
                "use_flip_limit": False,
                "max_flips_per_day": 6,
                "use_emergency_sl": True,
                "atr_len": 14,
                "atr_mult": 3.0,
                "close_at_end": False,
            }
            # opt_params is injected via make_chart_html signature below
            if opt_params:
                params.update(opt_params)
            # Derive leverage_mult from position_usd if not explicitly set
            try:
                if params.get("leverage_mult", None) is None:
                    pu = float(params.get("position_usd", 1000.0))
                    cap0 = float(params.get("capital_usd", float(capital_usd) if capital_usd is not None else 10000.0))
                    params["leverage_mult"] = (pu / cap0) if cap0 > 0 else 0.0
            except Exception:
                params["leverage_mult"] = 0.0
            params_for_metrics = params
            bt = backtest_strategy3(bars, **params)
        elif strategy == "my_strategy_tv_like.py":
            bt = backtest_sma_cross_tv_like(bars, fee_rate=0.0, slippage_bps=0.0, close_at_end=False)
        elif strategy == "my_strategy3_tv_like.py":
            bt = backtest_strategy3_tv_like(
                bars,
                initial_capital=10000.0,
                percent_of_equity=50.0,
                commission_percent=0.10,
                slippage_ticks=2,
                tick_size=0.0001,  # ✅ syminfo.mintick
                use_no_trade=True,
                adx_len=14,
                adx_smooth=14,
                adx_no_trade_below=14.0,
                st_atr_len=14,
                st_factor=4.0,
                use_rev_cooldown=True,
                rev_cooldown_hrs=8,
                use_flip_limit=False,
                max_flips_per_day=6,
                use_emergency_sl=True,
                atr_len=14,
                atr_mult=3.0,
                close_at_end=False,
            )
        else:
            bt = backtest_sma_cross(bars, close_at_end=False)
        trades_table_html = _build_trades_table_html(bt, bars=bars, params=(params_for_metrics if strategy == "my_strategy3.py" else None))
    except Exception as e:
        trades_table_html = f"<pre style='padding:10px;color:#f88'>Failed to build trades table: {html.escape(str(e))}</pre>"

    symbol_options = "\n".join(
        f"<option value='{html.escape(s)}' {'selected' if s == symbol else ''}>{html.escape(s)}</option>"
        for s in symbols_list
    )

    tf_buttons = []
    for tf_val, tf_lbl in TF_BUTTONS:
        active = "active" if tf_val == tf else ""
        tf_buttons.append(f"<button class='tf-btn {active}' onclick=\"setTf('{tf_val}')\">{tf_lbl}</button>")
    tf_buttons_html = "\n".join(tf_buttons)

    strategy_options = "\n".join(
        f"<option value='{html.escape(s)}' {'selected' if s == strategy else ''}>{html.escape(s)}</option>"
        for s in available_strategies
    )

    # Optimizer dropdown (per-strategy, optional)
    opt_controls_html = ""
    if opt_strategy and (opt_results is not None) and strategy == "my_strategy3.py":
        # opt_results rows: (id, created_at, best_score, best_metrics)
        def _fmt_created(x) -> str:
            try:
                # datetime
                if hasattr(x, "strftime"):
                    return x.strftime("%Y-%m-%d %H:%M:%S")
                return str(x)
            except Exception:
                return str(x)

        opt_opts = [
            f"<option value='' {'selected' if not opt_id else ''}>Current (defaults)</option>"
        ]
        import json as _json

        def _to_dict(x):
            if x is None:
                return {}
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    return _json.loads(x)
                except Exception:
                    return {}
            return {}

        for rid, created_at, best_score, best_metrics in (opt_results or []):
            m = _to_dict(best_metrics)
            ret = m.get("ret")
            dd = m.get("dd")
            trades = m.get("trades")

            parts = [f"#{int(rid)} {html.escape(_fmt_created(created_at))}"]
            if best_score is not None:
                parts.append(f"score={float(best_score):.2f}")
            if isinstance(ret, (int, float)):
                parts.append(f"ret={ret*100:+.1f}%")
            if isinstance(dd, (int, float)):
                parts.append(f"dd={dd*100:.1f}%")
            if isinstance(trades, (int, float)):
                parts.append(f"trades={int(trades)}")
            label = " ".join(parts)
            sel = "selected" if (opt_id is not None and int(opt_id) == int(rid)) else ""
            opt_opts.append(f"<option value='{int(rid)}' {sel}>{label}</option>")

        # Put the tail controls (Last/Limit/Apply) onto the next line so they
        # don't get pushed far to the right on wide screens.
        opt_controls_html = f"""
        <label>Optimized</label>
        <select name="opt_id" title="Load optimized parameters">{''.join(opt_opts)}</select>
        <span class="flex-break"></span>
        <label>Last</label>
        <input name="opt_last" value="{int(opt_last)}" style="width:70px" title="How many recent results to show"/>
        """
    # --- Parameter playground (strategy3 only): select + one slider that updates Plotly via fetch ---
    slider_html = ""
    if strategy == "my_strategy3.py":
        MAIN_PLAY_KEYS = ["st_factor", "atr_mult", "adx_no_trade_below", "rev_cooldown_hrs", "max_flips_per_day"]
        OTHER_PLAY_KEYS = ["adx_len", "adx_smooth", "st_atr_len", "atr_len"]
        PLAY_SPECS = {
            # main
            "st_factor": {"min": 1.0, "max": 8.0, "step": 0.05, "default": 4.0, "kind": "float", "digits": 4},
            "atr_mult": {"min": 1.0, "max": 10.0, "step": 0.05, "default": 3.0, "kind": "float", "digits": 3},
            "adx_no_trade_below": {"min": 5.0, "max": 50.0, "step": 0.5, "default": 14.0, "kind": "float", "digits": 2},
            "rev_cooldown_hrs": {"min": 0, "max": 48, "step": 1, "default": 8, "kind": "int", "digits": 0},
            "max_flips_per_day": {"min": 1, "max": 30, "step": 1, "default": 6, "kind": "int", "digits": 0},
            # прочее (manual tuning)
            "adx_len": {"min": 5, "max": 60, "step": 1, "default": 14, "kind": "int", "digits": 0},
            "adx_smooth": {"min": 5, "max": 60, "step": 1, "default": 14, "kind": "int", "digits": 0},
            "st_atr_len": {"min": 2, "max": 60, "step": 1, "default": 10, "kind": "int", "digits": 0},
            "atr_len": {"min": 2, "max": 60, "step": 1, "default": 14, "kind": "int", "digits": 0},
        }
        play_param = "st_factor"
        try:
            play_param = str(cur_params.get('_play_param', play_param))
        except Exception:
            play_param = 'st_factor'
        if play_param not in PLAY_SPECS:
            play_param = 'st_factor'
        spec = PLAY_SPECS[play_param]
        try:
            cur_v = cur_params.get(play_param, spec.get('default'))
            cur_v = float(cur_v) if spec.get('kind') != 'int' else int(float(cur_v))
        except Exception:
            cur_v = spec.get('default')
        # Build select options (with a visual separator before "прочее")
        opt_html = []
        for k in MAIN_PLAY_KEYS:
            sel = 'selected' if k == play_param else ''
            opt_html.append(f"<option value='{k}' {sel}>{k}</option>")
        opt_html.append("<option disabled>──────── Прочее ────────</option>")
        for k in OTHER_PLAY_KEYS:
            sel = 'selected' if k == play_param else ''
            opt_html.append(f"<option value='{k}' {sel}>{k}</option>")
        # Put raw JSON into <script type="application/json"> so JS can JSON.parse it.
        # Avoid closing the script tag accidentally (</script>) by escaping "/" after "<".
        specs_json = json.dumps(PLAY_SPECS).replace("</", "<\/")
        digits = int(spec.get('digits', 4))
        cur_v_s = f"{cur_v:.{digits}f}" if spec.get('kind') != 'int' else str(int(cur_v))
        slider_html = f"""
        <div class="params-card" id="playCard">
          <div class="params-head">
            <b>Playground</b>
            <span class="muted">выбери параметр и крути ползунок — обновляется только график (fetch + Plotly.react)</span>
          </div>
          <script id="play_specs_json" type="application/json">{specs_json}</script>
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:8px;">
            <div style="min-width:160px;display:flex;gap:8px;align-items:center;">
              <select id="play_param" style="min-width:160px;">{''.join(opt_html)}</select>
            </div>
            <input id="play_slider" type="range" min="{spec['min']}" max="{spec['max']}" step="{spec['step']}" value="{cur_v}" style="flex:1 1 260px;"/>
            <div style="min-width:160px;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
              <span id="play_val">{cur_v_s}</span>
              <span id="play_status" class="muted" style="margin-left:8px;"></span>
            </div>
            <button id="play_save" type="button" class="apply" style="background:#0e1830;border:1px solid #2b6dff;">Save</button>
          </div>
          <div class="muted" style="margin-top:6px;">Обновление происходит при отпускании ползунка. Таблица сделок ниже тоже пересчитывается (график + таблица).</div>
          <div class="muted" id="play_save_msg" style="margin-top:6px;"></div>
        </div>
        """


    manual_trade_html = """
    <div class="manual-trade-box" style="margin:12px 0;padding:12px;border:1px solid #2a3b5f;border-radius:10px;background:#0b1220;">
      <div style="font-weight:700;margin-bottom:8px;color:#dbeafe;">Manual Trade (Bybit)</div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <input id="mt_token" type="text" placeholder="Trade token (optional)"
               style="min-width:220px;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;" />
        <input id="mt_symbol" type="text" placeholder="Symbol (e.g. BTCUSDC)" style="min-width:140px;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;" />
    <input id="mt_tf" type="text" value="2h" placeholder="tf"
               style="width:70px;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;" />
        <input id="mt_note" type="text" value="manual from terminal" placeholder="note"
               style="min-width:220px;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;" />
        <button id="btnLong" type="button"
                style="padding:6px 12px;border-radius:8px;border:1px solid #166534;background:#16a34a;color:white;cursor:pointer;">LONG</button>
        <button id="btnShort" type="button"
                style="padding:6px 12px;border-radius:8px;border:1px solid #991b1b;background:#dc2626;color:white;cursor:pointer;">SHORT</button>
        <button id="btnClose" type="button"
                style="padding:6px 12px;border-radius:8px;border:1px solid #9a3412;background:#f97316;color:white;cursor:pointer;">CLOSE</button>
      </div>
      <pre id="mt_result" style="margin-top:8px;white-space:pre-wrap;font-size:12px;color:#cbd5e1;"></pre>
    </div>
    """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Chart</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0b1220; color: #e6e6e6; }}
    .topbar {{ padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; border-bottom: 1px solid #1b2940; position: sticky; top: 0; background: #0b1220; z-index: 10; }}
    .flex-break {{ flex-basis: 100%; height: 0; }}
    .topbar label {{ font-size: 13px; opacity: 0.85; }}
    select, input {{ background: #0e1830; color: #e6e6e6; border: 1px solid #1b2940; border-radius: 8px; padding: 7px 10px; outline: none; }}
    .tf-row {{ padding: 10px 14px 0 14px; display: flex; gap: 6px; flex-wrap: wrap; }}
    .tf-btn {{ background: #0e1830; border: 1px solid #1b2940; color: #e6e6e6; padding: 6px 10px; border-radius: 999px; cursor: pointer; font-size: 12px; }}
    .tf-btn.active {{ background: #1b5cff33; border-color: #2b6dff; }}
    .apply {{ background: #2b6dff; border: 0; color: #fff; padding: 8px 12px; border-radius: 10px; cursor: pointer; font-weight: 600; }}
    .wrap {{ padding: 10px 14px 20px 14px; }}
    .trades-wrap {{ margin-top: 10px; }}
    .trades-metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px 18px; font-size: 13px; margin: 8px 0 12px 0; }}
    .trades-table table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .trades-table thead th {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid #1b2940; opacity: 0.9; position: sticky; top: 58px; background: #0b1220; z-index: 5; }}
    .trades-table tbody td {{ padding: 7px 6px; border-bottom: 1px solid #15243a; white-space: nowrap; }}
    .num {{ opacity: 0.85; }}
    .pnl-pos {{ color: #6ee7b7; }}
    .pnl-neg {{ color: #fb7185; }}
    .reason-stop {{ color: #fbbf24; }}
    .reason {{ opacity: 0.9; }}

    .navlink {{ display:inline-block; text-decoration:none; background:#0e1830; border:1px solid #1b2940; color:#e6e6e6; padding: 7px 10px; border-radius: 999px; font-weight: 700; }}
    .navlink.active {{ background:#1b5cff33; border-color:#2b6dff; }}

    .params-card {{ margin: 10px 0 12px 0; padding: 10px 12px; background: #0e1830; border: 1px solid #1b2940; border-radius: 12px; }}
    .params-head {{ display: flex; gap: 8px; align-items: baseline; }}
    .muted {{ opacity: 0.7; font-size: 12px; }}
    .params-grid {{ margin-top: 8px; display: grid; grid-template-columns: 220px 170px minmax(0, 1fr); gap: 6px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }}

    .p-gh {{ grid-column: 1 / -1; padding: 6px 10px; border-radius: 10px; font-weight: 700; font-size: 12px; letter-spacing: 0.3px; margin-top: 8px; }}
    .p-gh.g-base {{ background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.35); }}
    .p-gh.g-costs {{ background: rgba(249, 115, 22, 0.15); border: 1px solid rgba(249, 115, 22, 0.35); }}
    .p-gh.g-model {{ background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.35); }}
    .p-gh.g-other {{ background: rgba(148, 163, 184, 0.10); border: 1px solid rgba(148, 163, 184, 0.25); }}
    .p-d {{ opacity: 0.75; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    .p-k {{ opacity: 0.85; }}
    .p-v {{ opacity: 0.95; overflow-wrap: anywhere; }}
    .p-in {{ width: 100%; background: #0b1220; color: #e6e6e6; border: 1px solid #1b2940; border-radius: 8px; padding: 5px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }}
    .p-in:focus {{ outline: none; border-color: #2b6dff; box-shadow: 0 0 0 2px #2b6dff33; }}

    @media (max-width: 760px) {{
      .topbar label {{ font-size: 12px; }}
      select, input {{ flex: 1 1 140px; }}
      .apply {{ width: 100%; }}
      .params-grid {{ grid-template-columns: 1fr; }}
      .p-d {{ grid-column: 1 / -1; }}
      .p-v {{ margin-bottom: 6px; }}
    }}
  </style>
  <script>
    function __setOverridesEnabled(enabled) {{
      const els = document.querySelectorAll("[data-ov='1']");
      els.forEach(el => {{
        if (enabled) {{
          if (el.dataset.origName && !el.getAttribute("name")) {{
            el.setAttribute("name", el.dataset.origName);
          }}
        }} else {{
          if (el.getAttribute("name")) {{
            el.dataset.origName = el.getAttribute("name");
            el.removeAttribute("name");
          }}
        }}
      }});
    }}

    window.__enableOverrides = function() {{
      __setOverridesEnabled(true);
    }};

    function clearOverrides() {{
      const uo = document.getElementById("use_overrides");
      if (uo) uo.value = "0";
      __setOverridesEnabled(false);
    }}

    function setTf(tf) {{
      clearOverrides();
      document.getElementById("tf").value = tf;
      document.getElementById("chartForm").submit();
    }}

    document.addEventListener("DOMContentLoaded", () => {{
      // If overrides are not active, don't submit p_* fields (keeps URLs clean and lets opt/defaults reload).
      const uo = document.getElementById("use_overrides");
      if (!uo || uo.value !== "1") {{
        __setOverridesEnabled(false);
      }}

      // Changing main controls should reset overrides so base params load for the new mode.
      const selectors = [
        "select[name='symbol']",
        "select[name='strategy']",
        "input[name='tf']",
        "input[name='limit']",
        "select[name='opt_id']",
        "select[name='opt_last']",
        "select[name='opt_run']",
        "select[name='opt_pick']"
      ];
      selectors.forEach(sel => {{
        document.querySelectorAll(sel).forEach(el => {{
          el.addEventListener("change", () => {{
            clearOverrides();
          }});
        }});
      }});

      // Playground (select + slider): update only the plot via /api/chart_update
      const playParam = document.getElementById('play_param');
      const playSlider = document.getElementById('play_slider');
      const playVal = document.getElementById('play_val');
      const playSt = document.getElementById('play_status');
      const playSave = document.getElementById('play_save');
      const playSaveMsg = document.getElementById('play_save_msg');
      const playSpecEl = document.getElementById('play_specs_json');
      let playSpecs = null;
      try {{
        if (playSpecEl && playSpecEl.textContent) playSpecs = JSON.parse(playSpecEl.textContent);
      }} catch (e) {{
        playSpecs = null;
      }}

      function fmtVal(param, v) {{
        if (!playSpecs || !playSpecs[param]) return String(v);
        const sp = playSpecs[param];
        const digits = Number(sp.digits || 4);
        if (sp.kind === 'int') return String(Math.round(Number(v)));
        return Number(v).toFixed(digits);
      }}

      function applySpec(param) {{
        if (!playSpecs || !playSpecs[param] || !playSlider) return;
        const sp = playSpecs[param];
        playSlider.min = String(sp.min);
        playSlider.max = String(sp.max);
        playSlider.step = String(sp.step);

        // Pull current value from params input if present, otherwise default.
        const inp = document.getElementById('p_' + param);
        let v = (inp && inp.value !== '') ? inp.value : String(sp.default);
        if (sp.kind === 'int') v = String(Math.round(Number(v)));
        playSlider.value = v;
        if (playVal) playVal.textContent = fmtVal(param, v);
      }}

      async function doPlayUpdate() {{
        if (!playParam || !playSlider) return;
        const param = playParam.value;
        const sp = playSpecs && playSpecs[param] ? playSpecs[param] : null;
        let v = playSlider.value;
        if (sp && sp.kind === 'int') v = String(Math.round(Number(v)));

        if (playVal) playVal.textContent = fmtVal(param, v);

        // Enable overrides and sync the numeric input in the params panel (so Apply keeps it).
        const uo = document.getElementById('use_overrides');
        if (uo) uo.value = '1';
        if (window.__enableOverrides) window.__enableOverrides();
        const inp = document.getElementById('p_' + param);
        if (inp) inp.value = v;

        const form = document.getElementById('chartForm');
        const fd = new FormData(form);
        fd.set('use_overrides', '1');
        fd.set('p_' + param, v);
        const qs = new URLSearchParams(fd);
        const url = '/api/chart_update?' + qs.toString();

        if (doPlayUpdate.ctrl) doPlayUpdate.ctrl.abort();
        doPlayUpdate.ctrl = new AbortController();
        if (playSt) playSt.textContent = 'loading…';

        try {{
          const res = await fetch(url, {{ signal: doPlayUpdate.ctrl.signal }});
          const j = await res.json();
          if (!j || !j.ok) throw new Error('bad response');
          if (window.Plotly && document.getElementById('chartPlot')) {{
            Plotly.react('chartPlot', j.data || [], j.layout || {{}}, j.config || {{ responsive: true }});
          }}
          // Update trades table (metrics + trades) if provided
          if (j.trades_html) {{
            const tw = document.getElementById('tradesWrap');
            if (tw) tw.outerHTML = j.trades_html;
          }}
          if (playSt) playSt.textContent = '';
        }} catch (e) {{
          if (e && e.name === 'AbortError') return;
          if (playSt) playSt.textContent = 'error';
        }}
      }}

      function collectOverrides() {{
        // Collect all editable p_* controls as overrides (numbers + bools).
        const out = {{}};
        const els = document.querySelectorAll("[data-ov='1']");
        els.forEach(el => {{
          if (!el || !el.id || !el.id.startsWith('p_')) return;
          const key = el.id.substring(2);
          let v = (el.value !== undefined) ? String(el.value) : '';
          if (v === '') return;

          if (el.tagName === 'SELECT') {{
            // bool selects use true/false
            if (v === 'true') out[key] = true;
            else if (v === 'false') out[key] = false;
            else out[key] = v;
            return;
          }}

          const n = Number(v);
          if (Number.isNaN(n)) return;

          // int keys
          if (key === 'rev_cooldown_hrs' || key === 'max_flips_per_day' || key === 'slippage_ticks') {{
            out[key] = Math.round(n);
          }} else {{
            out[key] = n;
          }}
        }});
        return out;
      }}

      async function saveSnapshot() {{
        if (!playSave) return;
        const symbolEl = document.querySelector("select[name='symbol']");
        const tfEl = document.querySelector("input[name='tf']");
        const limitEl = document.querySelector("input[name='limit']");
        const stratEl = document.querySelector("select[name='strategy']");
        const optEl = document.querySelector("select[name='opt_id']");
        const capEl = document.getElementById('capital_usd');

        const payload = {{
          symbol: symbolEl ? symbolEl.value : '',
          tf: tfEl ? tfEl.value : '',
          limit: limitEl ? Number(limitEl.value || 0) : 0,
          strategy: stratEl ? stratEl.value : '',
          opt_id: optEl ? optEl.value : null,
          capital_usd: capEl ? Number(capEl.value || 0) : 0,
          overrides: collectOverrides(),
        }};

        if (playSaveMsg) playSaveMsg.textContent = 'Saving…';

        try {{
          const res = await fetch('/api/save_chart_snapshot', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(payload)
          }});
          const j = await res.json();
          if (!j || !j.ok) throw new Error((j && j.error) || 'save failed');

          const id = j.id;
          let msg = 'Saved';
          if (id !== null && id !== undefined) {{
            const url = new URL(window.location.href);
            url.searchParams.set('opt_id', String(id));
            url.searchParams.set('use_overrides', '0');
            msg = `Saved as #${{id}} — reload via link`; 
            if (playSaveMsg) {{
              playSaveMsg.innerHTML = `<a href="${{url.toString()}}" style="color:#93c5fd;">${{msg}}</a>`;
            }}
          }} else {{
            if (playSaveMsg) playSaveMsg.textContent = msg;
          }}
        }} catch (e) {{
          if (playSaveMsg) playSaveMsg.textContent = 'Save error';
        }}
      }}

      if (playParam && playSlider && playSpecs) {{
        // Initial spec
        applySpec(playParam.value);

        // Changing param updates slider bounds and triggers chart update.
        playParam.addEventListener('change', () => {{
          applySpec(playParam.value);
          doPlayUpdate();
        }});

        // Update on release (change) to avoid spamming server.
        playSlider.addEventListener('change', doPlayUpdate);

        // Live value preview while dragging (no fetch).
        playSlider.addEventListener('input', () => {{
          if (playVal) playVal.textContent = fmtVal(playParam.value, playSlider.value);
        }});
      }}

      if (playSave) {{
        playSave.addEventListener('click', saveSnapshot);
      }}


      // Manual trade (server-side Bybit) buttons
      let mtSending = false;
      function mtSetButtonsDisabled(disabled) {{
        ['btnLong', 'btnShort', 'btnClose'].forEach(function(id) {{
          const el = document.getElementById(id);
          if (!el) return;
          el.disabled = disabled;
          el.style.opacity = disabled ? '0.6' : '1';
          el.style.cursor = disabled ? 'not-allowed' : 'pointer';
        }});
      }}
      function mtGetCurrentSymbolBase() {{
        const manualEl = document.getElementById('mt_symbol');
        const manualVal = (manualEl && manualEl.value) ? String(manualEl.value).trim() : '';
        if (manualVal) return manualVal;

        const sel = document.querySelector("select[name='symbol']");
        if (sel && sel.value) return String(sel.value).trim();

        const qs = new URLSearchParams(location.search);
        const fromUrl = qs.get('symbol');
        if (fromUrl && fromUrl.trim()) return fromUrl.trim();

        return 'APTUSDT';
      }}
      async function mtSendTradeAction(action) {{
        if (mtSending) return;
        const out = document.getElementById('mt_result');
        const tokenEl = document.getElementById('mt_token');
        const tfEl = document.getElementById('mt_tf');
        const noteEl = document.getElementById('mt_note');
        const token = ((tokenEl && tokenEl.value) || '').trim();
        const tfv = ((tfEl && tfEl.value) || '2h').trim();
        const note = ((noteEl && noteEl.value) || 'manual from terminal').trim();
        const symbolBase = mtGetCurrentSymbolBase();
        const symbolBridge = symbolBase.endsWith('.P') ? symbolBase : (symbolBase + '.P');
        if (['LONG', 'SHORT', 'CLOSE'].indexOf(action) === -1) {{ if (out) out.textContent = 'Unsupported action: ' + action; return; }}
        if (!confirm('Отправить ' + action + ' на ' + symbolBridge + ' (tf=' + tfv + ')?')) return;
        const payload = {{ symbol: symbolBridge, tf: tfv, action: action, note: note }};
        mtSending = true;
        mtSetButtonsDisabled(true);
        if (out) out.textContent = '[' + new Date().toLocaleString() + '] Sending ' + action + '...\\n' + JSON.stringify(payload, null, 2);
        try {{
          const headers = {{ 'Content-Type': 'application/json' }};
          if (token) headers['x-trade-token'] = token;
          const resp = await fetch('/trade/execute', {{
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
          }});
          const txtResp = await resp.text();
          let parsed;
          try {{ parsed = JSON.parse(txtResp); }} catch (_) {{ parsed = txtResp; }}
          if (out) {{
            var payloadResp = parsed || {{}};
            var r = payloadResp.response || payloadResp;
            var rr = (r && r.result) ? r.result : {{}};
            var ex = (rr && rr.response) ? rr.response : ((r && r.response) ? r.response : {{}});
            var exResult = (ex && ex.result) ? ex.result : {{}};
            var orderId = exResult.orderId || '-';
            var retCode = (typeof ex.retCode !== 'undefined') ? ex.retCode : null;
            var retMsg = (typeof ex.retMsg !== 'undefined') ? ex.retMsg : '';
            var okFlag = resp.ok && !payloadResp.error && (r.ok !== false) && (rr.ok !== false) && (!rr.error) && (retCode === null || retCode === 0);
            if (okFlag) {{
              out.textContent = '[' + new Date().toLocaleString() + '] ' + action + ' ' + payload.symbol + ' — OK (orderId=' + orderId + ')';
            }} else {{
              out.textContent = '[' + new Date().toLocaleString() + '] ' + action + ' ' + payload.symbol + ' — ERROR (' + (retMsg || rr.error || payloadResp.detail || 'Request failed') + ')';
            }}
          }}
        }} catch (e) {{
          if (out) out.textContent = '[' + new Date().toLocaleString() + '] ERROR\\n' + (e && e.message ? e.message : String(e));
        }} finally {{
          mtSending = false;
          mtSetButtonsDisabled(false);
        }}
      }}
      var mtBtnLong = document.getElementById('btnLong');
      var mtBtnShort = document.getElementById('btnShort');
      var mtBtnClose = document.getElementById('btnClose');
      if (mtBtnLong) mtBtnLong.addEventListener('click', function() {{ mtSendTradeAction('LONG'); }});
      if (mtBtnShort) mtBtnShort.addEventListener('click', function() {{ mtSendTradeAction('SHORT'); }});
      if (mtBtnClose) mtBtnClose.addEventListener('click', function() {{ mtSendTradeAction('CLOSE'); }});

    }});
  </script>
</head>
<body>
  <form id="chartForm" class="topbar" method="get" action="/chart">
    <a class="navlink active" href="/chart?symbol={html.escape(symbol)}&tf={tf_e}&strategy={html.escape(strategy)}&limit={int(limit)}">📈 Chart</a>
    <a class="navlink" href="/optimize">🧪 Optimizer</a>
    <label>Symbol</label>
    <select name="symbol">{symbol_options}</select>

    <label>TF</label>
    <input id="tf" name="tf" value="{tf_e}" style="width:70px"/>

    <label>Strategy</label>
    <select name="strategy">{strategy_options}</select>

    {opt_controls_html}

    <input type="hidden" name="use_overrides" id="use_overrides" value="{int(use_overrides)}"/>

    <label>Limit</label>
    <input name="limit" value="{int(limit)}" style="width:90px"/>

    <button class="apply" type="submit">Apply</button>
  </form>

  <div class="tf-row">{tf_buttons_html}</div>
  <div class="wrap">{params_html}{slider_html}{manual_trade_html}{plot_html}{trades_table_html}</div>
</body>
</html>
"""


__all__ = ["make_chart_html", "build_plotly_payload"]

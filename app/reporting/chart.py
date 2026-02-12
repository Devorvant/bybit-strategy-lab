from __future__ import annotations

import datetime
import html
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.backtest.sma_backtest import backtest_sma_cross
from app.backtest.strategy2_backtest import backtest_sma_adx_filter
from app.backtest.strategy3_backtest import backtest_strategy3
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


def _build_plot_html(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    strategy: str,
) -> str:
    """Return Plotly HTML fragment (no outer <html> tag)."""
    if not bars:
        return '<div style="padding:12px;font-size:14px;">No bars yet for this symbol/tf.</div>'

    df = pd.DataFrame(bars, columns=["ts", "o", "h", "l", "c", "v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")

    # Indicators (keep in sync with strategy defaults)
    df["sma20"] = df["c"].rolling(20).mean()
    df["sma50"] = df["c"].rolling(50).mean()

    cross_up = (df["sma20"].shift(1) <= df["sma50"].shift(1)) & (df["sma20"] > df["sma50"])
    cross_dn = (df["sma20"].shift(1) >= df["sma50"].shift(1)) & (df["sma20"] < df["sma50"])

    # Backtest to draw trades + equity
    if strategy == "my_strategy3.py":
        bt = backtest_strategy3(
            bars,
            position_usd=1000.0,
            adx_len=14,
            adx_smooth=14,
            adx_no_trade_below=14.0,
            st_atr_len=14,
            st_factor=4.0,
            use_no_trade=True,
            use_rev_cooldown=True,
            rev_cooldown_hrs=8,
            use_flip_limit=False,
            max_flips_per_day=6,
            use_emergency_sl=True,
            atr_len=14,
            sl_atr_mult=3.0,
            close_at_end=False,
        )
        # Attach Supertrend/ADX series (aligned with `bars`) so plotting can
        # render overlays without KeyError.
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
            entry_text.append(
                f"{tr.side} entry<br>px={tr.entry_price:.6g}<br>ts={tr.entry_ts}"
            )

            exit_x.append(pd.to_datetime(tr.exit_ts, unit="ms"))
            exit_y.append(ts_to_px.get(tr.exit_ts, tr.exit_price))
            exit_text.append(
                f"{tr.side} exit<br>px={tr.exit_price:.6g}<br>pnl={tr.pnl:.2f} USD"
            )

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
    # full_html=False so we can wrap with our own UI
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _build_trades_table_html(bt) -> str:
    """
    HTML-таблица сделок, похожая по духу на TradingView:
    - последние сделки сверху
    - нумерация по хронологии (максимальный номер = самая свежая)
    - если есть открытая позиция, она отображается первой строкой (OPEN)
    """
    trades = list(getattr(bt, "trades", []) or [])

    def dt_str(ts: int) -> str:
        """Format timestamp to a readable UTC datetime.

        Bybit kline timestamps are in **milliseconds**, but we also tolerate seconds.
        """
        try:
            ts_int = int(ts)
            secs = (ts_int / 1000.0) if ts_int > 10_000_000_000 else float(ts_int)
            return datetime.datetime.utcfromtimestamp(secs).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    total_trades = len(trades)
    wins = sum(1 for t in trades if float(getattr(t, "pnl", 0.0)) > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0

    # cumulative PnL по закрытым сделкам (хронологически)
    cum_pnl = []
    cum = 0.0
    for tr in trades:
        cum += float(getattr(tr, "pnl", 0.0))
        cum_pnl.append(cum)

    realized_pnl = cum_pnl[-1] if cum_pnl else 0.0

    open_pos = getattr(bt, "open_position", None)
    open_pnl = float(getattr(open_pos, "unrealized_pnl", 0.0)) if open_pos else 0.0
    net_pnl = realized_pnl + open_pnl
    avg_pnl = (realized_pnl / total_trades) if total_trades else 0.0

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
        reason_cls = "reason-stop" if reason_norm == "STOP" else "reason"
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

    # открытая позиция (если есть) — первой строкой
    if open_pos is not None:
        trade_no = total_trades + 1
        side = str(getattr(open_pos, "side", "LONG"))
        entry_dt = dt_str(getattr(open_pos, "entry_ts", 0))
        exit_dt = dt_str(getattr(open_pos, "current_ts", getattr(open_pos, "entry_ts", 0))) + " (OPEN)"
        entry_px = f"{float(getattr(open_pos, 'entry_price', 0.0)):.6g}"
        exit_px = f"{float(getattr(open_pos, 'current_price', 0.0)):.6g}"
        rows.append(
            row_html(
                trade_no,
                "OPEN",
                side,
                "OPEN",
                entry_dt,
                exit_dt,
                entry_px,
                exit_px,
                open_pnl,
                net_pnl,
            )
        )

    # закрытые сделки: newest-first, но trade_no из хронологии
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
                "CLOSED",
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
    <div class="trades-wrap">
      <div class="trades-metrics">
        <div><b>Total trades:</b> {total_trades}{' (+1 open)' if open_pos is not None else ''}</div>
        <div><b>Win rate:</b> {win_rate:.1f}%</div>
        <div><b>Realized PnL:</b> {realized_pnl:,.2f} USD</div>
        <div><b>Unrealized PnL:</b> {open_pnl:,.2f} USD</div>
        <div><b>Net PnL:</b> {net_pnl:,.2f} USD</div>
        <div><b>Avg realized / trade:</b> {avg_pnl:,.2f} USD</div>
      </div>

      <div class="trades-table">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Type</th>
              <th>Side</th>
              <th>Reason</th>
              <th>Entry time</th>
              <th>Exit time</th>
              <th>Entry px</th>
              <th>Exit px</th>
              <th>PnL</th>
              <th>Cum PnL</th>
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
) -> str:
    """Render a simple chart page with controls (symbol/tf/limit)."""

    symbol_e = html.escape(symbol)
    tf_e = html.escape(tf)

    symbols_list = [s for s in (symbols or [])] or [symbol]
    # Ensure current symbol is present
    if symbol not in symbols_list:
        symbols_list = [symbol] + [s for s in symbols_list if s != symbol]

    tfs_list = [x for x in (tfs or [])] or [x[0] for x in TF_BUTTONS]
    if tf not in tfs_list:
        tfs_list = [tf] + [x for x in tfs_list if x != tf]

    available_strategies = [
        "my_strategy.py",
        "my_strategy2.py",
        "my_strategy3.py",
        "my_strategy3_tv_like.py",
    ]
    if strategy not in available_strategies:
        strategy = "my_strategy.py"

    plot_html = _build_plot_html(bars, strategy=strategy)

    # Build trades table (TradingView-like)
    try:
        # close_at_end=False чтобы отображать открытую позицию (если она есть)
        if strategy == "my_strategy2.py":
            bt = backtest_sma_adx_filter(bars, close_at_end=False)
        elif strategy == "my_strategy3.py":
            bt = backtest_strategy3(bars, close_at_end=False)
        elif strategy == "my_strategy3_tv_like.py":
            bt = backtest_strategy3_tv_like(
                bars,
                initial_capital=10000.0,
                percent_of_equity=50.0,
                commission_percent=0.10,
                slippage_ticks=2,
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
        trades_table_html = _build_trades_table_html(bt)
    except Exception as e:
        trades_table_html = f"<div class=\"trades-empty\">Trades table error: {html.escape(str(e))}</div>"

    # Build dropdown options
    symbol_options = "\n".join(f'<option value="{html.escape(s)}"></option>' for s in symbols_list)
    tf_options = "\n".join(f'<option value="{html.escape(x)}"></option>' for x in tfs_list)
    strategy_select_options = "\n".join(
        f'<option value="{html.escape(s)}" {"selected" if s == strategy else ""}>{html.escape(s)}</option>'
        for s in available_strategies
    )

    # Build timeframe buttons
    btns = []
    for tf_val, tf_label in TF_BUTTONS:
        active = "active" if tf_val == tf else ""
        btns.append(
            f'<button type="button" class="btn {active}" data-tf="{html.escape(tf_val)}">{html.escape(tf_label)}</button>'
        )
    tf_buttons_html = "\n".join(btns)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>Chart {symbol_e} tf={tf_e}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background: #0b1220; color: #e5e7eb; }}
    .topbar {{
      position: sticky; top: 0; z-index: 10;
      background: #0b1220;
      padding: 10px 12px;
      display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .group {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
    label {{ font-size: 13px; opacity: 0.95; display:flex; gap:6px; align-items:center; }}
    select, input {{
      background: #111827; color: #e5e7eb; border: 1px solid rgba(255,255,255,0.12);
      padding: 6px 8px; border-radius: 10px; outline: none;
    }}
    input[type=number] {{ width: 120px; }}
    .btn {{
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      color: #e5e7eb;
      padding: 6px 10px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 13px;
    }}
    .btn:hover {{ background: rgba(255,255,255,0.08); }}
    .btn.active {{ background: #2563eb; border-color: #2563eb; }}
    .btn.primary {{ background: rgba(37,99,235,0.15); border-color: rgba(37,99,235,0.45); }}
    .chart-wrap {{ padding: 10px; background: #0b1220; }}
    .trades-wrap {{ padding: 10px; background: #0b1220; }}
    .trades-table table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .trades-table th, .trades-table td {{
      padding: 6px 8px;
      border: 1px solid rgba(255,255,255,0.14);
      text-align: left;
      white-space: nowrap;
    }}
    .trades-table th {{ font-weight: 600; color: rgba(255,255,255,0.92); }}
    .trades-table td {{ color: rgba(255,255,255,0.88); }}
    .trades-table thead th {{ position: sticky; top: 0; background: #0b1220; z-index: 2; }}
    .trades-table tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.02); }}
    .trades-table tbody tr:hover {{ background: rgba(255,255,255,0.05); }}

    .trades-table td.num {{ text-align: right; width: 40px; }}
    .trades-table td.dt {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .trades-table td.px {{ text-align: right; }}
    .pnl-pos {{ color: #7CFC00; }}
    .pnl-neg {{ color: #FF6B6B; }}
    .reason-stop {{ color: #FFB86C; font-weight: 700; }}
    /* Plotly uses white background by default; keep it readable */
    .plotly-graph-div {{ border-radius: 14px; overflow: hidden; }}
  </style>
</head>
<body>
	  <div class=\"topbar\">
	    <form id=\"ctrl\" class=\"group\" method=\"get\" action=\"/chart\" autocomplete=\"off\">
	      <label>Symbol
	        <input name=\"symbol\" id=\"symbol\" list=\"symbols\" value=\"{html.escape(symbol)}\" spellcheck=\"false\" />
	        <datalist id=\"symbols\">{symbol_options}</datalist>
	      </label>

	      <label>TF
	        <input name=\"tf\" id=\"tf\" list=\"tfs\" value=\"{html.escape(tf)}\" spellcheck=\"false\" style=\"width:70px\" />
	        <datalist id=\"tfs\">{tf_options}</datalist>
	      </label>

	      <label>Strategy
	        <select name=\"strategy\" id=\"strategy\" style=\"width:170px\">
	          {strategy_select_options}
	        </select>
	      </label>

	      <label>Limit
	        <input name=\"limit\" id=\"limit\" type=\"number\" min=\"10\" max=\"50000\" value=\"{int(limit)}\" />
	      </label>

	      <button class=\"btn primary\" type=\"submit\">Apply</button>
	    </form>

    <div class=\"group\" style=\"gap:8px;\">
      {tf_buttons_html}
    </div>
  </div>

	  <div class=\"chart-wrap\">
	    {plot_html}
	  </div>

	  <div class=\"trades-wrap\">
	    {trades_table_html}
	  </div>

  <script>
    const form = document.getElementById('ctrl');
    const tfInput = document.getElementById('tf');
    const symInput = document.getElementById('symbol');

    // Submit on change (after typing/selecting from datalist)
    tfInput.addEventListener('change', () => form.submit());
    symInput.addEventListener('change', () => form.submit());

    // Enter submits (nice for manual typing)
    tfInput.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') form.submit();
    }});
    symInput.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') form.submit();
    }});

    // TF buttons set tf and submit
    document.querySelectorAll('button[data-tf]').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        tfInput.value = btn.getAttribute('data-tf');
        form.submit();
      }});
    }});
  </script>
</body>
</html>"""

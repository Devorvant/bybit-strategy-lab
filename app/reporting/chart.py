from __future__ import annotations

import html
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.backtest.sma_backtest import backtest_sma_cross


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


def _build_plot_html(bars: Sequence[Tuple[int, float, float, float, float, float]]) -> str:
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
    bt = backtest_sma_cross(bars, position_usd=1000.0, fast_n=20, slow_n=50, close_at_end=True)
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
    """TradingView-like trades list.

    bt is BacktestResult returned by `backtest_sma_cross`.
    """
    trades = getattr(bt, "trades", None) or []
    if not trades:
        return "<div class=\"trades-empty\">No trades</div>"

    # Summary
    pnls = [float(getattr(t, "pnl", 0.0) or 0.0) for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = (wins / len(pnls)) * 100.0 if pnls else 0.0
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0

    # Build rows
    rows = []
    cum = 0.0
    for i, t in enumerate(trades, start=1):
        side = html.escape(str(getattr(t, "side", "")))
        entry_ts = int(getattr(t, "entry_ts", 0) or 0)
        exit_ts = int(getattr(t, "exit_ts", 0) or 0)
        entry_px = float(getattr(t, "entry_price", 0.0) or 0.0)
        exit_px = float(getattr(t, "exit_price", 0.0) or 0.0)
        pnl = float(getattr(t, "pnl", 0.0) or 0.0)
        cum += pnl

        entry_dt = pd.to_datetime(entry_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
        exit_dt = pd.to_datetime(exit_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
        dur_s = max(0, (exit_ts - entry_ts) / 1000.0)
        dur_h = dur_s / 3600.0

        pnl_cls = "pnl-pos" if pnl > 0 else ("pnl-neg" if pnl < 0 else "pnl-zero")

        rows.append(
            """
            <tr>
              <td class=\"num\">{i}</td>
              <td class=\"side\">{side}</td>
              <td>{entry_dt}</td>
              <td class=\"num\">{entry_px:.6g}</td>
              <td>{exit_dt}</td>
              <td class=\"num\">{exit_px:.6g}</td>
              <td class=\"num {pnl_cls}\">{pnl:.2f}</td>
              <td class=\"num\">{cum:.2f}</td>
              <td class=\"num\">{dur_h:.2f}h</td>
            </tr>
            """.format(
                i=i,
                side=side,
                entry_dt=entry_dt,
                entry_px=entry_px,
                exit_dt=exit_dt,
                exit_px=exit_px,
                pnl_cls=pnl_cls,
                pnl=pnl,
                cum=cum,
                dur_h=dur_h,
            )
        )

    rows_html = "\n".join(rows)

    return f"""
<section class=\"trades\">
  <div class=\"trades-head\">
    <div class=\"trades-title\">Trades</div>
    <div class=\"trades-metrics\">
      <span><b>{len(trades)}</b> trades</span>
      <span>Win rate <b>{win_rate:.1f}%</b></span>
      <span>Net PnL <b class=\"{('pnl-pos' if total_pnl>0 else 'pnl-neg' if total_pnl<0 else 'pnl-zero')}\">{total_pnl:.2f}</b></span>
      <span>Avg <b>{avg_pnl:.2f}</b></span>
    </div>
  </div>

  <div class=\"table-wrap\">
    <table class=\"trades-table\">
      <thead>
        <tr>
          <th>#</th>
          <th>Side</th>
          <th>Entry time (UTC)</th>
          <th class=\"num\">Entry</th>
          <th>Exit time (UTC)</th>
          <th class=\"num\">Exit</th>
          <th class=\"num\">PnL</th>
          <th class=\"num\">Cum PnL</th>
          <th class=\"num\">Dur</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</section>
"""


def make_chart_html(
    bars: Sequence[Tuple[int, float, float, float, float, float]],
    *,
    symbol: str = "APTUSDT",
    tf: str = "120",
    limit: int = 5000,
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

    plot_html = _build_plot_html(bars)

    # Build trades table (TradingView-like)
    try:
        bt = backtest_sma_cross(bars)
        trades_table_html = _build_trades_table_html(bt)
    except Exception as e:
        trades_table_html = f"<div class=\"trades-empty\">Trades table error: {html.escape(str(e))}</div>"

    # Build dropdown options
    symbol_options = "\n".join(f'<option value="{html.escape(s)}"></option>' for s in symbols_list)
    tf_options = "\n".join(f'<option value="{html.escape(x)}"></option>' for x in tfs_list)

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

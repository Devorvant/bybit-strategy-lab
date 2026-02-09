import pandas as pd
import plotly.graph_objects as go

def make_chart_html(bars):
    # bars: [(ts,o,h,l,c,v)...] ts in ms
    if not bars:
        return "<html><body><h3>No bars yet</h3></body></html>"

    df = pd.DataFrame(bars, columns=["ts","o","h","l","c","v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")

    fig = go.Figure(data=[go.Candlestick(
        x=df["dt"], open=df["o"], high=df["h"], low=df["l"], close=df["c"]
    )])
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    return fig.to_html(include_plotlyjs="cdn", full_html=True)

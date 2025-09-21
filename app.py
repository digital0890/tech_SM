import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand Analysis")

# Centered title
st.markdown("<h1 style='text-align: center;'>📈 Crypto & Gold Supply & Demand Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Sidebar for settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # انتخاب ارز از لیست
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "Gold"]
    symbol = st.selectbox("Select Symbol", options=symbols, index=1)

    timeframe = st.selectbox("Timeframe", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    lookback = st.slider("Lookback (for Supply/Demand points)", 1, 10, 3)

    # Default end datetime: today 23:59
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("End Date", value=default_end.date())
    end_time = st.time_input("End Time", value=default_end.time())

    required_candles = 500

    # Auto calculate start date based on timeframe and required candles
    tf_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1)
    }
    delta = tf_map[timeframe] * required_candles
    default_start = datetime.combine(end_date, end_time) - delta

    start_date = st.date_input("Start Date", value=default_start.date())
    start_time = st.time_input("Start Time", value=default_start.time())

# -------------------------------
# Convert to timestamp in ms
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
since = int(start_dt.timestamp() * 1000)
until = int(end_dt.timestamp() * 1000)

# -------------------------------
# Fetch data
# -------------------------------
if symbol == "Gold":
    # Map timeframe to yfinance intervals
    yf_tf_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "4h": "1h",   # yfinance ندارد 4h → با 1h جایگزین
        "1d": "1d"
    }
    yf_interval = yf_tf_map[timeframe]

    ticker = "GC=F"   # Gold Futures
    with st.spinner("Fetching Gold data from Yahoo Finance..."):
        df = yf.download(
            ticker,
            start=start_dt,
            end=end_dt,
            interval=yf_interval
        )

    if df.empty:
        st.error("No data found for Gold!")
        st.stop()

    # هماهنگ‌سازی با ساختار ccxt
    data = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    }).copy()

    data.index = data.index.tz_convert("Asia/Tehran")

else:
    exchange = ccxt.coinbase()
    ohlcv = []

    with st.spinner("Fetching crypto data from exchange..."):
        while since < until:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
            if len(batch) == 0:
                break
            ohlcv += batch
            since = batch[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

    if len(ohlcv) == 0:
        st.error("No data found! Check symbol or timeframe.")
        st.stop()

    # -------------------------------
    # Create DataFrame for crypto
    # -------------------------------
    data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
    data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
    data.set_index('timestamp', inplace=True)
    data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

# -------------------------------
# Calculations
# -------------------------------
data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
up = data[data["Close"] >= data["Open"]]
down = data[data["Close"] < data["Open"]]

supply_idx = []
demand_idx = []

for i in range(lookback, len(data)-lookback):
    high_window = data['High'].iloc[i-lookback:i+lookback+1]
    low_window = data['Low'].iloc[i-lookback:i+lookback+1]
    if data['High'].iloc[i] == max(high_window):
        supply_idx.append(i)
    if data['Low'].iloc[i] == min(low_window):
        demand_idx.append(i)

supply_idx_filtered = [i for i in supply_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
demand_idx_filtered = [i for i in demand_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]

# -------------------------------
# Display number of identified points
# -------------------------------
st.markdown(f"### 🔴 Supply Points: {len(supply_idx_filtered)} | 🟢 Demand Points: {len(demand_idx_filtered)}")

# -------------------------------
# Plot chart with smooth animation
# -------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7,0.3],
                    subplot_titles=(f"{symbol} Candlestick Chart", "Volume"))

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
), row=1, col=1)

# Supply points
fig.add_trace(go.Scatter(
    x=data.index[supply_idx_filtered],
    y=data['High'].iloc[supply_idx_filtered] + 5,
    mode='markers',
    marker=dict(symbol='triangle-up', color='red', size=12),
    name='Supply'
), row=1, col=1)

# Demand points
fig.add_trace(go.Scatter(
    x=data.index[demand_idx_filtered],
    y=data['Low'].iloc[demand_idx_filtered] - 5,
    mode='markers',
    marker=dict(symbol='triangle-down', color='green', size=12),
    name='Demand'
), row=1, col=1)

# Up & down volume bars
fig.add_trace(go.Bar(
    x=up.index,
    y=up['Volume'],
    name="Up Volume",
    marker_color="green",
    opacity=0.8
), row=2, col=1)

fig.add_trace(go.Bar(
    x=down.index,
    y=down['Volume'],
    name="Down Volume",
    marker_color="red",
    opacity=0.8
), row=2, col=1)

# MA20 Volume
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Volume_MA20'],
    mode="lines",
    name="MA20 Volume",
    line=dict(color="orange", width=2)
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    showlegend=True,
    height=800,
    barmode="overlay",
    hovermode='x unified',
    transition={'duration': 500, 'easing': 'cubic-in-out'}
)

# -------------------------------
# Display centered chart
# -------------------------------
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

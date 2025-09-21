import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand")

st.markdown("<h1 style='text-align: center;'>📈 Crypto & Gold Supply/Demand Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    data_source = st.radio("Data Source", ["Crypto (CCXT)", "Stock/Gold (yfinance)"])
    
    lookback = st.slider("Lookback (for Supply/Demand points)", 1, 10, 3)
    
    # Default end datetime
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("End Date", value=default_end.date())
    end_time = st.time_input("End Time", value=default_end.time())
    
    required_candles = 500
    
    if data_source == "Crypto (CCXT)":
        symbol = st.text_input("Symbol (CCXT)", value="ETH/USD")
        timeframe = st.selectbox("Timeframe", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
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
    else:
        # Gold / yfinance
        ticker = st.text_input("Ticker (yfinance)", value="GC=F")  # GC=F for Gold
        interval_map = {
            "1m":"1m", "5m":"5m", "15m":"15m", "30m":"30m", "1h":"60m", "4h":"240m", "1d":"1d"
        }
        interval_choice = st.selectbox("Interval", list(interval_map.keys()), index=4)
    
        # Dates for yfinance
        default_start = datetime.combine(end_date, end_time) - timedelta(days=730)  # default 2 years
        start_date = st.date_input("Start Date", value=default_start.date(), key="yf_start")
        start_time = st.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time(), key="yf_start_time")
    
        # Convert start & end datetime
        start_dt_yf = datetime.combine(start_date, start_time)
        end_dt_yf = datetime.combine(end_date, end_time)
    
        # Download data with start & end
        with st.spinner(f"Downloading {ticker} data from yfinance..."):
            df = yf.download(
                ticker,
                start=start_dt_yf,
                end=end_dt_yf,
                interval=interval_map[interval_choice]
            )
    
        # Prepare DataFrame
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Tehran')
        data = df[['Open','High','Low','Close','Volume']].copy()


# -------------------------------
# Fetch data
# -------------------------------
if data_source == "Crypto (CCXT)":
    # convert timestamps
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    since = int(start_dt.timestamp() * 1000)
    until = int(end_dt.timestamp() * 1000)
    
    exchange = ccxt.coinbase()
    ohlcv = []
    with st.spinner("Fetching crypto data from CCXT..."):
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
    
    # DataFrame
    data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
    data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
    data.set_index('timestamp', inplace=True)
    data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]
else:
    with st.spinner(f"Downloading {ticker} data from yfinance..."):
        df = yf.download(ticker, period=period, interval=interval_map[interval_choice])
    
    # Prepare data in the same format
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
    df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Tehran')
    data = df[['Open','High','Low','Close','Volume']].copy()

# -------------------------------
# Calculations (same as before)
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
# Plot
# -------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7,0.3],
                    subplot_titles=(f"{symbol if data_source=='Crypto (CCXT)' else ticker} Candlestick Chart", "Volume"))

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
), row=1, col=1)

# Supply/Demand points
fig.add_trace(go.Scatter(
    x=data.index[supply_idx_filtered],
    y=data['High'].iloc[supply_idx_filtered] + 5,
    mode='markers',
    marker=dict(symbol='triangle-up', color='red', size=12),
    name='Supply'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data.index[demand_idx_filtered],
    y=data['Low'].iloc[demand_idx_filtered] - 5,
    mode='markers',
    marker=dict(symbol='triangle-down', color='green', size=12),
    name='Demand'
), row=1, col=1)

# Volume bars
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

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

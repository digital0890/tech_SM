import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time

st.set_page_config(layout="wide", page_title="Crypto Supply/Demand")

st.title("تحلیل نقاط Supply و Demand با CCXT و Plotly")

# -------------------------------
# تنظیمات کاربر
# -------------------------------
symbol = st.text_input("نماد (Symbol)", value="ETH/USD")
timeframe = st.selectbox("تایم‌فریم", options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)

start_date = st.date_input("تاریخ شروع", value=pd.to_datetime("2025-01-01"))
end_date   = st.date_input("تاریخ پایان", value=pd.to_datetime("2025-01-10"))

lookback = st.slider("lookback (تعیین نقاط Supply/Demand)", min_value=1, max_value=10, value=3)

# -------------------------------
# دریافت داده با CCXT
# -------------------------------
exchange = ccxt.coinbase()

since = int(pd.Timestamp(start_date).timestamp() * 1000)
until = int(pd.Timestamp(end_date).timestamp() * 1000)

ohlcv = []
with st.spinner("در حال دریافت داده‌ها از صرافی..."):
    while since < until:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
        if len(batch) == 0:
            break
        ohlcv += batch
        since = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

if len(ohlcv) == 0:
    st.error("داده‌ای یافت نشد! تایم‌فریم و تاریخ‌ها را بررسی کنید.")
    st.stop()

# -------------------------------
# ساخت DataFrame
# -------------------------------
data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
data.set_index('timestamp', inplace=True)
data = data[data.index <= pd.Timestamp(end_date).tz_localize('Asia/Tehran')]

# -------------------------------
# محاسبات
# -------------------------------
data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
up = data[data["Close"] >= data["Open"]]
down = data[data["Close"] < data["Open"]]

# نقاط Supply/Demand
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
# رسم نمودار
# -------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"نمودار کندل‌استیک ({symbol})", "حجم معاملات"))

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="قیمت"
), row=1, col=1)

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

fig.add_trace(go.Bar(
    x=up.index,
    y=up['Volume'],
    name="حجم صعودی",
    marker_color="green",
    opacity=0.8
), row=2, col=1)

fig.add_trace(go.Bar(
    x=down.index,
    y=down['Volume'],
    name="حجم نزولی",
    marker_color="red",
    opacity=0.8
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Volume_MA20'],
    mode="lines",
    name="MA20 حجم",
    line=dict(color="orange", width=2)
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    showlegend=True,
    height=800,
    barmode="overlay",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

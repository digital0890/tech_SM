import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np

# --------------------------------------------------
# Dark Modern Themed Streamlit App - Enhanced UI
# --------------------------------------------------

# -------------------------------
# Enhanced Styling & Theme (Dark Modern)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap');"

DARK_BG = "#0f1724"
CARD_BG = "#0b1220"
ACCENT = "#6EE7B7"
ACCENT_SECOND = "#60A5FA"
TEXT = "#E6EEF5"
MUTED = "#98A4B3"
ERROR = "#FB7185"

CUSTOM_CSS = f"""
{FONT_IMPORT}
:root {{
  --bg: {DARK_BG};
  --card: {CARD_BG};
  --accent: {ACCENT};
  --accent-2: {ACCENT_SECOND};
  --text: {TEXT};
  --muted: {MUTED};
  --error: {ERROR};
}}

[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, rgba(5,10,16,1) 0%, rgba(10,14,20,1) 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}}

[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, rgba(6,10,16,0.9), rgba(8,12,20,0.9));
  border-right: 1px solid rgba(255,255,255,0.03);
  padding: 18px 14px 24px 18px;
}}

.streamlit-card {{
  background: var(--card);
  border-radius: 12px;
  padding: 22px;
  box-shadow: 0 6px 18px rgba(2,6,12,0.6);
  border: 1px solid rgba(255,255,255,0.03);
  margin-bottom: 20px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.streamlit-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(2,6,12,0.8);
}}

.chart-container {{
  border-radius: 16px;
  overflow: hidden;
  margin-top: 10px;
}}

.header-container {{
  background: linear-gradient(90deg, rgba(11,18,32,0.8) 0%, rgba(15,23,36,0.6) 100%);
  padding: 20px 24px;
  border-radius: 16px;
  margin-bottom: 24px;
  border: 1px solid rgba(255,255,255,0.05);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

.stSelectbox, .stSlider, .stDateInput, .stTimeInput {{
  margin-bottom: 16px;
}}

.stSelectbox > div > div {{
  background-color: rgba(15, 23, 36, 0.7);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
}}

.stButton > button {{
  border-radius: 10px;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT_SECOND});
  color: #0f1724;
  font-weight: 600;
  border: none;
  transition: all 0.3s ease;
}}

.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(110, 231, 183, 0.3);
}}

.plotly-graph-div {{ background: transparent !important; }}

/* Custom metric cards */
.metric-card {{
  background: linear-gradient(135deg, rgba(11,18,32,0.8), rgba(15,23,36,0.6));
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.05);
  text-align: center;
  margin: 8px 0;
}}

.metric-value {{
  font-size: 24px;
  font-weight: 700;
  color: {ACCENT};
  margin: 8px 0;
}}

.metric-label {{
  font-size: 14px;
  color: {MUTED};
}}

/* Loading animation */
@keyframes pulse {{
  0% {{ opacity: 1; }}
  50% {{ opacity: 0.5; }}
  100% {{ opacity: 1; }}
}}

.pulse {{
  animation: pulse 1.5s ease-in-out infinite;
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand Analysis", page_icon="📊")

# -------------------------------
# Stochastic Oscillator with Derivative Function
# -------------------------------
def calculate_stochastic_with_derivative(data, k_period=14, d_period=3, derivative_period=5):
    """
    Calculate Stochastic Oscillator with Derivative
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K
    Derivative = Rate of change of %K
    """
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    # Calculate basic Stochastic
    data['stoch_k'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
    
    # Calculate derivative (rate of change)
    data['stoch_derivative'] = data['stoch_k'].diff(derivative_period) / derivative_period
    
    # Calculate second derivative (acceleration)
    data['stoch_second_derivative'] = data['stoch_derivative'].diff(derivative_period) / derivative_period
    
    # Calculate momentum signals
    data['stoch_momentum'] = data['stoch_k'].diff(1)
    data['stoch_trend'] = data['stoch_momentum'].rolling(window=3).mean()
    
    return data

# -------------------------------
# Sidebar with improved UI
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:20px; text-align:center;'>
        <h2 style='margin:0; background:linear-gradient(90deg, #6EE7B7, #60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>📊 Analysis</h2>
        <p style='margin:4px 0 0 0; color:#98A4B3; font-size:14px;'>Supply/Demand Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "DOGE/USD", "TRX/USD", "ADA/USD", "===========", "Gold"]
    symbol = st.selectbox("**Select Symbol**", options=symbols, index=1, help="Choose the asset to analyze")
    
    timeframe = st.selectbox("**Timeframe**", options=["1m","5m","15m","30m","1h","4h","1d"], index=4, help="Select the chart timeframe")
    lookback = st.slider("**Lookback**", 1, 10, 10, help="Number of periods to look back for Supply/Demand points")

    st.markdown("---")
    
    # Stochastic parameters
    st.markdown("**Stochastic Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        k_period = st.slider("**%K Period**", 5, 21, 14, help="Period for %K line")
    with col2:
        d_period = st.slider("**%D Period**", 2, 7, 3, help="Period for %D line (SMA of %K)")
    
    st.markdown("---")
    
    # Derivative parameters
    st.markdown("**Derivative Parameters**")
    col3, col4 = st.columns(2)
    with col3:
        derivative_period = st.slider("**Derivative Period**", 1, 10, 3, 
                                    help="Period for calculating rate of change")
    with col4:
        smooth_period = st.slider("**Smooth Period**", 1, 5, 2, 
                                help="Smoothing period for derivative")
    
    st.markdown("---")
    
    st.markdown("**Date Range**")
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("End Date", value=default_end.date(), label_visibility="collapsed")
    end_time = st.time_input("End Time", value=default_end.time(), label_visibility="collapsed")

    required_candles = 500
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

    start_date = st.date_input("Start Date", value=default_start.date(), label_visibility="collapsed")
    start_time = st.time_input("Start Time", value=default_start.time(), label_visibility="collapsed")
    
    st.markdown("---")
    
    # Add some metrics in the sidebar
    st.markdown("**Data Info**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Timeframe</div>
            <div class="metric-value">{timeframe}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lookback</div>
            <div class="metric-value">{lookback}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# Convert to timestamp
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
since = int(start_dt.timestamp() * 1000)
until = int(end_dt.timestamp() * 1000)

# -------------------------------
# Main content area
# -------------------------------
st.markdown(f"""
<div class="header-container">
    <h1 style="margin:0; font-size:32px;">{symbol}</h1>
    <p style="margin:4px 0 0 0; color:#98A4B3; font-size:16px;">
        Period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Fetch data
# -------------------------------
main_container = st.container()

with main_container:
    if symbol == "Gold":
        yf_tf_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "60m",
            "1d": "1d"
        }
        yf_interval = yf_tf_map[timeframe]

        ticker = "GC=F"
        with st.spinner("🔄 Fetching Gold data from Yahoo Finance..."):
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval=yf_interval,
                progress=False
            )

        if df.empty:
            st.error("No data found for Gold!")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        data = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }).copy()

        if timeframe == "4h":
            data = data.resample("4H").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()

        try:
            data.index = data.index.tz_convert("Asia/Tehran")
        except Exception:
            data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tehran")

    else:
        exchange = ccxt.coinbase()
        ohlcv = []

        with st.spinner("🔄 Fetching crypto data from exchange..."):
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

        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)
        data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

    # -------------------------------
    # Calculations
    # -------------------------------
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
    
    # Calculate Stochastic Oscillator with Derivative
    data = calculate_stochastic_with_derivative(data, k_period, d_period, derivative_period)
    
    # Apply smoothing to derivative if needed
    if smooth_period > 1:
        data['stoch_derivative_smooth'] = data['stoch_derivative'].rolling(window=smooth_period).mean()
        data['stoch_second_derivative_smooth'] = data['stoch_second_derivative'].rolling(window=smooth_period).mean()
    else:
        data['stoch_derivative_smooth'] = data['stoch_derivative']
        data['stoch_second_derivative_smooth'] = data['stoch_second_derivative']
    
    up = data[data["Close"] >= data["Open"]]
    down = data[data["Close"] < data["Open"]]

    supply_idx = []
    demand_idx = []

    for i in range(lookback, len(data)-lookback):
        high_window = data['High'].iloc[i-lookback:i+lookback+1]
        low_window = data['Low'].iloc[i-lookback:i+lookback+1]
        if data['High'].iloc[i] == high_window.max():
            supply_idx.append(i)
        if data['Low'].iloc[i] == low_window.min():
            demand_idx.append(i)

    supply_idx_filtered = [i for i in supply_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
    demand_idx_filtered = [i for i in demand_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]

    # Display some metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${data['Close'].iloc[-1]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_change = data['Close'].iloc[-1] - data['Open'].iloc[-1]
        change_percent = (price_change / data['Open'].iloc[-1]) * 100
        change_color = ACCENT if price_change >= 0 else ERROR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Change</div>
            <div class="metric-value" style="color:{change_color};">{change_percent:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Current Stochastic values
        current_k = data['stoch_k'].iloc[-1] if not pd.isna(data['stoch_k'].iloc[-1]) else 0
        current_d = data['stoch_d'].iloc[-1] if not pd.isna(data['stoch_d'].iloc[-1]) else 0
        stoch_color = ERROR if current_k > 80 else (ACCENT if current_k < 20 else MUTED)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stochastic %K</div>
            <div class="metric-value" style="color:{stoch_color};">{current_k:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stochastic %D</div>
            <div class="metric-value" style="color:{stoch_color};">{current_d:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Additional metrics for derivative
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        current_derivative = data['stoch_derivative_smooth'].iloc[-1] if not pd.isna(data['stoch_derivative_smooth'].iloc[-1]) else 0
        deriv_color = ACCENT if current_derivative > 0 else ERROR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stoch Derivative</div>
            <div class="metric-value" style="color:{deriv_color};">{current_derivative:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        current_second_deriv = data['stoch_second_derivative_smooth'].iloc[-1] if not pd.isna(data['stoch_second_derivative_smooth'].iloc[-1]) else 0
        second_deriv_color = ACCENT if current_second_deriv > 0 else ERROR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stoch Acceleration</div>
            <div class="metric-value" style="color:{second_deriv_color};">{current_second_deriv:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        current_momentum = data['stoch_momentum'].iloc[-1] if not pd.isna(data['stoch_momentum'].iloc[-1]) else 0
        momentum_color = ACCENT if current_momentum > 0 else ERROR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stoch Momentum</div>
            <div class="metric-value" style="color:{momentum_color};">{current_momentum:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col8:
        trend_direction = "Bullish" if data['stoch_trend'].iloc[-1] > 0 else "Bearish" if data['stoch_trend'].iloc[-1] < 0 else "Neutral"
        trend_color = ACCENT if trend_direction == "Bullish" else ERROR if trend_direction == "Bearish" else MUTED
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stoch Trend</div>
            <div class="metric-value" style="color:{trend_color};">{trend_direction}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Enhanced Chart with Stochastic and Derivative
    # -------------------------------
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        subplot_titles=('Price Chart with Supply/Demand Zones', 'Volume', 'Stochastic Oscillator', 'Stochastic Derivative')
    )

    # Price chart (row 1)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
        increasing_fillcolor="#26A69A",
        decreasing_fillcolor="#EF5350"
    ), row=1, col=1)

    data["Candle_Range"] = data["High"] - data["Low"]
    avg_range = data["Candle_Range"].mean() if len(data)>0 else 0
    offset = avg_range * 0.2

    fig.add_trace(go.Scatter(
        x=data.index[supply_idx_filtered],
        y=data['High'].iloc[supply_idx_filtered] + offset,
        mode='markers',
        marker=dict(symbol='triangle-down', color='rgba(251,113,133,0.95)', size=14, line=dict(width=2, color='white')),
        name='Supply Zone',
        hovertemplate='<b>Supply Zone</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index[demand_idx_filtered],
        y=data['Low'].iloc[demand_idx_filtered] - offset,
        mode='markers',
        marker=dict(symbol='triangle-up', color='rgba(110,231,183,0.95)', size=14, line=dict(width=2, color='white')),
        name='Demand Zone',
        hovertemplate='<b>Demand Zone</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
    ), row=1, col=1)

    # Volume chart (row 2)
    fig.add_trace(go.Bar(
        x=up.index,
        y=up['Volume'],
        name="Up Volume",
        marker_color="#26A69A",
        opacity=0.8
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=down.index,
        y=down['Volume'],
        name="Down Volume",
        marker_color="#EF5350",
        opacity=0.8
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA20'],
        mode="lines",
        name="MA20 Volume",
        line=dict(color="rgba(96,165,250,0.95)", width=2, dash='dash')
    ), row=2, col=1)

    # Stochastic chart (row 3)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stoch_k'],
        mode="lines",
        name="%K",
        line=dict(color=ACCENT, width=2),
        hovertemplate='%K: %{y:.2f}<extra></extra>'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stoch_d'],
        mode="lines",
        name="%D",
        line=dict(color=ACCENT_SECOND, width=2),
        hovertemplate='%D: %{y:.2f}<extra></extra>'
    ), row=3, col=1)

    # Add overbought/oversold lines to stochastic
    fig.add_hline(y=80, line_dash="dash", line_color=ERROR, opacity=0.7, row=3, col=1, annotation_text="Overbought")
    fig.add_hline(y=20, line_dash="dash", line_color=ACCENT, opacity=0.7, row=3, col=1, annotation_text="Oversold")

    # Derivative chart (row 4)
    # First derivative
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stoch_derivative_smooth'],
        mode="lines",
        name="Stoch Derivative",
        line=dict(color="#FFA726", width=2),
        hovertemplate='Derivative: %{y:.3f}<extra></extra>'
    ), row=4, col=1)

    # Second derivative (acceleration)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stoch_second_derivative_smooth'],
        mode="lines",
        name="Stoch Acceleration",
        line=dict(color="#AB47BC", width=2, dash='dot'),
        hovertemplate='Acceleration: %{y:.3f}<extra></extra>'
    ), row=4, col=1)

    # Zero line for derivative
    fig.add_hline(y=0, line_dash="solid", line_color=MUTED, opacity=0.9, row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=TEXT, size=12),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=1000,
        barmode="overlay",
        hovermode='x unified',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            bgcolor='rgba(11,18,32,0.7)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        ),
        margin=dict(l=40, r=24, t=60, b=40),
        transition={'duration': 400, 'easing': 'cubic-in-out'}
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=0.6, linecolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showline=False)
    
    # Update subplot titles
    fig.update_annotations(font_size=16, font_color=ACCENT)

    st.markdown("<div class='streamlit-card chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Add some summary information
    with st.expander("📈 View Data Summary"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candles", len(data))
            st.metric("Average Volume", f"{data['Volume'].mean():.2f}")
            
        with col2:
            st.metric("Highest Price", f"{data['High'].max():.2f}")
            st.metric("Lowest Price", f"{data['Low'].min():.2f}")
            
        with col3:
            st.metric("Price Change", f"{(data['Close'].iloc[-1] - data['Open'].iloc[0]):.2f}")
            st.metric("Change %", f"{((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0] * 100):.2f}%")
        
        # Stochastic summary
        st.markdown("---")
        st.subheader("Stochastic Oscillator Summary")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            current_k = data['stoch_k'].iloc[-1] if not pd.isna(data['stoch_k'].iloc[-1]) else 0
            current_d = data['stoch_d'].iloc[-1] if not pd.isna(data['stoch_d'].iloc[-1]) else 0
            st.metric("Current %K", f"{current_k:.2f}")
            st.metric("Current %D", f"{current_d:.2f}")
            
        with col5:
            stoch_signal = "Overbought" if current_k > 80 else ("Oversold" if current_k < 20 else "Neutral")
            stoch_color = ERROR if current_k > 80 else (ACCENT if current_k < 20 else MUTED)
            st.markdown(f"**Signal:** <span style='color:{stoch_color}'>{stoch_signal}</span>", unsafe_allow_html=True)
            
            # Cross signal
            if not pd.isna(data['stoch_k'].iloc[-1]) and not pd.isna(data['stoch_k'].iloc[-2]):
                if data['stoch_k'].iloc[-1] > data['stoch_d'].iloc[-1] and data['stoch_k'].iloc[-2] <= data['stoch_d'].iloc[-2]:
                    st.markdown("**Cross:** <span style='color:#6EE7B7'>Bullish (K above D)</span>", unsafe_allow_html=True)
                elif data['stoch_k'].iloc[-1] < data['stoch_d'].iloc[-1] and data['stoch_k'].iloc[-2] >= data['stoch_d'].iloc[-2]:
                    st.markdown("**Cross:** <span style='color:#FB7185'>Bearish (K below D)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Cross:** No significant cross")
                    
        with col6:
            st.markdown("**Stochastic Ranges:**")
            st.markdown("- **Oversold:** < 20")
            st.markdown("- **Neutral:** 20-80")
            st.markdown("- **Overbought:** > 80")
        
        # Derivative summary
        st.markdown("---")
        st.subheader("Stochastic Derivative Analysis")
        col7, col8, col9 = st.columns(3)

        with col7:
            st.metric("Current Derivative", f"{current_derivative:.3f}")
            st.metric("Derivative MA5", f"{data['stoch_derivative_smooth'].tail(5).mean():.3f}")
            
        with col8:
            deriv_signal = "Increasing" if current_derivative > 0.1 else ("Decreasing" if current_derivative < -0.1 else "Stable")
            deriv_color = ACCENT if current_derivative > 0.1 else (ERROR if current_derivative < -0.1 else MUTED)
            st.markdown(f"**Derivative Signal:** <span style='color:{deriv_color}'>{deriv_signal}</span>", unsafe_allow_html=True)
            
            accel_signal = "Accelerating" if current_second_deriv > 0.05 else ("Decelerating" if current_second_deriv < -0.05 else "Constant")
            accel_color = ACCENT if current_second_deriv > 0.05 else (ERROR if current_second_deriv < -0.05 else MUTED)
            st.markdown(f"**Acceleration:** <span style='color:{accel_color}'>{accel_signal}</span>", unsafe_allow_html=True)

        with col9:
            st.markdown("**Derivative Interpretation:**")
            if current_derivative > 0.2 and current_second_deriv > 0.05:
                st.markdown("📈 **Strong Bullish Momentum**")
            elif current_derivative > 0.2 and current_second_deriv < -0.05:
                st.markdown("⚠️ **Bullish but Slowing**")
            elif current_derivative < -0.2 and current_second_deriv < -0.05:
                st.markdown("📉 **Strong Bearish Momentum**")
            elif current_derivative < -0.2 and current_second_deriv > 0.05:
                st.markdown("⚠️ **Bearish but Slowing**")
            else:
                st.markdown("➡️ **Neutral Momentum**")

# End of file

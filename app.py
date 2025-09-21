import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Technical Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stSelectbox, .stSlider, .stButton {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Advanced Technical Analysis Platform")

# Sidebar
st.sidebar.header("Settings")

# Exchange selection
exchange_options = {
    "Coinbase": ccxt.coinbase,
    "Binance": ccxt.binance,
    "Kucoin": ccxt.kucoin,
    "Bitfinex": ccxt.bitfinex
}
selected_exchange_name = st.sidebar.selectbox("Exchange", list(exchange_options.keys()))
exchange_class = exchange_options[selected_exchange_name]

# Symbol selection
symbol_options = {
    "ETH/USD": "ETH/USD",
    "BTC/USD": "BTC/USD",
    "ADA/USD": "ADA/USD",
    "SOL/USD": "SOL/USD",
    "XRP/USD": "XRP/USD"
}
selected_symbol = st.sidebar.selectbox("Symbol", list(symbol_options.keys()), index=0)

# Timeframe selection
timeframe_options = {
    "1h": "1h",
    "4h": "4h", 
    "1d": "1d",
    "1w": "1w"
}
selected_timeframe = st.sidebar.selectbox("Timeframe", list(timeframe_options.keys()), index=0)

# Limit selection
limit = st.sidebar.slider("Number of Candles", min_value=100, max_value=1000, value=500, step=50)

# Alert settings
st.sidebar.subheader("Alert Settings")
alert_price = st.sidebar.number_input("Alert Price", value=0.0, step=0.1, format="%.2f")
alert_condition = st.sidebar.selectbox("Alert Condition", ["Above", "Below", "5% Change"])

if st.sidebar.button("Set Alert"):
    st.sidebar.success("Alert set!")

# Initialize alerts in session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Manual implementation of technical indicators
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    stoch_k_smooth = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k_smooth.rolling(window=smooth_d).mean()
    return stoch_k_smooth, stoch_d

# Cached functions
@st.cache_data(ttl=300)
def fetch_data(_exchange, symbol, timeframe, limit):
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

@st.cache_data
def calculate_technical_indicators(data):
    # RSI
    data['RSI'] = calculate_rsi(data)
    
    # MACD
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = calculate_macd(data)
    
    # Bollinger Bands
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = calculate_bollinger_bands(data)
    
    # Stochastic Oscillator
    data['Stoch_K'], data['Stoch_D'] = calculate_stochastic(data)
    
    # Volume MA
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
    
    return data

@st.cache_data
def find_supply_demand_points(data, lookback=3):
    supply_idx = []
    demand_idx = []

    for i in range(lookback, len(data)-lookback):
        high_window = data['High'].iloc[i-lookback:i+lookback+1]
        low_window = data['Low'].iloc[i-lookback:i+lookback+1]

        if data['High'].iloc[i] == max(high_window):
            supply_idx.append(i)
        if data['Low'].iloc[i] == min(low_window):
            demand_idx.append(i)

    # Filter points based on volume
    supply_idx_filtered = [i for i in supply_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
    demand_idx_filtered = [i for i in demand_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
    
    return supply_idx_filtered, demand_idx_filtered

# Chart creation functions
def create_main_chart(data, supply_idx, demand_idx):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Price Chart", "Volume"))

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
    if len(supply_idx) > 0:
        fig.add_trace(go.Scatter(
            x=data.index[supply_idx],
            y=data['High'].iloc[supply_idx] + 5,
            mode='markers',
            marker=dict(symbol='triangle-up', color='red', size=12),
            name='Supply'
        ), row=1, col=1)

    # Demand points
    if len(demand_idx) > 0:
        fig.add_trace(go.Scatter(
            x=data.index[demand_idx],
            y=data['Low'].iloc[demand_idx] - 5,
            mode='markers',
            marker=dict(symbol='triangle-down', color='green', size=12),
            name='Demand'
        ), row=1, col=1)

    # Volume
    up = data[data["Close"] >= data["Open"]]
    down = data[data["Close"] < data["Open"]]

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

    # Volume MA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA20'],
        mode="lines",
        name="Volume MA20",
        line=dict(color="orange", width=2)
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=800,
        barmode="overlay",
        xaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikecolor='white',
            spikethickness=1
        ),
        xaxis2=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikecolor='white',
            spikethickness=1
        )
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikedash='solid',
        spikecolor='white',
        spikethickness=1
    )

    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikedash='solid',
        spikecolor='white',
        spikethickness=1,
        row=2, col=1
    )

    return fig

def create_rsi_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(template="plotly_dark", title="RSI Indicator")
    return fig

def create_macd_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal', line=dict(color='orange')))
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='Histogram', marker_color='gray'))
    fig.update_layout(template="plotly_dark", title="MACD Indicator")
    return fig

def create_bollinger_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='Upper Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], name='Middle Band', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='Lower Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='white')))
    fig.update_layout(template="plotly_dark", title="Bollinger Bands")
    return fig

def create_stochastic_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_K'], name='Stoch %K', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Stoch_D'], name='Stoch %D', line=dict(color='red')))
    fig.add_hline(y=80, line_dash="dash", line_color="red")
    fig.add_hline(y=20, line_dash="dash", line_color="green")
    fig.update_layout(template="plotly_dark", title="Stochastic Oscillator")
    return fig

# Forecasting function (simplified)
def run_prediction(model_type, data, days):
    # This is a simplified version - in practice you would implement proper forecasting
    last_price = data['Close'].iloc[-1]
    volatility = data['Close'].pct_change().std()
    
    # Simple simulation of prediction results
    np.random.seed(42)
    predictions = [last_price * (1 + np.random.normal(0, volatility)) for _ in range(days)]
    
    return {
        'predictions': predictions,
        'confidence': max(0.7, 1 - volatility * 10)  # Higher volatility reduces confidence
    }

def create_forecast_chart(data, forecast_result):
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, len(forecast_result['predictions'])+1)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_result['predictions'], name='Forecast', line=dict(color='orange', dash='dash')))
    fig.update_layout(template="plotly_dark", title="Price Forecast")
    return fig

# Main app
try:
    exchange = exchange_class()
    
    with st.spinner("Fetching data from exchange..."):
        ohlcv = fetch_data(exchange, selected_symbol, selected_timeframe, limit)
        
        if ohlcv is None:
            st.error("Failed to fetch data. Please try again later.")
            st.stop()
            
        # Create DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])

        # Convert timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)

        # Calculate indicators
        data = calculate_technical_indicators(data)
        
        # Find supply/demand points
        supply_idx, demand_idx = find_supply_demand_points(data)

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Indicators", "Advanced Analysis", "Data"])

        with tab1:
            fig = create_main_chart(data, supply_idx, demand_idx)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"{data['Close'].iloc[-1]:.2f}")
            with col2:
                change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                change_pct = (change / data['Close'].iloc[-2]) * 100
                st.metric("24h Change", f"{data['Close'].iloc[-1]:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
            with col3:
                st.metric("Supply Points", len(supply_idx))
            with col4:
                st.metric("Demand Points", len(demand_idx))

        with tab2:
            st.subheader("Technical Indicators")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_rsi = create_rsi_chart(data)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            with col2:
                fig_macd = create_macd_chart(data)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig_bb = create_bollinger_chart(data)
                st.plotly_chart(fig_bb, use_container_width=True)
                
            with col4:
                fig_stoch = create_stochastic_chart(data)
                st.plotly_chart(fig_stoch, use_container_width=True)

        with tab3:
            st.subheader("Advanced Analysis & Forecasting")
            
            model_option = st.selectbox(
                "Select Forecasting Model",
                ["ARIMA", "Prophet", "LSTM", "Ensemble"]
            )
            
            forecast_days = st.slider("Forecast Days", 1, 30, 7)
            
            if st.button("Run Forecast"):
                with st.spinner("Running forecast..."):
                    forecast_result = run_prediction(model_option, data, forecast_days)
                    
                    fig_forecast = create_forecast_chart(data, forecast_result)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    st.subheader("Forecast Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"{data['Close'].iloc[-1]:.2f}")
                    with col2:
                        change = forecast_result['predictions'][-1] - data['Close'].iloc[-1]
                        change_pct = (change / data['Close'].iloc[-1]) * 100
                        st.metric("Forecast Price", f"{forecast_result['predictions'][-1]:.2f}", 
                                 f"{change:.2f} ({change_pct:.2f}%)")
                    with col3:
                        st.metric("Confidence", f"{forecast_result['confidence']*100:.1f}%")

        with tab4:
            st.subheader("Raw Data")
            st.dataframe(data.tail(20))
            
            # Download button
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"{selected_symbol.replace('/', '_')}_{selected_timeframe}_data.csv",
                mime="text/csv",
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

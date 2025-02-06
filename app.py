import streamlit as st
import pandas as pd
import os
import vectorbt as vbt

from modules.data_loader import load_ticker_data
from modules.analytics import compute_moving_averages, generate_signals
from modules.dashboard import (render_portfolio_tab, render_price_chart, render_risk_metrics,
                               render_advanced_metrics, render_ml_strategy)

# Set page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="Hedges Backtesting Dashboard")

st.title("Hedges Backtesting Dashboard")
st.subheader("Test trading strategies against historical data", divider="rainbow")

# Sidebar configuration
st.sidebar.header("Configuration")
selected_tickers = st.sidebar.multiselect("Ticker(s)", options=[f.replace(".csv", "") for f in os.listdir("data") if f.endswith(".csv")], default=["NVDA", "TSLA"])
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-31"))
fast_window = st.sidebar.slider("Fast MA Window", 5, 50, 10)
slow_window = st.sidebar.slider("Slow MA Window", 10, 200, 50)

with st.sidebar.expander("Moving Average Options"):
    st.html("""
    <div style="font-size:13px;">
        <span style="font-weight:600;color:#333;">Fast MA Window:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Number of days to compute the fast moving average.</li>
        <li>Shorter windows can react faster to price changes but may lead to more false signals.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Slow MA Window:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Number of days to compute the slow moving average.</li>
        <li>Longer windows can smooth out price fluctuations but may lag behind the trend.</li>
        </ul>
    </div>
    """)

# Validate input
if not selected_tickers:
    st.info("Please select at least one ticker.")
    st.stop()

if fast_window >= slow_window:
    st.sidebar.error("Fast MA window must be less than Slow MA window.")
    st.stop()

# Load data
data = pd.DataFrame()
data_folder = "data"
for t in selected_tickers:
    series = load_ticker_data(t, data_folder, start_date, end_date)
    if series.empty:
        st.error(f"No data for {t}.")
        st.stop()
    data[t] = series
if data.empty:
    st.error("No data found for the specified tickers and date range.")
    st.stop()

# Compute analytics, portfolio, etc.
fast_ma, slow_ma = compute_moving_averages(data, fast_window, slow_window)
entries, exits = generate_signals(fast_ma, slow_ma)
portfolio = vbt.Portfolio.from_signals(data, entries, exits, init_cash=10000, freq="1D")

# Set chosen ticker and its price series for later use
chosen_ticker = selected_tickers[0]
price_series = data[chosen_ticker]

# Create top-level tabs to streamline main area UX
main_tabs = st.tabs(["Portfolio", "Price & Trends", "Advanced Metrics", "ML Strategy", "Risk Metrics"])

with main_tabs[0]:
    # Portfolio section
    render_portfolio_tab(portfolio, data)

with main_tabs[1]:
    # Price & Moving Averages section
    render_price_chart(data, fast_ma, slow_ma, selected_tickers, fast_window, slow_window)

with main_tabs[2]:
    # Advanced Metrics section
    render_advanced_metrics(chosen_ticker, price_series, portfolio)

with main_tabs[3]:
    render_ml_strategy(data, chosen_ticker)

with main_tabs[4]:
    # Risk Metrics section
    render_risk_metrics(portfolio, chosen_ticker)

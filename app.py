import streamlit as st
import pandas as pd
import os
import vectorbt as vbt

from modules.data_loader import load_ticker_data
from modules.analytics import compute_moving_averages, generate_signals
from modules.ml_strategy import create_ml_dataset, split_dataset, train_and_predict, compute_rmse
from modules.dashboard import (render_portfolio_tab, render_price_chart,
                               render_advanced_metrics, render_ml_results)

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

# CONFIGURATION HELP

with st.sidebar.expander("Moving Average Parameters"):
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

st.sidebar.divider()

# Add ML strategy options
st.sidebar.subheader("ML Strategy Options")
ml_model = st.sidebar.selectbox(
    "Select ML Model",
    options=["Linear Regression", "Random Forest", "XGBoost", "Neural Network"],
    index=0
)
feature_window = st.sidebar.slider("Feature Window (days)", min_value=5, max_value=60, value=20)
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=5)
train_split = st.sidebar.slider("Training Split (%)", min_value=50, max_value=90, value=70)

# ML STRATEGY HELP

with st.sidebar.expander("ML Models"):
    st.html("""
    <div style="font-size:13px;">
        <span style="font-weight:600;color:#333;">Linear Regression:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Simple linear model that fits a line to the data.</li>
        <li>Interpretable but may underfit complex patterns.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Random Forest:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Ensemble model that averages multiple decision trees.</li>
        <li>Less interpretable but can capture complex patterns.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">XGBoost:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Gradient boosting model that optimizes a set of decision trees.</li>
        <li>Highly accurate but may overfit with too many trees.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Neural Network:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Deep learning model with multiple layers of neurons.</li>
        <li>Highly flexible but requires large amounts of data and tuning.</li>
        </ul>
    </div>
    """)

with st.sidebar.expander("ML Strategy Parameters"):
    st.html("""
    <div style="font-size:13px;">
        <span style="font-weight:600;color:#333;">Feature Window:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Number of lagged features to create for the model.</li>
        <li>More features can capture more information but may lead to overfitting.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Prediction Horizon:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Number of days ahead to predict the target price.</li>
        <li>Shorter horizons may be more accurate but less profitable.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Training Split:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Percentage of data to use for training the model.</li>
        <li>More training data can improve model performance but may lead to overfitting.</li>
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
main_tabs = st.tabs(["Portfolio", "Price & Trends", "Advanced Metrics", "ML Strategy"])

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
    # ML Strategy Results section
    ml_series = data[chosen_ticker]
    df_ml = create_ml_dataset(ml_series, feature_window, prediction_horizon)
    X_train, y_train, X_test, y_test = split_dataset(df_ml, train_split)
    y_pred, _ = train_and_predict(ml_model, X_train, y_train, X_test)
    rmse = compute_rmse(y_test, y_pred)
    render_ml_results(ml_model, rmse, y_test, y_pred)

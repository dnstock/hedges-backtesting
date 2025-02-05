import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go
import os

# Set page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="Hedges Backtesting Dashboard")

st.title("Hedges Backtesting Dashboard")
st.subheader("Test trading strategies against historical data", divider="rainbow")

# Use caching while loading CSV data
@st.cache_data
def load_ticker_data(ticker, data_folder, start_date, end_date):
    file_path = os.path.join(data_folder, f"{ticker}.csv")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df.loc[start_date:end_date]
        return df["close"]
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.Series()

# Get available tickers from data folder
data_folder = "data"
available_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
available_tickers = [f.replace(".csv", "") for f in available_files]

st.sidebar.header("Configuration")
# Multi-select from available tickers
selected_tickers = st.sidebar.multiselect("Ticker(s)", options=available_tickers, default=["NVDA", "TSLA"])
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-31"))
fast_window = st.sidebar.slider("Fast MA Window", min_value=5, max_value=50, value=10)
slow_window = st.sidebar.slider("Slow MA Window", min_value=10, max_value=200, value=50)

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

# Validate input
if not selected_tickers:
    st.info("Please select at least one ticker.")
    st.stop()

if fast_window >= slow_window:
    st.sidebar.error("Fast MA window must be less than Slow MA window.")
    st.stop()

# Load price data from CSV files using cached function
data = pd.DataFrame()
for t in selected_tickers:
    series = load_ticker_data(t, data_folder, start_date, end_date)
    if series.empty:
        st.error(f"No data found for ticker {t}. Please run get_data.py to fetch the data.")
        st.stop()
    data[t] = series

if data.empty:
    st.error("No data found for the specified tickers and date range.")
    st.stop()

# Compute moving averages
fast_ma = data.rolling(fast_window).mean()
slow_ma = data.rolling(slow_window).mean()

# Generate signals: enter when fast MA > slow MA, exit when fast MA < slow MA.
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma

# Backtest using vectorbt
portfolio = vbt.Portfolio.from_signals(data, entries, exits, init_cash=10000, freq="1D")

# Instead of st.columns, use tabs for Performance and Value
tabs_perf_value = st.tabs(["Performance", "Value"])

with tabs_perf_value[0]:
    st.header("Portfolio Performance")
    with st.expander("Show Metrics"):
        stats_df = portfolio.stats()
        if isinstance(stats_df, pd.Series):
            stats_df = stats_df.to_frame().T
        for col in stats_df.columns:
            if pd.api.types.is_timedelta64_dtype(stats_df[col]):
                stats_df[col] = stats_df[col].astype(str)
        st.dataframe(stats_df)

with tabs_perf_value[1]:
    st.header("Portfolio Value")
    portfolio_value = portfolio.value() if callable(portfolio.value) else portfolio.value
    fig_value = go.Figure()
    fig_value.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value,
        mode="lines",
        name="Portfolio Value"
    ))
    fig_value.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    st.plotly_chart(fig_value)

# Plot price along with moving averages for each ticker in separate tabs
st.header("Price & Moving Averages")
tabs = st.tabs(selected_tickers)
for i, t in enumerate(selected_tickers):
    with tabs[i]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[t], mode="lines", name=f"{t} Price"))
        fig.add_trace(go.Scatter(x=fast_ma.index, y=fast_ma[t], mode="lines", name=f"Fast MA ({fast_window})"))
        fig.add_trace(go.Scatter(x=slow_ma.index, y=slow_ma[t], mode="lines", name=f"Slow MA ({slow_window})"))
        fig.update_layout(title=f"{t}: Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

# New main area section for Advanced Metrics
st.header("Advanced Metrics")
# Automatically use the first selected ticker
chosen_ticker = selected_tickers[0]
price_series = data[chosen_ticker]
adv_tabs_main = st.tabs(["Bollinger Bands", "Drawdowns"])

with adv_tabs_main[0]:
    # Compute and display Bollinger Bands (20-day SMA with 2 std dev)
    sma = price_series.rolling(window=20).mean()
    std = price_series.rolling(window=20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=price_series.index, y=price_series, mode="lines", name=f"{chosen_ticker} Price"))
    fig_bb.add_trace(go.Scatter(x=price_series.index, y=sma, mode="lines", name="SMA 20"))
    fig_bb.add_trace(go.Scatter(x=price_series.index, y=upper, mode="lines", name="Upper Band"))
    fig_bb.add_trace(go.Scatter(x=price_series.index, y=lower, mode="lines", name="Lower Band"))
    fig_bb.update_layout(title=f"{chosen_ticker} Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_bb)

with adv_tabs_main[1]:
    # Display drawdowns from the portfolio for the chosen ticker
    dd = portfolio.drawdowns
    dd_df = dd.records
    st.subheader(f"{chosen_ticker} Drawdowns")
    st.dataframe(dd_df)
    drawdown_col = next((col for col in dd_df.columns if "drawdown" in col.lower()), None)
    if drawdown_col is None:
        st.error("No drawdown column found in the drawdowns data.")
    else:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Bar(
            x=dd_df.index,
            y=dd_df[drawdown_col],
            name="Drawdown"
        ))
        fig_dd.update_layout(title=f"{chosen_ticker} Drawdowns", xaxis_title="Index", yaxis_title=drawdown_col.title())
        st.plotly_chart(fig_dd)

# New main area section for ML Strategy Results
st.header("ML Strategy Results")
st.write("Selected Model:", ml_model)

import numpy as np
# Use the price series of the first selected ticker for ML predictions.
ml_series = data[selected_tickers[0]]

# Create lag features as predictors
lags = [ml_series.shift(i) for i in range(1, feature_window+1)]
X = pd.concat(lags, axis=1)
X.columns = [f"lag_{i}" for i in range(1, feature_window+1)]
# Create target: price after prediction_horizon days
y = ml_series.shift(-prediction_horizon)
df_ml = pd.concat([X, y.rename("target")], axis=1).dropna()

# Split data into training and testing sets
train_size = int(len(df_ml) * (train_split / 100))
train = df_ml.iloc[:train_size]
test = df_ml.iloc[train_size:]
X_train = train.drop("target", axis=1)
y_train = train["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

# Instantiate the selected ML model
if ml_model == "Linear Regression":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
elif ml_model == "Random Forest":
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif ml_model == "XGBoost":
    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=42)
elif ml_model == "Neural Network":
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)

# Train the model and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute RMSE as a performance metric
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"RMSE: {rmse:.2f}")

# Plot actual versus predicted prices
fig_ml = go.Figure()
fig_ml.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test,
    mode="lines",
    name="Actual"
))
fig_ml.add_trace(go.Scatter(
    x=y_test.index,
    y=y_pred,
    mode="lines",
    name="Predicted"
))
fig_ml.update_layout(
    title=f"{ml_model} Predictions (RMSE: {rmse:.2f})",
    xaxis_title="Date",
    yaxis_title="Price"
)
st.plotly_chart(fig_ml)

# Option to show raw data
if st.checkbox("Show Raw Data"):
    with st.expander("Price Data"):
        st.dataframe(data)

st.sidebar.divider()

with st.sidebar.expander("Strategy Parameters"):
    st.html("""
    <div style="font-size:13px;">
        <span style="font-weight:600;color:#333;">Fast MA Window:</span><br/>
        <ul style="font-size:13px;margin-left:16px;">
        <li>Uses fewer periods, so it reacts quickly to recent price changes.</li>
        <li>A smaller window produces more signals (both true and false), which can lead to more trades but also more noise and potential whipsaws.</li>
        </ul>
        
        <span style="font-weight:600;color:#333;">Slow MA Window:</span><br/>
        <ul style="font-size:13px;margin-left:16px;margin-bottom:0;">
        <li>Uses more periods, making it less sensitive and smoother.</li>
        <li>A larger window reduces noise and false signals, but may delay entry and exit signals.</li>
        </ul>
    </div>
    """)

with st.sidebar.expander("Impact on Performance"):
    st.markdown("""
    <span style="font-size:13px;">
    
    **Smaller Fast MA / Larger Slow MA:**  
    Can generate clear crossover signals but might lead to delayed exits or entries.

    **Smaller Both:**  
    May result in more frequent trading and increased transaction costs due to noise.

    **Larger Both:**  
    Results in fewer signals, potentially missing short-term opportunities but reducing whipsaws.
    
    </span>
    """, unsafe_allow_html=True)

with st.sidebar.expander("To Summarize"):
    st.write("""
    <span style="font-size:13px;">
    
    **Fast MA Window:** Determines how quickly the strategy responds to price changes.
    
    **Slow MA Window:** Affects the smoothness of the moving average and the lag in signal generation.

    **In essence:** Adjusting these parameters changes how quickly the strategy responds to market moves, affecting trade frequency, potential profits, and overall risk. Testing different settings helps to balance sensitivity with reliability.
    
    </span>
    """, unsafe_allow_html=True)

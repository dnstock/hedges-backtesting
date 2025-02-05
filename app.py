import streamlit as st
import vectorbt as vbt
import pandas as pd
import plotly.graph_objects as go
import os

st.title("Backtesting Dashboard for Trading Strategies")

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

# st.sidebar.markdown("""
# **How These Windows Work:**

# - **Fast MA Window:**
#   The number of periods used for the fast moving average. Smaller values respond quickly to recent price changes, generating more signals (with the potential for increased noise and false signals).

# - **Slow MA Window:**
#   The number of periods used for the slow moving average. Larger values smooth out price fluctuations and reduce noise, but may delay signal generation.

# Adjust these parameters to balance responsiveness and reliability.
# """)

# Validate input
if not selected_tickers:
    st.info("Please select at least one ticker.")
    st.stop()

if fast_window >= slow_window:
    st.sidebar.error("Fast MA window must be less than Slow MA window.")
    st.stop()

# Load price data from CSV files
data_folder = "data"
tickers = ticker.split()
data = pd.DataFrame()

for t in tickers:
    file_path = os.path.join(data_folder, f"{t}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = df.loc[start_date:end_date]
        data[t] = df["close"]
    else:
        st.error(f"No data found for ticker {t}. Please run get_data.py to fetch the data.")
        st.stop()

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

# Display portfolio performance metrics
st.header("Portfolio Performance")
stats_df = portfolio.stats()
if isinstance(stats_df, pd.Series):
    stats_df = stats_df.to_frame().T
# Convert timedelta columns to strings to avoid serialization errors
for col in stats_df.columns:
    if pd.api.types.is_timedelta64_dtype(stats_df[col]):
        stats_df[col] = stats_df[col].astype(str)
st.write(stats_df)

# Plot portfolio value over time using a manual plot
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

# Plot price along with moving averages
st.header("Price & Moving Averages")
fig = go.Figure()
for t in tickers:
    fig.add_trace(go.Scatter(x=data.index, y=data[t], mode="lines", name=f"{t} Price"))
    fig.add_trace(go.Scatter(x=fast_ma.index, y=fast_ma[t], mode="lines", name=f"{t} Fast MA ({fast_window})"))
    fig.add_trace(go.Scatter(x=slow_ma.index, y=slow_ma[t], mode="lines", name=f"{t} Slow MA ({slow_window})"))
fig.update_layout(title="Price and MAs", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# Option to show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Price Data")
    st.write(data)

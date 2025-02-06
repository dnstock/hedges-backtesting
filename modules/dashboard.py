import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_portfolio_tab(portfolio, data):
    st.header("My Portfolio")
    portfolio_tabs = st.tabs(["Value", "Performance", "Raw Data"])
    with portfolio_tabs[0]:
        render_value_tab(portfolio)
    with portfolio_tabs[1]:
        render_performance_tab(portfolio)
    with portfolio_tabs[2]:
        st.dataframe(data)

def render_performance_tab(portfolio):
    stats_df = portfolio.stats()
    if isinstance(stats_df, pd.Series):
        stats_df = stats_df.to_frame().T
    for col in stats_df.columns:
        if pd.api.types.is_timedelta64_dtype(stats_df[col]):
            stats_df[col] = stats_df[col].astype(str)
    st.dataframe(stats_df)

def render_value_tab(portfolio):
    portfolio_value = portfolio.value() if callable(portfolio.value) else portfolio.value
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode="lines", name="Portfolio Value"))
    fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

def render_price_chart(data, fast_ma, slow_ma, tickers, fast_window, slow_window):
    st.header("Price & Moving Averages")
    tabs = st.tabs(tickers)
    for i, t in enumerate(tickers):
        with tabs[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[t], mode="lines", name=f"{t} Price"))
            fig.add_trace(go.Scatter(x=fast_ma.index, y=fast_ma[t], mode="lines", name=f"Fast MA ({fast_window})"))
            fig.add_trace(go.Scatter(x=slow_ma.index, y=slow_ma[t], mode="lines", name=f"Slow MA ({slow_window})"))
            fig.update_layout(title=f"{t}: Price and Moving Averages", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

def render_advanced_metrics(chosen_ticker, price_series, portfolio):
    st.header("Advanced Metrics")
    adv_tabs = st.tabs(["Bollinger Bands", "Drawdowns"])
    with adv_tabs[0]:
        sma = price_series.rolling(window=20).mean()
        std = price_series.rolling(window=20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode="lines", name=f"{chosen_ticker} Price"))
        fig.add_trace(go.Scatter(x=price_series.index, y=sma, mode="lines", name="SMA 20"))
        fig.add_trace(go.Scatter(x=price_series.index, y=upper, mode="lines", name="Upper Band"))
        fig.add_trace(go.Scatter(x=price_series.index, y=lower, mode="lines", name="Lower Band"))
        fig.update_layout(title=f"{chosen_ticker} Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    with adv_tabs[1]:
        dd = portfolio.drawdowns
        dd_df = dd.records
        st.subheader(f"{chosen_ticker} Drawdowns")
        st.dataframe(dd_df)
        drawdown_col = next((col for col in dd_df.columns if "drawdown" in col.lower()), None)
        if drawdown_col is None:
            st.error("No drawdown column found in the drawdowns data.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dd_df.index, y=dd_df[drawdown_col], name="Drawdown"))
            fig.update_layout(title=f"{chosen_ticker} Drawdowns", xaxis_title="Index", yaxis_title=drawdown_col.title())
            st.plotly_chart(fig)

def render_risk_metrics(portfolio, chosen_ticker):
    st.header("Risk Metrics")
    risk_tabs = st.tabs(["Drawdowns", "Raw Statistics"])
    with risk_tabs[0]:
        st.write("Drawdown Plot:")
        fig = portfolio.drawdowns.plot(column=chosen_ticker)
        st.plotly_chart(fig)
    with risk_tabs[1]:
        risk_stats = portfolio.stats()
        st.table(risk_stats)

def render_ml_strategy(data, chosen_ticker):
    from modules.ml_strategy import create_ml_dataset, split_dataset, train_and_predict, compute_rmse
    st.header("ML Strategy")

    row1col1, row1col2 = st.columns(2)
    with row1col1:
        ml_model = st.selectbox(
            "Select ML Model",
            options=["Linear Regression", "Random Forest", "XGBoost", "Neural Network"],
            index=0
        )
    with row1col2:
        feature_window = st.slider("Feature Window (days)", min_value=5, max_value=60, value=20)

    row2col1, row2col2 = st.columns(2)
    with row2col1:
        train_split = st.slider("Training Split (%)", min_value=50, max_value=90, value=70)
    with row2col2:
        prediction_horizon = st.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=5)

    row3col1, row3col2 = st.columns(2)
    with row3col1:
        with st.expander("Model Options"):
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
    with row3col2:
        with st.expander("Parameter Options"):
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

    ml_series = data[chosen_ticker]
    df_ml = create_ml_dataset(ml_series, feature_window, prediction_horizon)
    X_train, y_train, X_test, y_test = split_dataset(df_ml, train_split)
    y_pred, _ = train_and_predict(ml_model, X_train, y_train, X_test)
    rmse = compute_rmse(y_test, y_pred)

    # Display the results
    st.subheader("Results")
    st.write("Selected Model:", ml_model)
    st.write(f"RMSE: {rmse:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(title=f"{ml_model} Predictions (RMSE: {rmse:.2f})", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

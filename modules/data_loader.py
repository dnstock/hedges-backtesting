import os
import pandas as pd
import streamlit as st

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

"""
Usage example: 
python get_data.py --tickers NVDA TSLA --start 2024-01-01 --end 2024-12-31 --output data

This will fetch data for Nvidia and Tesla from January 1, 2024 to December 31, 2024 
and save to a folder named data.
"""
import os
import requests
import pandas as pd
import argparse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("Please set the POLYGON_API_KEY environment variable.")

def fetch_stock_data(ticker, start_date, end_date, timespan="minute", multiplier=1):
    """
    Fetches aggregated OHLCV data from Polygon.io for a given ticker and date range.
    Uses pagination if more than 50,000 records are returned.
    """
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
        f"{multiplier}/{timespan}/{start_date}/{end_date}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    results = []
    url = base_url
    with requests.Session() as session:
        while True:
            resp = session.get(url, params=params)
            data = resp.json()
            if "results" in data:
                results.extend(data["results"])
            else:
                print(f"Error fetching data for {ticker}: {data}")
                break
            if data.get("next_url"):
                # next_url already includes all parameters including apiKey.
                url = data["next_url"]
                params = {}  # clear params for subsequent requests
            else:
                break

    if results:
        df = pd.DataFrame(results)
        # Convert the timestamp (in ms) to datetime and set as index
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date")
        # Select and rename columns to match vectorbt's expectations
        df = df[["o", "h", "l", "c", "v"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    else:
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical US stock data from Polygon.io and prepare for vectorbt"
    )
    parser.add_argument(
        "--tickers", type=str, nargs="+", required=True, help="Ticker symbols (e.g., AAPL MSFT)"
    )
    parser.add_argument(
        "--start", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Output folder to store CSV files"
    )
    # New arguments to optimize data retrieval for ML strategies
    parser.add_argument(
        "--timespan", type=str, default="minute", help="Data timespan (e.g., minute, hour, day)"
    )
    parser.add_argument(
        "--multiplier", type=int, default=1, help="Multiplier for timespan aggregation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for ticker in args.tickers:
        print(f"Fetching data for {ticker}...")
        df = fetch_stock_data(ticker, args.start, args.end, args.timespan, args.multiplier)
        if df.empty:
            print(f"No data returned for {ticker}.")
            continue
        file_path = os.path.join(args.output, f"{ticker}.csv")
        df.to_csv(file_path)
        print(f"Saved data for {ticker} to {file_path}")

if __name__ == "__main__":
    main()

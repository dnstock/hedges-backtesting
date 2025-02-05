# hedges-backtesting
Vector-based backtesting for AI-driven investment engine.

## Installation
```bash
pip install -r requirements.txt
```

## Get Historical Data
This will fetch data for Nvidia and Tesla from January 1, 2024 to December 31, 2024 and save to a folder named `/data`:
```bash
python get_data.py --tickers NVDA TSLA --start 2024-01-01 --end 2024-12-31 --output data
```

## Run the Web App
```bash
streamlit run app.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
```

import pandas as pd

def compute_moving_averages(data, fast_window, slow_window):
    fast_ma = data.rolling(fast_window).mean()
    slow_ma = data.rolling(slow_window).mean()
    return fast_ma, slow_ma

def generate_signals(fast_ma, slow_ma):
    entries = fast_ma > slow_ma
    exits = fast_ma < slow_ma
    return entries, exits

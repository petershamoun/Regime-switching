import pandas as pd
import yfinance as yf
import yaml
from pathlib import Path

def load_config():
    """Load parameters from config/config.yaml"""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_prices():
    """Download and prepare price data from Yahoo Finance"""
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]

    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    data = data.dropna()
    data = data.pct_change().dropna()
    return data

if __name__ == "__main__":
    df = load_prices()
    print("Loaded data:")
    print(df.tail())

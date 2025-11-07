from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yfinance as yf
import yaml


def default_config_path() -> Path:
    """Return the default location of the regime detection config."""
    return Path(__file__).resolve().parents[2] / "configs" / "regime_detection.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML configuration for the regime-detection stage."""
    cfg_path = Path(config_path) if config_path else default_config_path()
    with cfg_path.expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for the provided tickers."""
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise ValueError("No price data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data

    return prices.sort_index().ffill()


def load_prices(
    cfg: Dict[str, Any] | None = None,
    *,
    config_path: Path | str | None = None,
    returns: bool = True,
) -> pd.DataFrame:
    """Load prices (or returns) according to the configuration."""
    cfg = cfg or load_config(config_path)
    tickers = cfg["data"]["tickers"]
    start = cfg["data"]["start"]
    end = cfg["data"].get("end")

    prices = download_prices(tickers, start=start, end=end)
    prices = prices.dropna(how="all")
    if returns:
        return prices.pct_change().dropna()
    return prices


if __name__ == "__main__":
    df = load_prices(returns=True)
    print("Loaded returns:")
    print(df.tail())

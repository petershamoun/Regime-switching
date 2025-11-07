from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import yfinance as yf

SECTORS: List[str] = [
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLK",
    "XLI",
    "XLB",
    "XLV",
    "XLU",
    "XLRE",
    "XLC",
]

BENCH: List[str] = ["SPY", "IEF", "^VIX"]


def _ensure_iterable(obj: Iterable[str] | str) -> Sequence[str]:
    if isinstance(obj, str):
        return [obj]
    return list(obj)


def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download (or load cached) adjusted close prices for the given tickers.

    Args:
        tickers: Iterable of ticker symbols.
        start: Start date (inclusive).
        end: End date (exclusive). Defaults to None (use latest available).
        cache_path: Optional CSV path to cache the downloaded data.

    Returns:
        DataFrame of adjusted close prices indexed by date.
    """
    tickers_list = sorted(set(_ensure_iterable(tickers)))
    cache_file = Path(cache_path) if cache_path else None

    if cache_file and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached = cached.sort_index()
        # ensure coverage of requested tickers and start date
        if set(tickers_list).issubset(cached.columns):
            filtered = cached.loc[start:end] if end else cached.loc[start:]
            if not filtered.empty:
                return filtered[tickers_list]

    data = yf.download(
        tickers=tickers_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers_list}")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data["Adj Close"].to_frame(name=tickers_list[0]) if "Adj Close" in data else data

    prices = prices.sort_index().ffill()

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)

    return prices[tickers_list]


def load_all(
    start: str,
    end: str | None = None,
    tickers: Iterable[str] | None = None,
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    """Load all required prices with caching.

    Args:
        start: Start date.
        end: Optional end date.
        tickers: Optional explicit list of tickers.
        cache_dir: Directory for cached CSV files.

    Returns:
        DataFrame of adjusted close prices.
    """
    universe = sorted(set(_ensure_iterable(tickers) if tickers else SECTORS + BENCH))
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir_path / "prices.csv"
    prices = download_prices(universe, start=start, end=end, cache_path=cache_file)
    return prices[universe]

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

SECTORS = [
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
BENCH = ["SPY", "IEF", "^VIX"]


def _ensure_dataframe(data: pd.DataFrame | pd.Series, tickers: Iterable[str]) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        df = data.to_frame(name=tickers[0])
    else:
        df = data.copy()
    df = df.sort_index()
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Missing tickers in downloaded data: {missing}")
    return df.loc[:, tickers]


def download_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: Optional[str | pd.Timestamp] = None,
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Download (or load cached) adjusted close prices for the given tickers.
    """
    tickers = list(dict.fromkeys(tickers))
    cache_file: Optional[Path] = Path(cache_path) if cache_path else None

    if cache_file and cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached = cached.sort_index()
        if all(t in cached.columns for t in tickers):
            filtered = cached.loc[pd.IndexSlice[:], tickers]
            filtered = filtered[filtered.index >= pd.Timestamp(start)]
            if end:
                filtered = filtered[filtered.index <= pd.Timestamp(end)]
            if not filtered.empty:
                return filtered

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    prices = (
        data.rename_axis(index="date")
        .sort_index()
        .pipe(_ensure_dataframe, tickers=tickers)
        .dropna(how="all")
    )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)

    return prices


def load_all(
    start: str | pd.Timestamp,
    end: Optional[str | pd.Timestamp] = None,
    cache_path: Optional[str | Path] = None,
    extra_tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load all price data required for the strategy.
    """
    tickers = list(SECTORS)
    tickers.extend(BENCH)
    if extra_tickers:
        tickers.extend(list(extra_tickers))
    tickers = list(dict.fromkeys(tickers))
    prices = download_prices(tickers=tickers, start=start, end=end, cache_path=cache_path)
    return prices.sort_index()

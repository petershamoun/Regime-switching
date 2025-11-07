from __future__ import annotations

import pandas as pd


def trailing_return(prices: pd.DataFrame, months: int = 12, skip_last: int = 1) -> pd.DataFrame:
    """
    Compute trailing returns over a monthly horizon while skipping the most recent months.
    """
    if months <= 0:
        raise ValueError("months must be positive.")
    if skip_last < 0:
        raise ValueError("skip_last must be non-negative.")

    monthly_prices = prices.resample("M").last()
    shifted_prices = monthly_prices.shift(skip_last)
    trailing = shifted_prices / shifted_prices.shift(months) - 1.0
    return trailing.dropna(how="all")


def absolute_momentum(series: pd.Series, months: int = 12, threshold: float = 0.0) -> pd.Series:
    """
    Simple absolute momentum signal based on monthly data.
    """
    if months <= 0:
        raise ValueError("months must be positive.")

    monthly_series = series.resample("M").last()
    shifted_series = monthly_series.shift(1)
    perf = shifted_series / shifted_series.shift(months) - 1.0
    signal = (perf > threshold).astype(int)
    return signal.dropna()

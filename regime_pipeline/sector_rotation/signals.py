from __future__ import annotations

import pandas as pd


def trailing_return(prices: pd.DataFrame, months: int = 12, skip_last: int = 1) -> pd.DataFrame:
    """Compute trailing returns (e.g., 12-1 momentum) from daily prices.

    Args:
        prices: DataFrame of daily prices.
        months: Number of trailing months to include.
        skip_last: Number of most recent months to skip.

    Returns:
        Monthly DataFrame of trailing returns.
    """
    monthly = prices.resample("M").last()
    shifted = monthly.shift(skip_last)
    momentum = shifted / shifted.shift(months) - 1
    return momentum.dropna(how="all")


def absolute_momentum(
    series: pd.Series,
    months: int = 12,
    threshold: float = 0.0,
) -> pd.Series:
    """Absolute momentum filter for a single asset.

    Args:
        series: Daily price series.
        months: Lookback window in months.
        threshold: Minimum return to qualify as risk-on.

    Returns:
        Monthly Series of 0/1 signals.
    """
    monthly = series.resample("M").last()
    trailing = monthly.pct_change(periods=months)
    signal = (trailing > threshold).astype(int)
    return signal.dropna()

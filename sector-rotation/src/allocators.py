from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def top_k_equal(scores: pd.DataFrame, universe: Iterable[str], k: int = 4) -> pd.DataFrame:
    """Allocate equally across the top-k scoring assets per date.

    Args:
        scores: DataFrame of scores indexed by date.
        universe: Eligible tickers.
        k: Number of top assets to select.

    Returns:
        DataFrame of equal weights across selected assets.
    """
    universe_list: List[str] = list(universe)
    weights = []
    index = []

    for dt, row in scores.iterrows():
        row = row.reindex(universe_list)
        valid = row.dropna()
        alloc = pd.Series(0.0, index=universe_list)
        if not valid.empty:
            top = valid.nlargest(min(k, len(valid)))
            alloc.loc[top.index] = 1.0 / len(top)
        weights.append(alloc)
        index.append(dt)

    return pd.DataFrame(weights, index=index, columns=universe_list)


def inverse_vol_weights(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute inverse-volatility weights for the provided price history.

    Args:
        prices: DataFrame of daily prices.
        lookback: Lookback window in trading days for volatility estimation.

    Returns:
        Single-row DataFrame of weights indexed by the last available date.
    """
    if prices.empty:
        raise ValueError("Prices DataFrame is empty.")

    returns = prices.pct_change().dropna()
    window = returns.tail(lookback) if len(returns) >= lookback else returns
    if window.empty:
        weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
    else:
        vol = window.std().replace(0.0, np.nan)
        inv_vol = 1.0 / vol
        if inv_vol.sum(min_count=1) == 0 or inv_vol.isna().all():
            weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        else:
            weights = inv_vol / inv_vol.sum()
            weights = weights.fillna(0.0)

    return pd.DataFrame([weights], index=[prices.index[-1]])


def hrp_weights(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute Hierarchical Risk Parity weights via recursive bisection.

    Args:
        prices: DataFrame of daily prices.
        lookback: Rolling window for covariance estimation.

    Returns:
        Single-row DataFrame of HRP weights.
    """
    if prices.empty:
        raise ValueError("Prices DataFrame is empty.")

    returns = prices.pct_change().dropna()
    window = returns.tail(lookback) if len(returns) >= lookback else returns
    if window.empty or window.shape[1] == 0:
        weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        return pd.DataFrame([weights], index=[prices.index[-1]])

    if len(prices.columns) == 1:
        weights = pd.Series(1.0, index=prices.columns)
        return pd.DataFrame([weights], index=[prices.index[-1]])

    cov = window.cov()
    corr = window.corr()

    if corr.isna().all().all():
        weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        return pd.DataFrame([weights], index=[prices.index[-1]])

    distance = np.sqrt((1 - corr).clip(lower=0) / 2.0)
    condensed = squareform(distance.values, checks=False)
    try:
        link = linkage(condensed, method="single")
        sort_ix = leaves_list(link).astype(int)
        ordered = corr.index[sort_ix].tolist()
    except Exception:
        weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
        return pd.DataFrame([weights], index=[prices.index[-1]])

    def _cluster_var(indices: List[str]) -> float:
        cov_slice = cov.loc[indices, indices]
        weights_vec = np.ones(len(indices)) / len(indices)
        return float(weights_vec @ cov_slice.values @ weights_vec)

    def _recursive_bisection(items: List[str]) -> dict[str, float]:
        if len(items) == 1:
            return {items[0]: 1.0}
        split = len(items) // 2
        left = items[:split]
        right = items[split:]
        left_var = _cluster_var(left)
        right_var = _cluster_var(right)
        total = left_var + right_var
        if total == 0:
            alpha_left = alpha_right = 0.5
        else:
            alpha_left = 1 - left_var / total
            alpha_right = 1 - right_var / total
        alloc = {}
        for key, value in _recursive_bisection(left).items():
            alloc[key] = value * alpha_left
        for key, value in _recursive_bisection(right).items():
            alloc[key] = value * alpha_right
        return alloc

    allocation = _recursive_bisection(ordered)
    weights_series = pd.Series(allocation, index=ordered)
    weights_series = weights_series / weights_series.sum()
    weights_series = weights_series.reindex(prices.columns, fill_value=0.0)

    return pd.DataFrame([weights_series], index=[prices.index[-1]])

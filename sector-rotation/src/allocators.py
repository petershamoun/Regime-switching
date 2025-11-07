from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def top_k_equal(scores: pd.DataFrame, universe: Iterable[str], k: int = 4) -> pd.DataFrame:
    """
    Select the top-k assets by score and allocate equal weight among them.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    universe = list(universe)
    weights = pd.DataFrame(0.0, index=scores.index, columns=universe)

    for date, row in scores.iterrows():
        ranked = row.loc[[c for c in universe if c in row.index]].dropna()
        if ranked.empty:
            continue
        selected = ranked.sort_values(ascending=False).head(k)
        allocation = 1.0 / len(selected)
        weights.loc[date, selected.index] = allocation

    return weights


def inverse_vol_weights(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Allocate weights proportional to inverse of rolling volatility.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    returns = prices.pct_change()
    rolling_vol = returns.rolling(lookback).std()
    inv_vol = 1.0 / rolling_vol.replace(0.0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    monthly_weights = weights.resample("M").last().fillna(0.0)
    return monthly_weights


def _cluster_variance(cov: pd.DataFrame) -> float:
    inv_diag = 1.0 / np.diag(cov)
    weights = inv_diag / inv_diag.sum()
    variance = float(weights.T @ cov.values @ weights)
    return variance


def _get_quasi_diag(linkage_matrix: np.ndarray, assets: List[str]) -> List[str]:
    dendro = dendrogram(linkage_matrix, labels=assets, no_plot=True)
    return dendro["ivl"]


def _hrp_allocation(cov: pd.DataFrame, ordered_assets: List[str]) -> pd.Series:
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]

        cov_left = cov.loc[left, left]
        cov_right = cov.loc[right, right]

        var_left = _cluster_variance(cov_left)
        var_right = _cluster_variance(cov_right)

        weight_left = 1.0 - var_left / (var_left + var_right)
        weight_right = 1.0 - weight_left

        weights.loc[left] *= weight_left
        weights.loc[right] *= weight_right

        clusters.append(left)
        clusters.append(right)

    return weights / weights.sum()


def hrp_weights(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Hierarchical risk parity allocation computed at monthly frequency.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    columns = prices.columns.tolist()
    if len(columns) == 1:
        monthly_index = prices.resample("M").last().index
        return pd.DataFrame(1.0, index=monthly_index, columns=columns)

    returns = prices.pct_change().dropna()
    monthly_index = returns.resample("M").last().index
    weights_list = []

    for date in monthly_index:
        window = returns.loc[:date].tail(lookback)
        if window.empty or window.shape[0] < max(20, lookback // 2):
            weights_list.append(pd.Series(0.0, index=columns, name=date))
            continue

        cov = window.cov()
        corr = window.corr().fillna(0.0)
        dist = np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0)))
        condensed = squareform(dist.values, checks=False)

        try:
            link = linkage(condensed, method="single")
            ordered_assets = _get_quasi_diag(link, corr.columns.tolist())
            allocation = _hrp_allocation(cov, ordered_assets)
        except ValueError:
            allocation = pd.Series(0.0, index=columns)

        allocation = allocation.reindex(columns, fill_value=0.0)
        weights_list.append(pd.Series(allocation, name=date))

    weights = pd.DataFrame(weights_list)
    weights.index.name = "date"
    return weights.fillna(0.0)

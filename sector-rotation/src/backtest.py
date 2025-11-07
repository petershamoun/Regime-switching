from __future__ import annotations

import numpy as np
import pandas as pd


def cap_turnover(prev_weights: pd.Series, target_weights: pd.Series, cap: float = 0.30) -> pd.Series:
    """
    Limit turnover between two weight vectors to the specified cap.
    """
    if cap <= 0:
        return target_weights

    prev_weights = prev_weights.fillna(0.0)
    target_weights = target_weights.fillna(0.0)

    diff = target_weights - prev_weights
    turnover = diff.abs().sum()
    if turnover <= cap or np.isclose(turnover, 0.0):
        return target_weights

    scale = cap / turnover
    adjusted = prev_weights + diff * scale
    return adjusted


def portfolio_returns(weights: pd.DataFrame, prices: pd.DataFrame, fee_bps: float = 10.0) -> pd.Series:
    """
    Compute daily portfolio returns with transaction fee deductions.
    """
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    returns = prices.pct_change().fillna(0.0)

    shifted_weights = weights.shift().fillna(0.0)
    gross_returns = (shifted_weights * returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    fees = turnover * (fee_bps / 10000.0)
    net_returns = gross_returns - fees
    return net_returns


def vol_target(returns: pd.Series, target_annual_vol: float = 0.12, lookback: int = 63) -> pd.Series:
    """
    Apply volatility targeting to a return series.
    """
    if target_annual_vol <= 0:
        return returns

    rolling_vol = returns.rolling(lookback).std()
    annual_vol = rolling_vol * np.sqrt(252)
    scaling = target_annual_vol / annual_vol.replace(0.0, np.nan)
    scaling = scaling.shift(1).fillna(1.0)
    scaled_returns = returns * scaling
    return scaled_returns


def perf_stats(returns: pd.Series) -> dict[str, float]:
    """
    Compute performance statistics for a daily return series.
    """
    returns = returns.dropna()
    if returns.empty:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    ann_ret = (1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0
    ann_vol = returns.std() * np.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cumulative = (1.0 + returns).cumprod()
    peaks = cumulative.cummax()
    drawdowns = cumulative / peaks - 1.0
    max_dd = drawdowns.min()

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

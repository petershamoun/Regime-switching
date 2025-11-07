from __future__ import annotations

import numpy as np
import pandas as pd


def cap_turnover(
    prev_w: pd.Series,
    target_w: pd.Series,
    cap: float = 0.30,
) -> pd.Series:
    """Scale the rebalance toward the target to respect a turnover cap.

    Args:
        prev_w: Previous period weights.
        target_w: Desired target weights.
        cap: Maximum allowable turnover (0-1).

    Returns:
        Adjusted target weights respecting the turnover constraint.
    """
    prev_aligned, target_aligned = prev_w.align(target_w, fill_value=0.0)
    diff = target_aligned - prev_aligned
    turnover = 0.5 * diff.abs().sum()
    if turnover <= cap or turnover == 0:
        return target_aligned

    scale = cap / turnover
    adjusted = prev_aligned + diff * scale
    return adjusted


def portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    fee_bps: float = 10,
) -> pd.Series:
    """Compute daily portfolio returns from periodic weights.

    Args:
        weights: DataFrame of portfolio weights (e.g., monthly).
        prices: DataFrame of daily prices.
        fee_bps: Transaction fee per trade expressed in basis points.

    Returns:
        Series of daily net returns after fees.
    """
    prices = prices.sort_index()
    returns = prices.pct_change().fillna(0.0)
    weights_daily = (
        weights.reindex(returns.index)
        .ffill()
        .reindex(columns=returns.columns, fill_value=0.0)
        .fillna(0.0)
    )
    lagged_weights = weights_daily.shift(1).fillna(0.0)
    gross = (lagged_weights * returns).sum(axis=1)

    turnover = 0.5 * (weights_daily.sub(weights_daily.shift(1)).abs().sum(axis=1))
    transaction_cost = turnover * (fee_bps / 10_000)
    net = gross - transaction_cost.fillna(0.0)
    return net


def vol_target(
    returns: pd.Series,
    target_annual_vol: float = 0.12,
    lookback: int = 63,
    max_leverage: float = 5.0,
) -> pd.Series:
    """Apply volatility targeting to a return series.

    Args:
        returns: Daily returns.
        target_annual_vol: Desired annualized volatility.
        lookback: Lookback window in days for realized volatility.
        max_leverage: Maximum leverage multiplier.

    Returns:
        Volatility-targeted return series.
    """
    daily_target = target_annual_vol / np.sqrt(252)
    realized = returns.rolling(window=lookback).std()
    scale = daily_target / realized.replace(0.0, np.nan)
    scale = scale.clip(upper=max_leverage).shift(1).fillna(1.0)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return returns * scale


def perf_stats(returns: pd.Series) -> dict[str, float]:
    """Compute performance statistics for a daily return series."""
    if returns.empty:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_dd = drawdown.min()

    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

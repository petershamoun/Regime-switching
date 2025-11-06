import pandas as pd
import numpy as np

def backtest(prices: pd.Series, pos: pd.Series, cash: pd.Series | None = None, tc_bps: float = 5.0) -> pd.DataFrame:
    """Close-to-close backtest with simple transaction costs."""
    rets = prices.pct_change().fillna(0.0)
    pos = pos.reindex(rets.index).ffill().fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    tc = turnover * (tc_bps / 1e4)  # one-way

    strat = pos * rets - tc
    if cash is not None:
        cash_ret = cash.pct_change().fillna(0.0)
        strat = pos * rets + (1 - pos) * cash_ret - tc

    equity = (1 + strat).cumprod()
    dd = equity / equity.cummax() - 1
    return pd.DataFrame({
        "bench_ret": rets,
        "strat_ret": strat,
        "position": pos,
        "turnover": turnover,
        "equity": equity,
        "drawdown": dd
    })

def annualized_stats(returns: pd.Series, freq: int = 252) -> dict:
    r = (1 + returns).prod() ** (freq / max(len(returns), 1)) - 1
    vol = returns.std() * np.sqrt(freq)
    sharpe = 0.0 if vol == 0 else (returns.mean() * freq) / vol
    mdd = ((1 + returns).cumprod() / ((1 + returns).cumprod().cummax()) - 1).min()
    return {"ann_return": r, "ann_vol": vol, "sharpe": sharpe, "max_dd": mdd}

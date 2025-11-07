from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class HMMConfig:
    lookback: int = 750
    n_states: int = 2
    rebalance: str = "M"
    random_state: Optional[int] = None


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Construct features for regime classification.
    """
    if "SPY" not in prices.columns or "^VIX" not in prices.columns:
        raise ValueError("Prices must include 'SPY' and '^VIX' for feature construction.")

    spy_ret = prices["SPY"].pct_change()
    vix_diff = prices["^VIX"].diff()

    feats = pd.DataFrame(
        {
            "ret_spy": spy_ret,
            "d_vix": vix_diff,
        },
        index=prices.index,
    ).dropna()

    return feats


def _fit_window(model: GaussianHMM, window: pd.DataFrame) -> tuple[int, np.ndarray]:
    model.fit(window.values)
    states = model.predict(window.values)
    return states[-1], states


def fit_predict_hmm(
    features: pd.DataFrame,
    lookback: int = 750,
    n_states: int = 2,
    rebalance: str = "M",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit a rolling Gaussian HMM and produce risk-on flags at the rebalancing frequency.
    """
    if features.empty:
        raise ValueError("Features dataframe is empty.")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
    )

    features = features.sort_index()
    rebal_dates = features.resample(rebalance).last().dropna().index

    results = []
    last_flag = 0

    for date in rebal_dates:
        window = features.loc[:date].tail(lookback)
        window = window.dropna()
        if len(window) < max(30, n_states * 10):
            results.append((date, last_flag))
            continue

        try:
            current_state, states = _fit_window(model, window)
        except ValueError:
            results.append((date, last_flag))
            continue

        spy_returns = window["ret_spy"].to_numpy()
        state_means = {}
        for state in range(n_states):
            mask = states == state
            if mask.any():
                state_means[state] = float(spy_returns[mask].mean())
            else:
                state_means[state] = -np.inf

        risk_on_state = max(state_means, key=state_means.get)
        flag = int(current_state == risk_on_state)
        results.append((date, flag))
        last_flag = flag

    risk_flags = pd.DataFrame(results, columns=["date", "risk_on"]).set_index("date")
    return risk_flags

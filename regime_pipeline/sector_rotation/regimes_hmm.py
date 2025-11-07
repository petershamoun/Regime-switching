from __future__ import annotations

import pandas as pd
from hmmlearn import hmm


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Construct HMM feature matrix from price data.

    Args:
        prices: Wide DataFrame of prices indexed by date.

    Returns:
        DataFrame with feature columns `ret_spy` and `d_vix`.
    """
    if "SPY" not in prices.columns or "^VIX" not in prices.columns:
        missing = {"SPY", "^VIX"}.difference(prices.columns)
        raise KeyError(f"Missing required tickers for feature construction: {missing}")

    ret_spy = prices["SPY"].pct_change()
    d_vix = prices["^VIX"].diff()
    features = pd.DataFrame({"ret_spy": ret_spy, "d_vix": d_vix})
    return features.dropna()


def fit_predict_hmm(
    features: pd.DataFrame,
    lookback: int = 750,
    n_states: int = 2,
    rebal: str = "M",
) -> pd.DataFrame:
    """Fit a rolling Gaussian HMM and infer risk regimes.

    Args:
        features: DataFrame of features with daily frequency.
        lookback: Number of observations for each rolling fit.
        n_states: Number of hidden states.
        rebal: Rebalance frequency (pandas offset alias).

    Returns:
        DataFrame with a `risk_on` column (1 for risk-on, 0 otherwise).
    """
    if "ret_spy" not in features.columns:
        raise KeyError("Feature matrix must include 'ret_spy'.")

    feature_matrix = features.dropna().copy()
    if feature_matrix.empty:
        raise ValueError("No data available to fit the HMM.")

    evaluation_dates = feature_matrix.resample(rebal).last().index
    risk_series = pd.Series(index=feature_matrix.index, dtype=float)

    for current_date in evaluation_dates:
        window = feature_matrix.loc[:current_date].tail(lookback)
        if len(window) < max(n_states * 10, lookback // 2):
            continue

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=None,
        )

        try:
            model.fit(window.values)
            states = model.predict(window.values)
        except Exception:
            continue

        spy_means = {
            state: window.iloc[states == state]["ret_spy"].mean() for state in range(n_states)
        }
        risk_on_state = max(spy_means, key=spy_means.get)
        risk_flag = int(states[-1] == risk_on_state)
        risk_series.loc[current_date] = risk_flag

    risk_series = risk_series.reindex(feature_matrix.index).ffill().fillna(0.0).astype(int)
    return pd.DataFrame({"risk_on": risk_series})

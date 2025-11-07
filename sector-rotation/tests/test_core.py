from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest, data, regimes_hmm, signals


def test_download_prices_from_cache(tmp_path):
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    cache_df = pd.DataFrame(
        {
            "SPY": np.linspace(300, 305, len(dates)),
            "^VIX": np.linspace(20, 22, len(dates)),
        },
        index=dates,
    )
    cache = tmp_path / "prices.csv"
    cache_df.to_csv(cache)

    loaded = data.download_prices(["SPY", "^VIX"], start="2020-01-01", end="2020-01-10", cache_path=cache)
    assert set(loaded.columns) == {"SPY", "^VIX"}
    assert len(loaded) == len(cache_df)


def test_make_features_columns():
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(300, 310, len(dates)),
            "^VIX": np.linspace(20, 21, len(dates)),
        },
        index=dates,
    )
    feats = regimes_hmm.make_features(prices)
    assert {"ret_spy", "d_vix"}.issubset(feats.columns)
    assert feats.index[0] > dates[0]


def test_trailing_return_shape():
    index = pd.date_range("2019-01-01", periods=500, freq="B")
    prices = pd.DataFrame(
        {
            "XLY": np.linspace(100, 150, len(index)),
            "XLF": np.linspace(50, 80, len(index)),
        },
        index=index,
    )
    months = 12
    skip_last = 1
    momentum = signals.trailing_return(prices, months=months, skip_last=skip_last)
    monthly_rows = len(prices.resample("M").last())
    expected = monthly_rows - (months + skip_last)
    assert momentum.shape[0] == expected
    assert set(momentum.columns) == {"XLY", "XLF"}


def test_cap_turnover_limits_changes():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 1.0, "B": 0.0})
    capped = backtest.cap_turnover(prev, target, cap=0.30)
    turnover = (capped - prev).abs().sum()
    assert turnover <= 0.30 + 1e-9

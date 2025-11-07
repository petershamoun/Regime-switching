from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_pipeline.sector_rotation import data, regimes_hmm, signals


def test_download_prices_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tickers = ["AAA", "BBB"]
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product([["Adj Close"], tickers])
    values = np.arange(len(dates) * len(tickers)).reshape(len(dates), len(tickers))
    fake_df = pd.DataFrame(values, index=dates, columns=columns)

    call_counter = {"count": 0}

    def fake_download(*args, **kwargs):
        call_counter["count"] += 1
        return fake_df

    monkeypatch.setattr(data.yf, "download", fake_download)

    cache_file = tmp_path / "prices.csv"
    prices = data.download_prices(tickers, start="2020-01-01", cache_path=cache_file)
    assert set(prices.columns) == set(tickers)
    assert call_counter["count"] == 1
    assert cache_file.exists()

    # second call should hit the cache and avoid another download
    prices_cached = data.download_prices(tickers, start="2020-01-01", cache_path=cache_file)
    assert call_counter["count"] == 1
    pd.testing.assert_frame_equal(prices, prices_cached)


def test_make_features_columns() -> None:
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100, 110, len(dates)),
            "^VIX": np.linspace(20, 25, len(dates)),
        },
        index=dates,
    )
    features = regimes_hmm.make_features(prices)
    assert {"ret_spy", "d_vix"}.issubset(features.columns)
    assert not features.isna().all().any()


def test_trailing_return_shape() -> None:
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    prices = pd.DataFrame(
        {
            "XLY": np.linspace(100, 150, len(dates)),
            "XLP": np.linspace(90, 120, len(dates)),
        },
        index=dates,
    )
    momentum = signals.trailing_return(prices, months=12, skip_last=1)
    assert pd.infer_freq(momentum.index) == "M"
    assert set(momentum.columns) == {"XLY", "XLP"}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from . import backtest, data, model, plots, signals


@dataclass
class RegimeDetectionResult:
    """Container for regime-detection artifacts."""

    config: Dict
    returns: pd.DataFrame
    prices: pd.DataFrame
    probabilities: pd.DataFrame
    signal: pd.Series
    backtest: pd.DataFrame
    stats: Dict[str, float]

    def risk_flag(self) -> pd.Series:
        """Binary 1/0 risk-on signal derived from the trading position."""
        return (self.signal > 0.5).astype(int)


def run_regime_detection(
    config_path: Path | str | None = None,
    *,
    output_dir: Path | str | None = None,
    show_plots: bool = False,
) -> RegimeDetectionResult:
    """Execute the regime-detection workflow and optionally persist outputs."""
    cfg = data.load_config(config_path)

    bench = cfg["data"]["tickers"][0]
    cash = cfg["data"]["tickers"][1] if len(cfg["data"]["tickers"]) > 1 else None

    returns = data.load_prices(cfg, returns=True)
    returns = returns[[bench] + ([cash] if cash else [])].dropna()

    prices = (1 + returns).cumprod()

    res = model.fit_markov_model(returns[bench], k_regimes=cfg["model"]["n_states"])
    probabilities = model.extract_probabilities(res)
    bull_state = model.identify_bull_state(res)
    bull_col = f"Regime_{bull_state}"

    threshold = cfg["signals"]["threshold"]
    buy = max(0.5, threshold)
    sell = min(0.5, 1 - threshold)
    positions = signals.hysteresis_signal(probabilities[bull_col], buy=buy, sell=sell)
    smoothed = signals.smooth_positions(positions, k=3)

    cash_series = prices[cash] if cash else None
    bt = backtest.backtest(prices[bench], smoothed, cash=cash_series, tc_bps=cfg.get("trading_cost_bps", 5.0))
    stats = backtest.annualized_stats(bt["strat_ret"])

    result = RegimeDetectionResult(
        config=cfg,
        returns=returns,
        prices=prices,
        probabilities=probabilities,
        signal=smoothed,
        backtest=bt,
        stats=stats,
    )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        probabilities.to_parquet(out_dir / "regime_probabilities.parquet")
        result.risk_flag().to_csv(out_dir / "risk_signal.csv", header=["risk_on"])
        bt.to_csv(out_dir / "regime_backtest.csv")
        pd.Series(stats).to_csv(out_dir / "regime_stats.csv", header=["value"])

    if show_plots:
        plots.plot_regimes(prices[bench], probabilities, bull_col=bull_col)
        plots.plot_equity(bt)
        plots.plot_drawdown(bt)
        plt.show()

    return result

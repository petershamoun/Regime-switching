# src/run_baseline.py
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

from .data import load_config, load_prices
from .model import fit_markov_model, extract_probabilities, identify_bull_state
from .signals import hysteresis_signal, smooth_positions
from .backtest import backtest, annualized_stats
from .plots import plot_regimes, plot_equity, plot_drawdown


def main():
    # 1) config
    cfg = load_config()
    bench, cash = cfg["data"]["tickers"][0], (cfg["data"]["tickers"][1] if len(cfg["data"]["tickers"]) > 1 else None)

    # 2) data (returns for all tickers)
    rets = load_prices()     # this returns RETURNS (pct_change), by design in data.py
    rets = rets[[bench] + ([cash] if cash else [])].dropna()

    # build synthetic price indexes so the backtester can use "prices"
    prices = (1 + rets[bench]).cumprod()
    prices.name = bench
    cash_px = None
    if cash:
        cash_px = (1 + rets[cash]).cumprod()
        cash_px.name = cash

    # 3) fit MSM on benchmark returns
    res = fit_markov_model(rets[bench], k_regimes=cfg["model"]["n_states"])
    probs = extract_probabilities(res)
    bull_state = identify_bull_state(res)
    bull_col = f"Regime_{bull_state}"

    # 4) signals
    pos = hysteresis_signal(probs[bull_col], buy=max(0.5, cfg["signals"]["threshold"]), sell=min(0.5, 1-cfg["signals"]["threshold"]))
    pos = smooth_positions(pos, k=3)

    # 5) backtest
    bt = backtest(prices, pos, cash=cash_px, tc_bps=5.0)
    stats = annualized_stats(bt["strat_ret"])
    bench_stats = annualized_stats(bt["bench_ret"])

    print("\n=== Strategy ===")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print("\n=== Benchmark ===")
    for k, v in bench_stats.items():
        print(f"{k}: {v:.4f}")

    # 6) plots
    plot_regimes(prices, probs, bull_col=bull_col)
    plot_equity(bt)
    plot_drawdown(bt)
    plt.show()


if __name__ == "__main__":
    main()

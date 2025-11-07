## Overview

This project implements a production-ready, regime-switching sector rotation strategy that allocates across SPDR sector ETFs using a two-state hidden Markov model (HMM) to distinguish between risk-on and risk-off environments.

## Method

- **Regime detection:** Fit a rolling two-state HMM on daily SPY returns and VIX changes to infer risk-on vs risk-off periods.
- **Signals:** Use 12-1 cross-sectional momentum for sector selection and an absolute momentum filter on SPY to guard against broad market drawdowns.
- **Allocation:** In risk-on regimes allocate to the top momentum sectors with inverse-volatility weighting; in risk-off regimes fall back to defensive sectors with a hierarchical risk parity (HRP) weighting scheme, defaulting to equal-weight if HRP fails.
- **Portfolio management:** Rebala​​nce monthly with turnover caps, transaction fees, and volatility targeting to 12% annualized.

## Data

Daily adjusted close prices for SPDR sector ETFs (`XLY`, `XLP`, `XLE`, `XLF`, `XLK`, `XLI`, `XLB`, `XLV`, `XLU`, `XLRE`, `XLC`), `SPY`, `IEF`, and the CBOE `^VIX` index are sourced from Yahoo Finance via `yfinance`. Data is cached as CSV under `data/`.

## How to Run

1. Create and activate a local virtual environment (see commands below).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the backtest: `python scripts/run_backtest.py --config config.yaml`.

The script saves `data/equity_curve.csv`, `data/weights.csv`, and `data/equity.png`, and prints key performance statistics to the console.

## Config

Strategy parameters are configured through `config.yaml`. Important settings include:

- Backtest dates, rebalance frequency, turnover cap, fee assumptions, and volatility target.
- HMM hyperparameters and feature selection.
- Momentum lookbacks, sector universe, and defensive assets.

Modify `config.yaml` to experiment with different universes, lookbacks, or allocation rules without touching the code.

## Results

Running the baseline configuration will produce the equity curve plot at `data/equity.png` alongside CSV artifacts for further analysis.

## Limitations

- Yahoo Finance data can be revised or subject to outages.
- The simple two-state HMM may miss nuanced market regimes.
- The strategy omits slippage modeling and assumes that monthly rebalancing captures all necessary adjustments.

## Next Steps

- Experiment with additional features in the HMM (e.g., macro factors).
- Extend the allocation logic with downside protection overlays or dynamic leverage.
- Automate parameter sweeps and add a concise tear sheet for reporting.

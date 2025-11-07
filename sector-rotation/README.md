# Regime-Switching Sector Rotation

## Overview
This project implements a production-ready regime-switching sector rotation strategy that allocates across SPDR sector ETFs using a two-state hidden Markov model (HMM) to distinguish between risk-on and risk-off environments. The framework is fully configuration-driven and produces reproducible outputs and performance diagnostics.

## Method
- **Data**: Daily adjusted closes for the 11 SPDR sector ETFs, `SPY`, `IEF`, and `^VIX` are fetched via `yfinance` with local CSV caching.
- **Regimes**: A 2-state Gaussian HMM is refit monthly on a rolling 750-day window using features `[SPY daily returns, ΔVIX]`. The state with the higher average SPY return is mapped to risk-on.
- **Signals**: Cross-sectional momentum follows the classic 12–1 specification. A simple absolute momentum filter is applied to SPY.
- **Allocation**:
  - Risk-on + absolute momentum: top-k sectors by momentum, inverse-vol weighted.
  - Risk-off or failing absolute momentum: defensives (`XLP`, `XLV`, `XLU`) plus `IEF` via HRP (fallback to equal-weight).
- **Portfolio construction**: Monthly rebalance, 30% turnover cap, 10 bps trading costs, and 12% annualized volatility targeting.

## Data
All artifacts are written to the `data/` directory (configurable). The price loader caches downloads to accelerate repeated runs.

## How to Run
```bash
python -m venv .venv
. .venv/bin/activate            # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_backtest.py --config config.yaml
```
Outputs include `data/equity_curve.csv`, `data/weights.csv`, `data/equity.png`, and console performance statistics.

## Config
The strategy is driven by `config.yaml`. Key parameters:
- `start`, `end`: backtest window.
- `rebalance`: pandas frequency string (default monthly).
- `fee_bps`, `turnover_cap`, `target_annual_vol`.
- `hmm`: number of states, rolling lookback, optional random seed, and feature names.
- `signals`: momentum horizon, skip window, top-k sectors, and lookbacks for risk models.
- `defensives`, `sectors`, `bench`: tradable universes and defensive sleeve.

## Results
The plot below is generated after running the backtest:
![Equity Curve](data/equity.png)

## Limitations
- Relies on liquid ETF history; results may degrade for shorter samples.
- HMM estimation assumes Gaussian emissions and may be sensitive to lookback choice.
- Volatility targeting assumes continuous scaling without execution slippage.

## Next Steps
- Add CLI options for alternative strategies (e.g., trend-following comparison).
- Extend reporting with rolling risk metrics and factor attribution.
- Implement grid search utilities for allocation hyperparameters.

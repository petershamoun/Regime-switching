## Overview

This module houses Part 2 of the regime pipeline: a production-ready, regime-aware sector rotation strategy that consumes the risk signal generated in Part 1 (`regime_pipeline.regime_detection`).

## Method

- **Regime gate:** Load the binary risk-on/off series produced during Part 1 (or recompute it on the fly).  
  When the regime flag is off, the allocator falls back to defensive assets.
- **Signals:** Use 12-1 cross-sectional momentum for sector selection and an absolute momentum filter on `SPY` to guard against broad market drawdowns.
- **Allocation:** In risk-on regimes allocate to the top momentum sectors with inverse-volatility weighting; in risk-off regimes fall back to defensive sectors with a hierarchical risk parity (HRP) scheme, defaulting to equal-weight if HRP fails.
- **Portfolio management:** Rebalance monthly with turnover caps, transaction fees, and volatility targeting to 12% annualised.

## Data

Daily adjusted closes for SPDR sectors (`XLY`, `XLP`, `XLE`, `XLF`, `XLK`, `XLI`, `XLB`, `XLV`, `XLU`, `XLRE`, `XLC`), `SPY`, `IEF`, and `^VIX` are downloaded via `yfinance` and cached under `data/`.

## How to Run

1. Install the shared requirements: `pip install -r requirements.txt`.
2. Run Part 1 to generate the regime artefacts (see repository README).
3. Execute the sector rotation backtest:

   ```bash
   python scripts/run_sector_rotation.py \
     --config configs/sector_rotation.yaml \
     --regime-artifacts artifacts/regime_detection
   ```

The script saves `data/equity_curve.csv`, `data/weights.csv`, and `data/equity.png`, and prints key performance statistics to the console.

## Config

Strategy parameters live in `configs/sector_rotation.yaml`. Key knobs:

- Backtest dates, rebalance frequency, turnover cap, fee assumptions, and volatility target.
- Momentum lookbacks, selection depth (`top_k`), and the defensive asset list.

Pass a different config via `--config` to experiment with alternative universes or constraints.

## Limitations

- Yahoo Finance data can be revised or suffer outages; cache thoughtfully for production.
- The two-state regime flag inherits biases from Part 1; consider enriching it with macro or volatility features.
- Slippage and liquidity modelling remain simplified — extend before trading live capital.

## Next Steps

- Layer downside overlays (e.g., trailing stops) on the risk-off allocation.
- Industrialise the reporting module into a reproducible tear sheet.
- Parameter-sweep the regime and momentum knobs to gauge robustness.

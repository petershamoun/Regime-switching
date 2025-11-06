# Regime-switching

This repository implements a **Markov-Switching Model (MSM)** for identifying latent market regimes such as bull and bear states.  
The project provides a modular, reproducible pipeline for fitting, visualizing, and backtesting regime-based strategies on financial time series.

---

## 📈 Overview

Markets alternate between distinct return-volatility regimes — expansions and contractions, calm and turbulence.  
This project models those shifts using a **two-state Markov process**, where the probability of remaining in or switching between regimes evolves dynamically through time.

The MSM is estimated via **Maximum Likelihood**, using `statsmodels`’ hidden-Markov implementation.  
Outputs include smoothed regime probabilities, expected returns per state, and derived long/flat trading signals.

---

## ⚙️ Methodology

| Step | Description |
|------|--------------|
| **1. Data ingestion** | Load benchmark asset data (e.g. SPY, BIL) via Yahoo Finance or local CSVs. |
| **2. Model fitting** | Estimate a two-state Markov-Switching model on daily log returns. |
| **3. Regime labeling** | Identify bull vs. bear regimes based on mean and volatility characteristics. |
| **4. Signal generation** | Create long/flat positions based on smoothed probabilities. |
| **5. Backtesting** | Simulate equity curves and compute risk-adjusted performance metrics. |

---

## 🧩 Project Structure

Regime-switching/

│
├── config/
│ └── config.yaml # model + data configuration
│
├── src/
│ ├── init.py
│ ├── data.py # data loader and preprocessing
│ ├── model.py # MSM estimation and regime extraction
│ ├── signals.py # signal generation logic
│ ├── backtest.py # backtest and performance metrics
│ ├── plots.py # visualization helpers
│ └── run_baseline.py # main execution script
│
├── requirements.txt
└── README.md


---

## 🧠 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt

The script:

loads price data (default: SPY & BIL),

fits a two-state MSM on returns,

prints summary statistics,

and plots regime probabilities, equity curve, and drawdown.

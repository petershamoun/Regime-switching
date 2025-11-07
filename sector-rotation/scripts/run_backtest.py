from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import allocators, backtest, data, regimes_hmm, reporting, signals, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime-switching sector rotation backtest.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    return parser.parse_args()


def build_weights(
    prices: pd.DataFrame,
    config: dict,
    risk_flags: pd.DataFrame,
) -> pd.DataFrame:
    signals_cfg = config.get("signals", {})
    top_k = int(signals_cfg.get("top_k", 4))
    mom_months = int(signals_cfg.get("momentum_months", 12))
    skip_last = int(signals_cfg.get("skip_last_months", 1))
    inv_vol_lookback = int(signals_cfg.get("inverse_vol_lookback_days", 60))
    hrp_lookback = int(signals_cfg.get("hrp_lookback_days", 60))

    sectors = config.get("sectors", [])
    bench = config.get("bench", [])
    defensives = config.get("defensives", [])

    sector_prices = prices.loc[:, [c for c in sectors if c in prices.columns]]
    if sector_prices.empty:
        raise ValueError("No sector prices available in the data frame.")
    momentum_scores = signals.trailing_return(sector_prices, months=mom_months, skip_last=skip_last)
    monthly_index = momentum_scores.index

    inverse_vol = allocators.inverse_vol_weights(sector_prices, lookback=inv_vol_lookback)
    inverse_vol = inverse_vol.reindex(monthly_index).ffill().fillna(0.0)

    defensive_universe = [sym for sym in defensives if sym in prices.columns]
    if "IEF" in bench and "IEF" in prices.columns and "IEF" not in defensive_universe:
        defensive_universe.append("IEF")
    if not defensive_universe:
        defensive_universe = ["SPY"] if "SPY" in prices.columns else sectors[:1]

    defensive_prices = prices.loc[:, defensive_universe]
    hrp = allocators.hrp_weights(defensive_prices, lookback=hrp_lookback)
    hrp = hrp.reindex(monthly_index).ffill().fillna(0.0)

    if "SPY" not in prices.columns:
        raise ValueError("SPY prices required for absolute momentum signal.")
    abs_mom = signals.absolute_momentum(prices["SPY"], months=mom_months)
    abs_mom = abs_mom.reindex(monthly_index).ffill().fillna(0)

    risk_flags = risk_flags.reindex(monthly_index).ffill().fillna(0)

    all_assets = list(dict.fromkeys(sectors + bench))
    weights = pd.DataFrame(0.0, index=monthly_index, columns=all_assets)

    prev = pd.Series(0.0, index=all_assets)
    cap = float(config.get("turnover_cap", 0.30))

    for date in monthly_index:
        risk_on = int(risk_flags.loc[date, "risk_on"])
        abs_on = int(abs_mom.loc[date])

        if risk_on and abs_on:
            row_scores = momentum_scores.loc[date].dropna()
            ordered = row_scores.sort_values(ascending=False)
            selected = ordered.head(top_k).index.tolist()
            if not selected:
                selected = sectors
            base = inverse_vol.loc[date].reindex(selected).fillna(0.0)
            if base.sum() == 0.0:
                base = pd.Series(1.0 / len(selected), index=selected)
            else:
                base = base / base.sum()
        else:
            base = hrp.loc[date].reindex(defensive_universe).fillna(0.0)
            if base.sum() == 0.0:
                base = pd.Series(1.0 / len(defensive_universe), index=defensive_universe)
            else:
                base = base / base.sum()

        target = pd.Series(0.0, index=all_assets)
        target.loc[base.index] = base
        capped = backtest.cap_turnover(prev, target, cap=cap)
        weights.loc[date] = capped
        prev = capped

    weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return weights


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = utils.load_config(config_path)

    data_dir = (config_path.parent / config.get("data_dir", "data")).resolve()
    cache_path = config.get("data_cache")
    if cache_path:
        cache_path = (config_path.parent / cache_path).resolve()

    start = config.get("start")
    end = config.get("end") or None
    prices = data.load_all(start=start, end=end, cache_path=cache_path)

    features = regimes_hmm.make_features(prices)
    hmm_cfg = config.get("hmm", {})
    feature_cols = hmm_cfg.get("features", ["ret_spy", "d_vix"])
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing features required for HMM: {missing_cols}")
    risk_flags = regimes_hmm.fit_predict_hmm(
        features=features.loc[:, feature_cols],
        lookback=int(hmm_cfg.get("fit_lookback_days", 750)),
        n_states=int(hmm_cfg.get("n_states", 2)),
        rebalance=config.get("rebalance", "M"),
        random_state=hmm_cfg.get("random_state"),
    )

    weights = build_weights(prices, config, risk_flags)

    trade_assets = weights.columns.tolist()
    price_subset = prices.loc[:, trade_assets]

    daily_returns = backtest.portfolio_returns(
        weights=weights,
        prices=price_subset,
        fee_bps=float(config.get("fee_bps", 10)),
    )
    targeted_returns = backtest.vol_target(
        daily_returns,
        target_annual_vol=float(config.get("target_annual_vol", 0.12)),
    )

    equity_curve = (1.0 + targeted_returns).cumprod()

    data_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame({"equity": equity_curve})
    equity_df.to_csv(data_dir / "equity_curve.csv", index=True)
    weights.to_csv(data_dir / "weights.csv", index=True)

    reporting.save_equity_plot(targeted_returns, path=data_dir / "equity.png")

    stats = backtest.perf_stats(targeted_returns)
    print("Performance Statistics")
    for key, value in stats.items():
        if key == "max_dd":
            print(f"{key:>10}: {value:.2%}")
        else:
            print(f"{key:>10}: {value:.4f}")


if __name__ == "__main__":
    main()

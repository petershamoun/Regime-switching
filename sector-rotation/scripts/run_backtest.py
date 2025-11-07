from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src import allocators, backtest, data, regimes_hmm, reporting, signals, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime-switching sector rotation backtest.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file.")
    return parser.parse_args()


def build_universe(config: dict) -> tuple[List[str], List[str], List[str]]:
    sectors = list(dict.fromkeys(config.get("sectors", [])))
    defensives = list(dict.fromkeys(config.get("defensives", [])))
    bench = list(dict.fromkeys(config.get("bench", [])))
    investable = sorted(set(sectors + defensives + bench))
    feature_tickers = sorted(set(investable + ["SPY", "^VIX"]))
    return sectors, defensives, feature_tickers


def ensure_weights_sum(weights: pd.Series) -> pd.Series:
    total = weights.sum()
    if total > 0:
        return weights / total
    return weights


def determine_risk_off_universe(defensives: List[str], bench: List[str], prices: pd.DataFrame) -> List[str]:
    universe = [ticker for ticker in defensives if ticker in prices.columns]
    if "IEF" in prices.columns and "IEF" not in universe and "IEF" in bench + defensives:
        universe.append("IEF")
    return universe


def main() -> None:
    args = parse_args()
    config = utils.load_config(args.config)

    sectors, defensives, feature_tickers = build_universe(config)
    bench = list(dict.fromkeys(config.get("bench", [])))
    start = config.get("start", "2004-01-01")
    end = config.get("end")
    rebalance = config.get("rebalance", "M")

    prices = data.load_all(start=start, end=end, tickers=feature_tickers)
    prices = prices.dropna(how="all")

    features = regimes_hmm.make_features(prices)
    regimes = regimes_hmm.fit_predict_hmm(
        features,
        lookback=config.get("hmm", {}).get("fit_lookback_days", 750),
        n_states=config.get("hmm", {}).get("n_states", 2),
        rebal=rebalance,
    )

    signals_cfg = config.get("signals", {})
    momentum_scores = signals.trailing_return(
        prices[sectors],
        months=signals_cfg.get("momentum_months", 12),
        skip_last=signals_cfg.get("skip_last_months", 1),
    )

    abs_mom = signals.absolute_momentum(
        prices["SPY"],
        months=signals_cfg.get("momentum_months", 12),
        threshold=signals_cfg.get("absolute_threshold", 0.0),
    )

    rebalance_dates = momentum_scores.index
    risk_monthly = regimes.reindex(rebalance_dates, method="ffill").fillna(0)
    abs_monthly = abs_mom.reindex(rebalance_dates, method="ffill").fillna(0)

    investable = sorted(set(sectors + defensives + bench))
    weights_records: list[pd.Series] = []
    prev_weights = pd.Series(0.0, index=investable)

    inv_vol_lookback = signals_cfg.get("inverse_vol_lookback", 60)
    hrp_lookback = signals_cfg.get("hrp_lookback", 60)

    for dt in rebalance_dates:
        risk_on = int(risk_monthly.loc[dt, "risk_on"]) if "risk_on" in risk_monthly.columns else 0
        abs_on = int(abs_monthly.loc[dt]) if dt in abs_monthly.index else 0

        target = pd.Series(0.0, index=investable)

        if risk_on and abs_on:
            score_row = momentum_scores.loc[dt].dropna()
            ordered = score_row.reindex(sectors).dropna()
            top_k = ordered.nlargest(min(signals_cfg.get("top_k", 4), len(ordered)))
            if not top_k.empty:
                history = prices.loc[:dt, top_k.index].dropna(how="all")
                if not history.empty:
                    ivol = allocators.inverse_vol_weights(history, lookback=inv_vol_lookback).iloc[-1]
                    ivol = ivol.reindex(investable, fill_value=0.0)
                    target.update(ivol)
        else:
            defensive_universe = determine_risk_off_universe(defensives, bench, prices)
            if defensive_universe:
                history = prices.loc[:dt, defensive_universe].dropna(how="all")
                if not history.empty:
                    try:
                        hrp = allocators.hrp_weights(history, lookback=hrp_lookback).iloc[-1]
                        hrp = hrp.reindex(investable, fill_value=0.0)
                        if hrp.sum() <= 0:
                            raise ValueError("HRP returned zero weights.")
                        target.update(hrp)
                    except Exception:
                        equal_weight = 1.0 / len(defensive_universe)
                        for ticker in defensive_universe:
                            if ticker in target.index:
                                target.loc[ticker] = equal_weight

        target = ensure_weights_sum(target)
        adjusted = backtest.cap_turnover(prev_weights, target, cap=config.get("turnover_cap", 0.30))
        adjusted = ensure_weights_sum(adjusted)

        weights_records.append(adjusted)
        prev_weights = adjusted

    weights_df = pd.DataFrame(weights_records, index=rebalance_dates).reindex(columns=investable).fillna(0.0)

    portfolio_rets = backtest.portfolio_returns(weights_df, prices[investable], fee_bps=config.get("fee_bps", 10))
    targeted_rets = backtest.vol_target(
        portfolio_rets,
        target_annual_vol=config.get("target_annual_vol", 0.12),
        lookback=config.get("vol_target_lookback", 63),
    )

    equity_curve = (1 + targeted_rets).cumprod()
    stats = backtest.perf_stats(targeted_rets)

    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame({"returns": targeted_rets, "equity": equity_curve})
    equity_df.to_csv(data_path / "equity_curve.csv")
    weights_df.to_csv(data_path / "weights.csv")
    reporting.save_equity_plot(targeted_rets, data_path / "equity.png")

    print("Performance Summary")
    print("-------------------")
    print(f"Annualized Return: {stats['ann_ret']:.2%}")
    print(f"Annualized Volatility: {stats['ann_vol']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['max_dd']:.2%}")


if __name__ == "__main__":
    main()

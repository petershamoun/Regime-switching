from __future__ import annotations

import argparse

from .pipeline import run_regime_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the regime-detection stage of the pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to a regime detection config file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/regime_detection",
        help="Directory to persist regime detection artifacts.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable interactive visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_regime_detection(
        config_path=args.config,
        output_dir=args.output_dir,
        show_plots=not args.no_plots,
    )

    print("Regime Detection Summary")
    print("------------------------")
    for key, value in result.stats.items():
        print(f"{key}: {value:.4f}")

    print("\nLatest Risk State:")
    latest = result.risk_flag().iloc[-1]
    print("Risk-on" if latest == 1 else "Risk-off")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from regime_pipeline.regime_detection.pipeline import run_regime_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Part 1: regime detection.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a regime detection configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/regime_detection",
        help="Where to persist regime detection artifacts.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable interactive charts from the regime detection stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_regime_detection(
        config_path=args.config,
        output_dir=args.output_dir,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

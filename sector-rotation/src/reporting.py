from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_plot(returns: pd.Series, path: str | Path = "data/equity.png") -> None:
    """Save an equity curve plot to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    equity = (1 + returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(equity.index, equity.values, label="Equity Curve")
    plt.title("Regime-Switching Sector Rotation")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

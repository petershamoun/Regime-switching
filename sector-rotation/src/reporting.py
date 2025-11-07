from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_plot(returns: pd.Series, path: str | Path = "data/equity.png") -> None:
    """
    Save an equity curve plot to the specified path.
    """
    equity = (1.0 + returns).cumprod()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(equity.index, equity.values, label="Strategy")
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

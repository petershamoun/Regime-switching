import matplotlib.pyplot as plt
import pandas as pd

def plot_regimes(prices: pd.Series, probs: pd.DataFrame, bull_col: str = "Regime_0"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(prices, color="black", label="Price")
    ax2 = ax1.twinx()
    ax2.plot(probs[bull_col], color="blue", alpha=0.6, label="Bull prob")
    ax1.set_title("Price and Bull Regime Probability")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_equity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["equity"], label="Strategy Equity", color="green")
    ax.set_title("Strategy Equity Curve")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_drawdown(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.fill_between(df.index, df["drawdown"], color="red", alpha=0.5)
    ax.set_title("Drawdown")
    plt.tight_layout()
    plt.show()

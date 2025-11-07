import pandas as pd

def hysteresis_signal(bull_prob: pd.Series, buy=0.60, sell=0.40) -> pd.Series:
    """
    Long/flat with hysteresis:
      - go long when prob >= buy
      - go flat when prob <= sell
      - otherwise keep previous state
    """
    pos, state = pd.Series(index=bull_prob.index, dtype=float), 0.0
    for t, p in bull_prob.items():
        if p >= buy:
            state = 1.0
        elif p <= sell:
            state = 0.0
        pos.loc[t] = state
    return pos

def smooth_positions(pos: pd.Series, k: int = 3) -> pd.Series:
    if k <= 1:
        return pos
    return pos.rolling(k, min_periods=1).mean().clip(0, 1)

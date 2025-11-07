import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def fit_markov_model(returns: pd.Series, k_regimes: int = 2):
    """
    Fit a Markov-switching model on daily returns.
    """
    model = MarkovRegression(returns, k_regimes=k_regimes, trend="c", switching_variance=True)
    res = model.fit(disp=False)
    return res

def extract_probabilities(res):
    """
    Extract smoothed state probabilities.
    """
    probs = pd.DataFrame(res.smoothed_marginal_probabilities.T, columns=[f"Regime_{i}" for i in range(res.k_regimes)])
    return probs

def identify_bull_state(res):
    """
    Identify which regime corresponds to the 'bull' state (higher mean).
    """
    params = res.params

    if isinstance(params, pd.Series):
        means = [params.get(f"const[{i}]") for i in range(res.k_regimes)]
    else:
        names = getattr(res.model, "param_names", [])
        means = []
        for i in range(res.k_regimes):
            name = f"const[{i}]"
            idx = names.index(name) if name in names else i
            means.append(params[idx])

    return int(np.argmax(means))

if __name__ == "__main__":
    print("Model module loaded.")

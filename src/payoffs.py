import numpy as np

def european_call_payoff(S_T,K):

    return np.maximum(S_T-K,0)

def european_put_payoff(S_T,K):

    return np.maximum(K-S_T,0)

def asian_call_payoff(paths, K):
    avg_price = paths.mean(axis=1)
    return np.maximum(avg_price - K, 0)

def asian_put_payoff(paths, K):
    avg_price = paths.mean(axis=1)
    return np.maximum(K - avg_price, 0)


def barrier_call_knockout_payoff(paths, K, barrier, barrier_type="up-and-out"):
    
    if barrier_type == "up-and-out":
        breached = (paths >= barrier).any(axis=1)
    elif barrier_type == "down-and-out":
        breached = (paths <= barrier).any(axis=1)
    else:
        raise ValueError("barrier_type must be 'up-and-out' or 'down-and-out'")

    S_T = paths[:, -1]
    vanilla_payoff = np.maximum(S_T - K, 0)
    return np.where(breached, 0, vanilla_payoff)
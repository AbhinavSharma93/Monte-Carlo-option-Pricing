import numpy as np
from scipy.stats import norm
from src.simulator import simulate_gbm_paths
from src.payoffs import asian_call_payoff, asian_put_payoff, barrier_call_knockout_payoff

def price_european_option(S0, K, r, sigma, T, n_steps, n_paths, option_type="call",seed=None):
    #gbm paths from simulator
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    S_T=paths[:,-1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    #discounting back the future cash
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = discounted_payoffs.mean()
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_paths)

    return {"price": price, "std_error": std_error}

def black_scholes_price(S0, K, r, sigma, T, option_type="call"):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def price_asian_option(S0, K, r, sigma, T, n_steps, n_paths, option_type="call", seed=None):
    
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)

    if option_type == "call":
        payoffs = asian_call_payoff(paths, K)
    elif option_type == "put":
        payoffs = asian_put_payoff(paths, K)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return {"price": discounted.mean(), "std_error": discounted.std(ddof=1) / np.sqrt(n_paths)}


def price_barrier_option(S0, K, barrier, r, sigma, T, n_steps, n_paths, barrier_type="up-and-out", seed=None):
    
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    payoffs = barrier_call_knockout_payoff(paths, K, barrier, barrier_type)
    discounted = np.exp(-r * T) * payoffs
    return {"price": discounted.mean(), "std_error": discounted.std(ddof=1) / np.sqrt(n_paths)}
    

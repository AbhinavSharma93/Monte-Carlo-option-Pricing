import numpy as np
from src.simulator import simulate_gbm_paths


def simulate_gbm_antithetic(S0, r, sigma, T, n_steps, n_paths, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    paths_pos = np.zeros((n_paths, n_steps + 1))
    paths_neg = np.zeros((n_paths, n_steps + 1))
    paths_pos[:, 0] = S0
    paths_neg[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths_pos[:, t] = paths_pos[:, t - 1] * np.exp(drift + vol * Z)
        paths_neg[:, t] = paths_neg[:, t - 1] * np.exp(drift - vol * Z)

    return np.vstack([paths_pos, paths_neg])


def price_with_antithetic(S0, K, r, sigma, T, n_steps, n_paths, option_type="call", seed=None):
    
    paths = simulate_gbm_antithetic(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))

    return {"price": price, "std_error": std_error}
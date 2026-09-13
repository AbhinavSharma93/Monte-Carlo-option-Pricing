import numpy as np
from src.simulator import simulate_gbm_paths


def pathwise_delta_vega_call(S0, K, r, sigma, T, n_steps, n_paths, seed=None):
   
    if seed is not None:
        np.random.seed(seed)

    
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    S = np.full(n_paths, S0, dtype=float)
    Z_cumulative = np.zeros(n_paths)  # tracks sum  for vega

    for t in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp(drift + vol * Z)
        Z_cumulative += np.sqrt(dt) * Z

    S_T = S
    in_the_money = (S_T > K).astype(float)
    discount = np.exp(-r * T)

    # Delta
    delta_paths = in_the_money * (S_T / S0)
    delta = discount * delta_paths.mean()
    delta_se = discount * delta_paths.std(ddof=1) / np.sqrt(n_paths)

    # Vega
    dS_dsigma = S_T * (-sigma * T + Z_cumulative)
    vega_paths = in_the_money * dS_dsigma
    vega = discount * vega_paths.mean()
    vega_se = discount * vega_paths.std(ddof=1) / np.sqrt(n_paths)

    return {
        "delta": delta, "delta_se": delta_se,
        "vega": vega, "vega_se": vega_se
    }


def black_scholes_delta_vega(S0, K, r, sigma, T):
    
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    return {"delta": delta, "vega": vega}

def pathwise_theta_rho_call(S0, K, r, sigma, T, n_steps, n_paths, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    S = np.full(n_paths, S0, dtype=float)
    Z_cumulative = np.zeros(n_paths)

    for t in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp(drift + vol * Z)
        Z_cumulative += np.sqrt(dt) * Z

    S_T = S
    in_the_money = (S_T > K).astype(float)
    discount = np.exp(-r * T)
    payoff = np.maximum(S_T - K, 0)

    # Rho: dS_T/dr = S_T * T  
    dS_dr = S_T * T
    rho_paths = discount * (in_the_money * dS_dr - T * payoff)
    
    rho = rho_paths.mean()
    rho_se = rho_paths.std(ddof=1) / np.sqrt(n_paths)

    
    dlogST_dT = (r - 0.5 * sigma**2) + sigma * Z_cumulative / (2 * T)
    dS_dT = S_T * dlogST_dT
    theta_paths = discount * (in_the_money * dS_dT - r * payoff)
    
    theta = -theta_paths.mean()  
    theta_se = theta_paths.std(ddof=1) / np.sqrt(n_paths)

    return {
        "rho": rho, "rho_se": rho_se,
        "theta": theta, "theta_se": theta_se
    }


def black_scholes_theta_rho(S0, K, r, sigma, T):
   
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {"theta": theta, "rho": rho}



def lrm_gamma_call(S0, K, r, sigma, T, n_steps, n_paths, seed=None):
    
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    S_T = paths[:, -1]

    payoff = np.maximum(S_T - K, 0)

    
    Z = (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    weight = (Z**2 - 1 - Z * sigma * np.sqrt(T)) / (S0**2 * sigma**2 * T)
    discount = np.exp(-r * T)

    gamma_paths = discount * payoff * weight
    gamma = gamma_paths.mean()
    gamma_se = gamma_paths.std(ddof=1) / np.sqrt(n_paths)

    return {"gamma": gamma, "gamma_se": gamma_se}


def black_scholes_gamma(S0, K, r, sigma, T):
   
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return {"gamma": gamma}



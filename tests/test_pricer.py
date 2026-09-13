import numpy as np
import pytest
from src.simulator import simulate_gbm_paths
from src.pricer import (
    price_european_option, black_scholes_price,
    price_asian_option, price_barrier_option
)
from src.variance_reduction import price_with_antithetic
from src.greeks import (
    pathwise_delta_vega_call, black_scholes_delta_vega,
    pathwise_theta_rho_call, black_scholes_theta_rho,
    lrm_gamma_call, black_scholes_gamma
)

#  test parameters
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
n_steps = 252
N_LARGE = 100_000


def test_gbm_paths_start_at_S0():
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, 100, seed=1)
    assert np.allclose(paths[:, 0], S0)


def test_gbm_paths_are_positive():
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, 1000, seed=1)
    assert np.all(paths > 0)


def test_gbm_mean_matches_theory():
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, N_LARGE, seed=42)
    simulated_mean = paths[:, -1].mean()
    theoretical_mean = S0 * np.exp(r * T)
    assert abs(simulated_mean - theoretical_mean) / theoretical_mean < 0.01


def test_european_call_matches_black_scholes():
    mc = price_european_option(S0, K, r, sigma, T, n_steps, N_LARGE, 'call', seed=42)
    bs = black_scholes_price(S0, K, r, sigma, T, 'call')
    assert abs(mc['price'] - bs) < 3 * mc['std_error']


def test_european_put_matches_black_scholes():
    mc = price_european_option(S0, K, r, sigma, T, n_steps, N_LARGE, 'put', seed=42)
    bs = black_scholes_price(S0, K, r, sigma, T, 'put')
    assert abs(mc['price'] - bs) < 3 * mc['std_error']


def test_antithetic_reduces_variance():
    std_mc = price_european_option(S0, K, r, sigma, T, n_steps, 20000, 'call', seed=1)
    anti = price_with_antithetic(S0, K, r, sigma, T, n_steps, 10000, 'call', seed=1)
    assert anti['std_error'] <= std_mc['std_error']


def test_asian_cheaper_than_vanilla():
    vanilla = price_european_option(S0, K, r, sigma, T, n_steps, N_LARGE, 'call', seed=42)
    asian = price_asian_option(S0, K, r, sigma, T, n_steps, N_LARGE, 'call', seed=42)
    assert asian['price'] < vanilla['price']


def test_barrier_cheaper_than_vanilla():
    vanilla = price_european_option(S0, K, r, sigma, T, n_steps, N_LARGE, 'call', seed=42)
    barrier = price_barrier_option(S0, K, 130, r, sigma, T, n_steps, N_LARGE, 'up-and-out', seed=42)
    assert barrier['price'] < vanilla['price']


def test_delta_vega_match_black_scholes():
    mc = pathwise_delta_vega_call(S0, K, r, sigma, T, n_steps, N_LARGE, seed=42)
    bs = black_scholes_delta_vega(S0, K, r, sigma, T)
    assert abs(mc['delta'] - bs['delta']) < 3 * mc['delta_se']
    assert abs(mc['vega'] - bs['vega']) < 3 * mc['vega_se']


def test_theta_rho_match_black_scholes():
    mc = pathwise_theta_rho_call(S0, K, r, sigma, T, n_steps, N_LARGE, seed=42)
    bs = black_scholes_theta_rho(S0, K, r, sigma, T)
    assert abs(mc['theta'] - bs['theta']) < 3 * mc['theta_se']
    assert abs(mc['rho'] - bs['rho']) < 3 * mc['rho_se']


def test_gamma_matches_black_scholes():
    mc = lrm_gamma_call(S0, K, r, sigma, T, n_steps, 500_000, seed=42)
    bs = black_scholes_gamma(S0, K, r, sigma, T)
    assert abs(mc['gamma'] - bs['gamma']) < 3 * mc['gamma_se']
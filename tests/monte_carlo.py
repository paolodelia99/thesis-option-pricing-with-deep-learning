import numpy as np

from pricing_models.monte_carlo import mc_pricing_european, mc_pricing_american
from pricing_models.bsm import geo_paths

S, X_1, X_2 = 100, 110, 90
T = 0.166667
r, q = 0.025, 0
sigma = 0.2
N = 50
seed = 42

np.random.seed(seed)


def test_call_am_pricing():
    prices = geo_paths(S, T, r, q, sigma, 60, N)
    call = mc_pricing_american(prices.T, X_1, T, r, N, 'C')
    assert call == 1.6879781310787911


def test_call_eu_pricing():
    prices = geo_paths(S, T, r, q, sigma, 60, N)
    call = mc_pricing_european(prices.T, X_1, T, r, 'C')
    assert call == 0.028155447487949032


def test_compare_am_eu_call_pricing():
    prices = geo_paths(S, T, r, q, sigma, 60, N)
    am_call = mc_pricing_american(prices.T, X_1, T, r, N, 'C')
    eu_call = mc_pricing_european(prices.T, X_1, T, r, 'C')
    assert am_call > eu_call


def test_compare_am_eu_put_pricing():
    prices = geo_paths(S, T, r, q, sigma, 60, N)
    am_put = mc_pricing_american(prices.T, X_2, T, r, N, 'P')
    eu_put = mc_pricing_european(prices.T, X_2, T, r, 'P')
    assert am_put > eu_put

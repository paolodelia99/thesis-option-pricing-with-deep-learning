from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from numba import jit, int32, float32, float64

from pricing_models.bomp import binom_price_tree, binom_option_tree_c, binom_option_tree_p
from pricing_models.bsm import geo_paths
from pricing_models.heston import generate_heston_path
from pricing_models.monte_carlo import mc_pricing_american
from pricing_models.trinomial_tree import t_price_tree, t_option_tree_c, t_option_tree_p


def params_ranges(S,
                  strike_step: float = 1.0,
                  vol_range: Tuple[float, float, float] = (0.02, 1.02, 0.02),
                  interest_range: Tuple[float, float, float] = (0.002, 0.1, 0.01),
                  tau_range: Tuple[float, float, float] = (0.15, 1.1, 0.02)) -> Tuple[
    np.array, np.array, np.array, np.array]:
    """
    Generate ranges for strike prices, volatility, interest rates and time to expiration

    :param strike_step:
    :param S: the underlying price
    :param vol_range: (start_vol, end_vol, step_vol) for the volatility range
    :param interest_range: (start_vol, end_vol, step_vol) for the volatility range
    :param tau_range: (start_vol, end_vol, step_vol) for the volatility range
    :return: (strike_range, volatility_range, interest_rates_range, time_to_expiration_range)
    """
    strikes = np.arange(S // 2, S + (S // 2) + 1, strike_step)
    vols = np.arange(vol_range[0], vol_range[1], vol_range[2])
    interests = np.arange(interest_range[0], interest_range[1], interest_range[2])
    taus = np.arange(tau_range[0], tau_range[1], tau_range[2])

    return strikes, vols, interests, taus


@jit(float64[:, :](float32, float32, float32, float32), nopython=True, fastmath=True)
def generate_binom_option_chain_c(S, T, r, sigma) -> np.array:
    """
    Generate the option chain using the binomial tree model given the price of the underlying at time 0, the time to
    expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :return: pandas DataFrame containing the option chain
    """
    n = int(np.floor(365 * T))
    strikes = np.arange((2 * S) // 3, ((S * 3) // 2) + 1, 1.0)

    delta_t = T / n
    up = np.exp(sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - (1 / up)) / (up - (1 / up))
    price_tree = binom_price_tree(S, n, up)
    option_chain = np.zeros((len(strikes), 7))

    for i, strike in enumerate(strikes):
        price = binom_option_tree_c(price_tree, strike, n, delta_t, r, p)[0, 0]
        option_chain[i] = np.array([S, strike, 1, sigma, r, T, price])

    return option_chain


@jit(float64[:, :](float32, float32, float32, float32), nopython=True, fastmath=True)
def generate_binom_option_chain_p(S, T, r, sigma) -> np.array:
    """
    Generate the option chain using the binomial tree model given the price of the underlying at time 0, the time to
    expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :return: pandas DataFrame containing the option chain
    """
    n = int(np.floor(365 * T))
    strikes = np.arange((2 * S) // 3, ((S * 3) // 2) + 1, 1.0)

    delta_t = T / n
    up = np.exp(sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - (1 / up)) / (up - (1 / up))
    price_tree = binom_price_tree(S, n, up)
    option_chain = np.zeros((len(strikes), 7))

    for i, strike in enumerate(strikes):
        price = binom_option_tree_p(price_tree, strike, n, delta_t, r, p)[0, 0]
        option_chain[i] = np.array([S, strike, 1, sigma, r, T, price])

    return option_chain


def binom_option_data(S,
                      type_: str,
                      strike_step: float = 1.0,
                      vol_range: Tuple[float, float, float] = (0.02, 1.02, 0.02),
                      interest_range: Tuple[float, float, float] = (0.01, 0.11, 0.01),
                      tau_range: Tuple[float, float, float] = (0.15, 1.1, 0.02)
                      ) -> pd.DataFrame:
    """
    Generate the option data using the binomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param type_: option's type: 'C' for a call 'P' for a put
    :param strike_step:
    :param S: the underlying price
    :param vol_range: (start_vol, end_vol, step_vol) for the volatility range
    :param interest_range: (start_vol, end_vol, step_vol) for the volatility range
    :param tau_range: (start_vol, end_vol, step_vol) for the volatility range
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    chains = []
    prices = np.arange((S // 4) * 3, (S // 4) * 5, strike_step)

    for price in prices:
        _, vols, rs, taus = params_ranges(
            price,
            strike_step=2.5,
            vol_range=vol_range,
            interest_range=interest_range,
            tau_range=tau_range
        )
        for v, r, t in product(vols, rs, taus):
            if type_ == 'C':
                opt_chain = generate_binom_option_chain_c(price, v, r, t)
            else:
                opt_chain = generate_binom_option_chain_p(price, v, r, t)
            chains.append(opt_chain)

    option_data = pd.DataFrame(np.concatenate(chains),
                               columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration',
                                        'Option Price'])
    option_data['Type'] = type_
    return option_data


@jit(float64[:, :](float32, float32, float32, float32), nopython=True, fastmath=True)
def generate_trinom_option_chain_c(S, T, r, sigma) -> pd.DataFrame:
    """
    Generate the option chain using the trinomial tree model given the price of the underlying at time 0, the time to
    expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option chain
    """
    n = int(np.floor(365 * T))
    strikes = np.arange((2 * S) // 3, ((S * 3) // 2) + 1, 1.0)

    delta_t = T / n
    up = np.exp(sigma * np.sqrt(2 * delta_t))
    down = 1 / up
    p_up = ((np.exp(r * delta_t / 2) - np.exp(-sigma * np.sqrt(delta_t / 2))) / (
            np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
    p_down = ((np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(r * (delta_t / 2))) / (
            np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
    p_mid = 1 - (p_up + p_down)
    price_tree = t_price_tree(S, n, up, down)
    option_chain = np.zeros((len(strikes), 7))

    for i, strike in enumerate(strikes):
        price = t_option_tree_c(price_tree, strike, n, delta_t, r, p_up, p_down, p_mid)[0, 0]
        option_chain[i] = np.array([S, strike, 1, sigma, r, T, price])

    return option_chain


@jit(float64[:, :](float32, float32, float32, float32), nopython=True, fastmath=True)
def generate_trinom_option_chain_p(S, T, r, sigma) -> pd.DataFrame:
    """
    Generate the option chain using the trinomial tree model given the price of the underlying at time 0, the time to
    expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option chain
    """
    n = int(np.floor(365 * T))
    strikes = np.arange((2 * S) // 3, ((S * 3) // 2) + 1, 1.0)

    delta_t = T / n
    up = np.exp(sigma * np.sqrt(2 * delta_t))
    down = 1 / up
    p_up = ((np.exp(r * delta_t / 2) - np.exp(-sigma * np.sqrt(delta_t / 2))) / (
            np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
    p_down = ((np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(r * (delta_t / 2))) / (
            np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
    p_mid = 1 - (p_up + p_down)
    price_tree = t_price_tree(S, n, up, down)
    option_chain = np.zeros((len(strikes), 7))

    for i, strike in enumerate(strikes):
        price = t_option_tree_p(price_tree, strike, n, delta_t, r, p_up, p_down, p_mid)[0, 0]
        option_chain[i] = np.array([S, strike, 1, sigma, r, T, price])

    return option_chain


def trinomial_option_data(S,
                          type_: str,
                          strike_step: float = 1.0,
                          vol_range: Tuple[float, float, float] = (0.02, 1.02, 0.02),
                          interest_range: Tuple[float, float, float] = (0.01, 0.11, 0.01),
                          tau_range: Tuple[float, float, float] = (0.15, 1.1, 0.02)
                          ) -> pd.DataFrame:
    """
    Generate the option data using the trinomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param type_: option's type: 'C' for a call 'P' for a put
    :param strike_step:
    :param S: the underlying price
    :param vol_range: (start_vol, end_vol, step_vol) for the volatility range
    :param interest_range: (start_vol, end_vol, step_vol) for the volatility range
    :param tau_range: (start_vol, end_vol, step_vol) for the volatility range
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    chains = []
    prices = np.arange((S // 4) * 3, (S // 4) * 5, strike_step)

    for price in prices:
        _, vols, rs, taus = params_ranges(
            price,
            strike_step=2.5,
            vol_range=vol_range,
            interest_range=interest_range,
            tau_range=tau_range
        )
        for v, r, t in product(vols, rs, taus):
            if type_ == 'C':
                opt_chain = generate_trinom_option_chain_c(price, v, r, t)
            else:
                opt_chain = generate_trinom_option_chain_p(price, v, r, t)
            chains.append(opt_chain)

    option_data = pd.DataFrame(np.concatenate(chains),
                               columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration',
                                        'Option Price'])
    option_data['Type'] = type_
    return option_data


def mc_option_chain_geo(S, T, r, sigma, n: int, type_: str) -> pd.DataFrame:
    """
    Generate the option chain using the monte carlo simulation of geometric Brownian motion given the price of the
    underlying at time 0, the time to expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :param n: number of simulation
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option chain
    """
    strikes = np.arange(S // 2, S * 2 + 1, 1)
    option_chain = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])
    steps = int(np.round(T * 365))
    paths = geo_paths(S, T, r, 0, sigma, steps, n)

    for strike in strikes:
        opt_price = mc_pricing_american(paths.T, strike, T, r, steps, type_)
        option_chain = option_chain.append({
            'Price': S,
            'Strike': strike,
            'Type': type_,
            'Vol': sigma,
            'Interest Rate': r,
            'Time to Expiration': T,
            'Option Price': opt_price
        }, ignore_index=True)

    return option_chain


def mc_option_chain_heston(S, T, r, kappa,
                           theta, v_0, rho, xi, n: int, type_: str) -> pd.DataFrame:
    """
    Generate the option chain using the monte carlo simulation of heston paths given the price of the
    underlying at time 0, the time to expiration, the interest free rate and other params specif for the heston model

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param kappa:
    :param theta:
    :param v_0:
    :param rho:
    :param xi:
    :param n: number of simulations
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option chain
    """
    strikes = np.arange(S // 2, S * 2 + 1, 1)
    option_chain = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Kappa', 'Theta', 'V0', 'Rho', 'Xi', 'Interest Rate', 'Time to Expiration',
                 'Option Price'])
    paths = generate_heston_path(S, T, r, kappa, theta, v_0, rho, xi, int(np.round(T * 365)), n)

    for strike in strikes:
        opt_price = mc_pricing_american(paths, strike, T, r, n, type_)
        option_chain = option_chain.append({
            'Price': S,
            'Strike': strike,
            'Type': type_,
            'Kappa': kappa,
            'Theta': theta,
            'Rho': rho,
            'V0': v_0,
            'Xi': xi,
            'Interest Rate': r,
            'Time to Expiration': T,
            'Option Price': opt_price
        }, ignore_index=True)

    return option_chain


def mc_option_data_geo(
        S,
        n: int,
        type_: str,
        strike_step: float = 1.0,
        vol_range: Tuple[float, float, float] = (0.1, 1.1, 0.1),
        interest_range: Tuple[float, float, float] = (0.01, 0.11, 0.01),
        tau_range: Tuple[float, float, float] = (0.15, 1.1, 0.02)):
    """
    Generate the option data using the trinomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param n: number of simulations
    :param type_: option's type: 'C' for a call 'P' for a put
    :param vol_range: (start_vol, end_vol, step_vol) for the volatility range
    :param interest_range: (start_vol, end_vol, step_vol) for the volatility range
    :param tau_range: (start_vol, end_vol, step_vol) for the volatility range
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    strikes, vols, interests, taus = params_ranges(S,
                                                   vol_range=vol_range,
                                                   interest_range=interest_range,
                                                   tau_range=tau_range)
    option_data = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])

    for tau, r, sigma in product(taus, interests, vols):
        opt_chain = mc_option_chain_geo(S, tau, r, sigma, n, type_)
        option_data = option_data.append(opt_chain, ignore_index=True)

    return option_data

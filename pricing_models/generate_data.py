from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from pricing_models.bomp import bomp
from pricing_models.bsm import geo_paths
from pricing_models.heston import generate_heston_path
from pricing_models.monte_carlo import mc_pricing_american
from pricing_models.trinomial_tree import topm


def params_ranges(S,
                  vol_range: Tuple[float, float, float] = (0.02, 1.02, 0.02),
                  interest_range: Tuple[float, float, float] = (0.002, 0.1, 0.01),
                  tau_range: Tuple[float, float, float] = (0.15, 1.1, 0.02)) -> Tuple[
    np.array, np.array, np.array, np.array]:
    """
    Generate ranges for strike prices, volatility, interest rates and time to expiration

    :param S: the underlying price
    :param vol_range: (start_vol, end_vol, step_vol) for the volatility range
    :param interest_range: (start_vol, end_vol, step_vol) for the volatility range
    :param tau_range: (start_vol, end_vol, step_vol) for the volatility range
    :return: (strike_range, volatility_range, interest_rates_range, time_to_expiration_range)
    """
    strikes = np.arange(S - S // 3, S * 2 + 1, 1)
    vols = np.arange(vol_range[0], vol_range[1], vol_range[2])
    interests = np.arange(interest_range[0], interest_range[1], interest_range[2])
    taus = np.arange(tau_range[0], tau_range[1], tau_range[2])

    return strikes, vols, interests, taus


def generate_binom_option_chain(S, T, r, sigma, type_: str) -> pd.DataFrame:
    """
    Generate the option chain using the binomial tree model given the price of the underlying at time 0, the time to
    expiration, the interest free rate and the underlying volatility

    :param S: underlying price at time t=0
    :param T: time to expiration (in years)
    :param r: interest free rate
    :param sigma: underlying volatility
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option chain
    """
    strikes = np.arange(S // 2, S * 2 + 1, 0.5)
    option_chain = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])

    for strike in strikes:
        price = bomp(S, strike, T, r, sigma, int(np.floor(365 * T)), type_)
        option_chain = option_chain.append(
            {'Price': S, 'Strike': strike, 'Type': type_, 'Vol': sigma, 'Interest Rate': r, 'Time to Expiration': T,
             'Option Price': price}, ignore_index=True)

    return option_chain


def binom_option_data(S, type_: str) -> pd.DataFrame:
    """
    Generate the option data using the binomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    strikes, vols, interests, taus = params_ranges(S)
    option_data = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])

    for X, sigma, r, T in product(strikes, vols, interests, taus):
        price = bomp(S, X, T, r, sigma, int(np.floor(365 * T)), type_)
        option_data = option_data.append(
            {'Price': S, 'Strike': X, 'Type': type_, 'Vol': sigma, 'Interest Rate': r, 'Time to Expiration': T,
             'Option Price': price}, ignore_index=True)

    return option_data


def trinomial_option_data(S, type_: str) -> pd.DataFrame:
    """
    Generate the option data using the trinomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    strikes, vols, interests, taus = params_ranges(S)
    option_data = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])

    for X, sigma, r, T in product(strikes, vols, interests, taus):
        price = topm(S, X, T, r, sigma, int(np.floor(365 * T)), type_)
        option_data = option_data.append(
            {'Price': S, 'Strike': X, 'Type': type_, 'Vol': sigma, 'Interest Rate': r, 'Time to Expiration': T,
             'Option Price': price}, ignore_index=True)

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


def mc_option_data_geo(S, n: int, type_: str):
    """
    Generate the option data using the trinomial tree model given the price of the underlying at time 0

    :param S: underlying price at time t=0
    :param n: number of simulations
    :param type_: option's type: 'C' for a call 'P' for a put
    :return: pandas DataFrame containing the option data for different ranges  of params
    """
    strikes, vols, interests, taus = params_ranges(S,
                                                   interest_range=(0.025, 0.03, 0.05),
                                                   vol_range=(0.1, 1.0, 0.1))
    option_data = pd.DataFrame(
        columns=['Price', 'Strike', 'Type', 'Vol', 'Interest Rate', 'Time to Expiration', 'Option Price'])

    for tau, r, sigma in product(taus, interests, vols):
        opt_chain = mc_option_chain_geo(S, tau, r, sigma, n, type_)
        option_data = option_data.append(opt_chain, ignore_index=True)

    return option_data

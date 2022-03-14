from typing import Tuple

import numpy as np
from scipy import stats


def get_d1_d2(S, X, T, t, r, sigma) -> Tuple[np.single, np.single]:
    """
    Compute d1 and d2 values for the black-scholes pricing model


    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate
    :param sigma: underlying volatility
    :return: (d1, d2)
    """
    d1 = (np.log(S / X) + (r + sigma * sigma / 2.) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    return d1, d2


def black_scholes(S, X, T, t, r, sigma, o_type: str = "C") -> np.single:
    """
    Compute option price using the black-scholes model

    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate (in percentual)
    :param sigma: underlying volatility
    :param o_type: option type, "C" for a call option and "P" for a put option
    :return: the black-scholes option price
    """
    d1, d2 = get_d1_d2(S, X, T, t, r, sigma)
    if o_type == "C":
        return S * stats.norm.cdf(d1, 0, 1) - X * np.exp(-r * (T - t)) * stats.norm.cdf(d2, 0, 1)
    else:
        return X * np.exp(-r * (T - t)) * stats.norm.cdf(-d2, 0, 1) - S * stats.norm.cdf(-d1, 0, 1)


def delta(S, X, T, t, r, sigma, o_type: str = "C") -> np.single:
    """
    Compute option's delta

    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate (in percentual)
    :param sigma: underlying volatility
    :param o_type: option type, "C" for a call option and "P" for a put option
    :return: the option's delta
    """
    d1, _ = get_d1_d2(S, X, T, t, r, sigma)
    if o_type == "C":
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1) - 1


def gamma(S, X, T, t, r, sigma) -> np.single:
    """
    Compute the option's gamma

    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate (in percentual)
    :param sigma: underlying volatility
    :return: the option's gamma
    """
    d1, _ = get_d1_d2(S, X, T, t, r, sigma)
    return stats.norm.pdf(d1) / (S * sigma * np.sqrt(T - t))


def vega(S, X, T, t, r, sigma):
    """
    Compute the option's vega

    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate (in percentual)
    :param sigma: underlying volatility
    :return: the option's vega
    """
    d1, _ = get_d1_d2(S, X, T, t, r, sigma)
    return S * stats.norm.pdf(d1) * np.sqrt(T - t)


def theta(S, X, T, t, r, sigma, o_type: str = "C") -> np.single:
    """
    Compute the option's theta

    :param S: underlying price
    :param X: option's strike price
    :param T: option's time to maturity (in years)
    :param t: current time (in years)
    :param r: interest rate (in percentual)
    :param sigma: underlying volatility
    :param o_type: option type, "C" for a call option and "P" for a put option
    :return: the option's theta
    """
    d1, d2 = get_d1_d2(S, X, T, t, r, sigma)
    tmp_1 = - (S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T - t))
    if o_type == "C":
        return tmp_1 - r * X * np.exp(-r * (T - t)) * stats.norm.cdf(d2)
    else:
        return tmp_1 + r * X * np.exp(-r * (T - t)) * stats.norm.cdf(-d2)

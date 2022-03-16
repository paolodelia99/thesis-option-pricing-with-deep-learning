import numpy as np


def mc_pricing_european(prices: np.array, X, T, r, type_="C") -> np.single:
    """
    Compute the european option price given the prices path

    :param prices: array containing the last prices for the simulated paths
    :param X: option's strike price
    :param T: time to maturity (in years)
    :param r: interest rate
    :param type_: option's type
    :return:
    """
    if type_ == "C":
        payoffs = np.maximum(prices - X, 0)
        return np.mean(payoffs) * np.exp(-r * T)
    elif type_ == "P":
        payoffs = np.maximum(X - prices, 0)
        return np.mean(payoffs) * np.exp(-r * T)
    else:
        raise ValueError("type_ must be 'put' or 'call'")


def mc_pricing_american(prices: np.array, X, T, r, type_="C") -> np.single:
    """
    Compute the american option price given the prices path

    :param prices: array containing the last prices for the simulated paths
    :param X: option's strike price
    :param T: time to maturity (in years)
    :param r: interest rate
    :param type_: option's type
    :return:
    """
    pass
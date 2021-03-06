import numpy as np

from numba import jit, int32, float32, float64


@jit(float64[:, :](float32, int32, float32), nopython=True, fastmath=True)
def binom_price_tree(S: np.single, n: np.int32, up: np.single) -> np.array:
    """
    Compute binomial price tree

    :param S: initial underlying price
    :param n: height of the binomial tre
    :param up: up factor
    :return lower triangular matrix (n+1 x n+1) that contains the binomial price tree
    """
    prices = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(i + 1):
            prices[i, j] = S * (up ** (i - j)) * ((1 / up) ** j)

    return prices


@jit(float64[:, :](float64[:, :], float32, int32, float32, float32, float32), nopython=True, fastmath=True)
def binom_option_tree_c(prices: np.array, X: np.single, n: np.int32, delta_t: np.single, r: np.single,
                        p: np.single) -> np.array:
    """

    :param prices: binomial price tree
    :param X: option's strike price
    :param n: height of the binomial tree
    :param delta_t: T / n
    :param r: interest free rate
    :param p: probability of going up
    :return: lower triangular matrix (n+1 x n+1) that contains the option price at each node of the binomial tre
    """
    option_p = np.zeros((n + 1, n + 1))
    option_p[n, :] = np.maximum(np.zeros(n + 1), (prices[n, :] - X))

    for i in range(n - 1, -1, -1):
        for j in range(0, i + 1):
            option_p[i, j] = np.exp(-r * delta_t) * (p * option_p[i + 1, j] + (1 - p) * option_p[i + 1, j + 1])

    return option_p


@jit(float32(float32, float32, float32, float32, float32, int32), nopython=True, fastmath=True)
def bomp_c(S, X, T, r, sigma, n: np.int32) -> np.single:
    """
    Compute the call american option price using the binomial option pricing model

    :param S: underlying price
    :param X: option's strike price
    :param T: time to maturity
    :param r: annual interest rate
    :param sigma: underlying volatility
    :param n: height of the binomial tree
    :return: np.single which is the price of the american option
    """
    delta_t = T / n
    up = np.exp(sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - (1 / up)) / (up - (1 / up))

    prices = binom_price_tree(S, n, up)

    option_p = binom_option_tree_c(prices=prices, X=X, n=n, delta_t=delta_t, r=r, p=p)

    return option_p[0, 0]


@jit(float64[:, :](float64[:, :], float32, int32, float32, float32, float32), nopython=True, fastmath=True)
def binom_option_tree_p(prices: np.array, X: np.single, n: np.int32, delta_t: np.single, r: np.single,
                        p: np.single) -> np.array:
    """

    :param prices: binomial price tree
    :param X: option's strike price
    :param n: height of the binomial tree
    :param delta_t: T / n
    :param r: interest free rate
    :param p: probability of going up
    :return: lower triangular matrix (n+1 x n+1) that contains the option price at each node of the binomial tree
    """
    option_p = np.zeros((n + 1, n + 1))
    option_p[n, :] = np.maximum(np.zeros(n + 1), (X - prices[n, :]))

    for i in range(n - 1, -1, -1):
        for j in range(0, i + 1):
            option_p[i, j] = np.exp(-r * delta_t) * (p * option_p[i + 1, j] + (1 - p) * option_p[i + 1, j + 1])

    return option_p


@jit(float32(float32, float32, float32, float32, float32, int32), nopython=True, fastmath=True)
def bomp_p(S, X, T, r, sigma, n: np.int32) -> np.single:
    """
    Compute the put american option price using the binomial option pricing model

    :param S: underlying price
    :param X: option's strike price
    :param T: time to maturity
    :param r: annual interest rate
    :param sigma: underlying volatility
    :param n: height of the binomial tree
    :return: np.single which is the price of the american option
    """
    delta_t = T / n
    up = np.exp(sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - (1 / up)) / (up - (1 / up))

    prices = binom_price_tree(S, n, up)

    option_p = binom_option_tree_p(prices=prices, X=X, n=n, delta_t=delta_t, r=r, p=p)

    return option_p[0, 0]

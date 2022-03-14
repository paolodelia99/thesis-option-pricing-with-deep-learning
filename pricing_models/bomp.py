import numpy as np


def bomp(S, X, T, r, sigma, n: np.int, o_type: str = "C"):
    """
    Compute the option price using the binomial option pricing model

    :param S: underlying price
    :param X: option's strike price
    :param T: time to maturity
    :param r: annual interest rate
    :param sigma: underlying volatility
    :param n: height of the binomial tree
    :param o_type: option's type, "C" for a call option "P" for a put option
    :return: lower triangular matrix (n+1)x(n+1) that contains the option price at each node of the binomial tree
    """
    delta_t = T / n
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1 / up
    p = (np.exp(r * delta_t) - down) / (up - down)
    prices = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(i + 1):
            prices[i, j] = S * (up ** (i - j)) * ((1 / up) ** j)

    option_p = np.zeros((n + 1, n + 1))
    if o_type == "C":
        option_p[n, :] = np.maximum(np.zeros(n + 1), (prices[n, :] - X))
    else:
        option_p[n, :] = np.maximum(np.zeros(n + 1), (X - prices[n, :]))

    for i in range(n - 1, -1, -1):
        for j in range(0, i + 1):
            option_p[i, j] = np.exp(-r * delta_t) * (p * option_p[i + 1, j] + (1 - p) * option_p[i + 1, j + 1])

    return option_p

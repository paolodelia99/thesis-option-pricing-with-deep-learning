import numpy as np

exp = np.exp
sqrt = np.sqrt


def t_price_tree(S0, n, up, down):
    """

    :param S0:
    :param n:
    :param up:
    :param down:
    :return:
    """
    prices = np.zeros((n + 1, n * 2 + 1))

    for i in range(n + 1):
        for j in range(i * 2 + 1):
            prices[i, j] = S0 * (up ** i) * (down ** j)

    return prices


def t_option_tree(prices: np.array, X: np.single, n, delta_t, r, p_up, p_down, p_mid, type_: str):
    """

    :param prices:
    :param X:
    :param n:
    :param delta_t:
    :param r:
    :param p_up:
    :param p_down:
    :param p_mid:
    :param type_:
    :return:
    """
    option_p = np.zeros((n + 1, n * 2 + 1))

    if type_ == "C":
        option_p[n, :] = np.maximum(np.zeros(n * 2 + 1), (prices[n, :] - X))
    else:
        option_p[n, :] = np.maximum(np.zeros(n * 2 + 1), (X - prices[n, :]))

    for i in range(n - 1, -1, -1):
        for j in range(0, i * 2 + 1):
            # option_p[i, j] = np.exp(-r * delta_t) * (
            #         p_up * option_p[i + 1, j + 1] + p_down * option_p[i + 1, j - 1] + p_mid * option_p[i + 1, j])
            if type_ == 'C':
                option_p[i, j] = np.maximum(prices[i, j] - X, np.exp(-r * delta_t) * (
                        p_up * option_p[i + 1, j] + p_down * option_p[i + 1, j + 2] + p_mid * option_p[i + 1, j + 1]))
            else:
                option_p[i, j] = np.maximum(X - prices[i, j], np.exp(-r * delta_t) * (
                        p_up * option_p[i + 1, j] + p_down * option_p[i + 1, j + 2] + p_mid * option_p[i + 1, j + 1]))

    return option_p


def topm(S, X, T, r, sigma, n: np.int, type_: str = "C"):
    """
    Compute the american option price using the trinomial option pricing model

    :param S: underlying price
    :param X: option's strike price
    :param T: time to maturity
    :param r: annual interest rate
    :param sigma: underlying volatility
    :param n: height of the trinomial tree
    :param type_: option's type, "C" for a call option "P" for a put option
    :return: np.single which is the price of the american option
    """
    delta_t = T / n
    up = exp(sigma * np.sqrt(2 * delta_t))
    down = 1 / up
    m = 1
    p_up = ((exp(r * delta_t / 2) - exp(-sigma * sqrt(delta_t / 2))) / (
            exp(sigma * sqrt(delta_t / 2)) - exp(-sigma * sqrt(delta_t / 2)))) ** 2
    p_down = ((exp(sigma * sqrt(delta_t / 2)) - exp(r * (delta_t / 2))) / (
            exp(sigma * sqrt(delta_t / 2)) - exp(-sigma * sqrt(delta_t / 2)))) ** 2
    p_mid = 1 - (p_up + p_down)

    prices = t_price_tree(S0=S, n=n, up=up, down=down)

    option_p_t = t_option_tree(prices, X, n, delta_t, r, p_up, p_down, p_mid, type_)

    return option_p_t[0, 0]

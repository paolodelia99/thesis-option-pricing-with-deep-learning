import numpy as np


def generate_heston_path(S, T, r, kappa, theta, v_0, rho, xi,
                         steps, n_paths, return_vol: bool = False):
    """
    
    :param S: 
    :param T: 
    :param r: 
    :param kappa: 
    :param theta: 
    :param v_0: 
    :param rho: 
    :param xi: 
    :param steps: 
    :param n_paths: 
    :param return_vol: 
    :return: 
    """
    dt = T / steps
    size = (n_paths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0

    for t in range(steps):
        mu = np.array([0, 0])
        cov = np.array([[1, rho],
                        [rho, 1]])
        WT = np.random.multivariate_normal(mu, cov=cov, size=n_paths) * np.sqrt(dt)
        S_t = S_t * (np.exp((r - v_t / 2) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs

    return prices

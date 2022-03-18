import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


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


def process_ith_day(i, prices, strike, reg, r_daily, T, type_: str = 'C'):
    def update_fv(row):
        if row['if_execute']:
            return row['execution_value'] * (1 + r_daily) ** (T - i)
        else:
            return row['option_value_at_maturity']

    day_i = prices[[i, 'option_value_at_maturity']]

    if type_ == 'C':
        day_i = day_i[day_i[i] > strike]
    else:
        day_i = day_i[day_i[i] < strike]

    if day_i.shape[0] == 0:
        return

    reg.fit(day_i.values, day_i['option_value_at_maturity'])

    day_i['expected_continuation_value'] = reg.predict(day_i.values) * (1 / (1 + r_daily))

    if type_ == 'C':
        day_i['execution_value'] = prices[i].apply(lambda x: max(0, x - strike))
    else:
        day_i['execution_value'] = prices[i].apply(lambda x: max(0, strike - x))

    day_i['if_execute'] = day_i['execution_value'] > day_i['expected_continuation_value']
    day_i['updated_option_value_at_maturity'] = day_i.apply(update_fv, axis=1)
    prices.loc[day_i.index, 'option_value_at_maturity'] = day_i['updated_option_value_at_maturity']
    prices.loc[day_i[day_i['if_execute'] == True].index, 'executed_on'] = i


def mc_pricing_american(prices: np.array, X, T, r, N, type_="C") -> np.single:
    """
    Compute the american option price given the prices path

    :param prices: array containing the last prices for the simulated paths
    :param X: option's strike price
    :param T: time to maturity (in years)
    :param r: interest rate
    :param type_: option's type
    :return:
    """
    r_daily = (r * T) / N
    reg = LinearRegression()

    prices_df = pd.DataFrame(prices)
    prices_df['payoff_at_maturity'] = prices_df[N-1] - X
    prices_df['payoff_at_maturity'] = prices_df['payoff_at_maturity'].apply(lambda x: max(x, 0))
    prices_df['option_value_at_maturity'] = prices_df['payoff_at_maturity']
    prices_df['executed_on'] = N -1

    for day in range(60 - 1, 0, -1):
        process_ith_day(day, prices_df, X, reg, r_daily, T, type_)

    return prices_df['option_value_at_maturity'].mean() * (1+r_daily)**(-0.16666)

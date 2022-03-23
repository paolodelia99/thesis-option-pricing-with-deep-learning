import argparse

from pricing_models.generate_data import binom_option_data, trinomial_option_data, mc_option_data_geo

S = 100

if __name__ == '__main__':
    binom_synthetic_calls = binom_option_data(
        S,
        'C',
        vol_range=(0.05, 1.05, 0.05),
        interest_range=(0.01, 0.11, 0.01),
        tau_range=(0.1, 1.1, 0.1)
    )
    binom_synthetic_calls.to_csv('data/binom_synthetic_calls.csv')
    del binom_synthetic_calls

    binom_synthetic_puts = binom_option_data(
        S,
        'P',
        vol_range=(0.05, 1.05, 0.05),
        interest_range=(0.01, 0.11, 0.01),
        tau_range=(0.1, 1.1, 0.1)
    )
    binom_synthetic_puts.to_csv('data/binom_synthetic_puts.csv')

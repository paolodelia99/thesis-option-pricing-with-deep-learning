import argparse

from pricing_models.generate_data import binom_option_data, trinomial_option_data, mc_option_data_geo

S = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str,
                        help='type of data to generate: binomial (bin), trinomial (tri), montecarlo(mc)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.type == 'bin':
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
    elif args.type == 'tri':
        trinomial_synthetic_calls = trinomial_option_data(
            S,
            'C',
            vol_range=(0.05, 1.05, 0.05),
            interest_range=(0.01, 0.11, 0.01),
            tau_range=(0.1, 1.1, 0.1)
        )
        trinomial_synthetic_calls.to_csv('data/trinomial_synthetic_calls.csv')
        del trinomial_synthetic_calls

        trinomial_synthetic_puts = trinomial_option_data(
            S,
            'P',
            vol_range=(0.05, 1.05, 0.05),
            interest_range=(0.01, 0.11, 0.01),
            tau_range=(0.1, 1.1, 0.1)
        )
        trinomial_synthetic_puts.to_csv('data/trinomial_synthetic_puts.csv')
    else:
        mc_synthetic_calls = mc_option_data_geo(
            S,
            30,
            'C',
            vol_range=(0.05, 1.05, 0.05),
            interest_range=(0.01, 0.11, 0.01),
            tau_range=(0.1, 1.1, 0.1)
        )
        mc_synthetic_calls.to_csv('data/mc_synthetic_calls.csv')
        del mc_synthetic_calls

        mc_synthetic_puts = mc_option_data_geo(
            S,
            30,
            'P',
            vol_range=(0.05, 1.05, 0.05),
            interest_range=(0.01, 0.11, 0.01),
            tau_range=(0.1, 1.1, 0.1)
        )
        mc_synthetic_puts.to_csv('data/mc_synthetic_puts.csv')

# Option pricing with Deep Learning

Option pricing has always been an hard problem in computational finance. With 
my Bachelor Thesis Project I've tackled this particular problem with the use of 
Neural Networks. 

Neural Nets are known to be universal function approximators, so given enought data
they could, theoretically, approximate anykind of function. Why we cannot apply the same reasoning 
also to the world of Finance. Well, that's what I've tried to do.

My works is focused on American Option, because the fact that they can be expired
anytime before the time to expiration makes them trickier to price respect to the Europeans one.

## Content

- [Option & the Greeks](notebooks/Options.ipynb)
- [Option's trading strategies](notebooks/Options-Strategies.ipynb)
- Pricing models
  - [Binomial Model](pricing_models/bomp.py)
  - [Trinomial Model](pricing_models/trinomial_tree.py)
  - [Heston model with LSMC](notebooks/tff-lsmc-option-generator-heston-calls.ipynb)
- HPO notebooks
  - [Bayesian Optimization Binomial/Trinomial dataset](notebooks/AX-HPO-binomial-trinomial.ipynb)
  - [Bayesian Optimization Heston dataset](notebooks/AX-HPO-heston.ipynb)
  - [Bayesian Optimization on real option data]
- [IV Surface Notebook](notebooks/Finding_IV_Heston.ipynb)
- Greeks Notebooks
  - [Binomial/Trinomial Model Greeks](notebooks/Finding_the_option's_greeks_BinTri.ipynb)
  - [Heston Model Greeks](notebooks/Finding_Greeks_with_Autodiff_Heston.ipynb)


## Quickstart

Clone the repo

    git clone https://github.com/paolodelia99/thesis-option-pricing-with-deep-learning.git



# Author 

Paolo D'Elia

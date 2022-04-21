# Option pricing with Deep Learning

Option pricing has always been an hard problem in computational finance. With 
my Bachelor Thesis Project I've tackled this particular problem with the use of 
Neural Networks. 

Neural Nets are known to be universal function approximators, so given enought data
they could, theoretically, approximate anykind of function. Why we cannot apply the same reasoning 
also to the world of Finance. Well, that's what I've tried to do.

My works is focused on American Option, becuase the fact that they can be expired
anytime before the time to expiration makes them trickier to price respect to the Europeans one.

## Content

- [Option & the Greeks notebook](notebooks/Options.ipynb)
- Pricing models
  - Binomial Model
  - Trinomial Model
  - [Heston model](notebooks/tff-lsmc-option-generator-heston-calls.ipynb)
- HPO notebooks
  - Bayesian Optimization Binomial/Trinomial dataset
  - Bayesian Optimization Heston dataset

## Quickstart

Clone the repo

    git clone https://github.com/paolodelia99/thesis-option-pricing-with-deep-learning.git



# Author 

Paolo D'Elia

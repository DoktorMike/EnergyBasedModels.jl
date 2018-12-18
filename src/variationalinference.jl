
"""
    kldivergence(p, q)

Calculates the Kullback Leibler divergence between two vectors representing
samples for distributions `p` and `q`.
"""
kldivergence(p, q) = sum(@. p * log(p / q))

"""
    kldivergence_qp(μ, logσ)

Calculates Kullback Leibler divergence between a diagonal multivariate normal
distribution `q` and a N(0, 1) distribution `p`. The parameters `μ` and `logσ` comes
from the `q` distribution.
"""
kldivergence_qp(μ, logσ) = 0.5sum( @. exp(logσ)^2 + μ^2 - 2logσ - 1)


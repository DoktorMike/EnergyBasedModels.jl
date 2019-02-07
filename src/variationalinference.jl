"""
    z(μ, logσ)

Draw a sample from the parameterized Gaussian `q` distribution by using the
reparameterization trick.
"""
z(μ, logσ) = μ + exp(logσ)*randn()

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


"""
    elbo_bbvi(logpxz, λ)

Calculates the Evience Lower Bound for a probabilistic model. The inputs are the
log probability of that model as well as the variational parameters which in this case
is just a tuple of vectors (μ, logσ) which parameterizes the variational guide
distribution.
"""
function elbo_bbvi(logpxz, λ, nsamples::Int)
    μ, logσ = λ[1], λ[2]
    e = 0
    for s in 1:nsamples
        zₛ = z.(μ, logσ)
        e += logpxz(zₛ) - kldivergence_qp(λ[1], λ[2])
    end
    e/nsamples
end


"""
    bbvi(logpz, logz, q)

Performs vanilla Black Box Variational Inference on the model represented by logpz.
"""
function bbvi(logpxz, logqz, qz, x, y)
    S = 1 # Number of samples

    avggrad = 0
    for s in 1:S

        lp = logpxz(y, ) # Evaluate model which resamples the network
        θ = params(logpxz)
        avggrad += 1
    end

end



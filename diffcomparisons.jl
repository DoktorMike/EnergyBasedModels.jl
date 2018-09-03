using ReverseDiff
using ForwardDiff
using Zygote
using Plots
using Distributions

unicodeplots()

rng = Distributions.Gaussian(0, 0.01)

# Function definitions
f(x) = 3*x^2-4*x

fgzg(x) = Zygote.derivative(f, x) 
fgfd(x) = ForwardDiff.derivative(f, x) 


# Simple test
a = -10:0.01:10
scatter(a, f.(a))
scatter!(a, fgzg.(a))
scatter!(a, fgfd.(a))


function zygotederivtest(nhidden=100_000, ninput=1_000)
    rng = Distributions.Gaussian(0, 0.01)
    #g(x) = Zygote.derivative(f, x) 

    # Test case with single layer 10000 hidden nodes 100 inputs
    W = rand(rng, nhidden, ninput)
    b = rand(rng, nhidden)
    f(x) = tanh(sum(W*x .+ b))

    x = rand(rng, ninput)
    g = Zygote.gradient(() -> f(x), Zygote.Params([W, b]))
    g[W], g[b]
end

function forwarddiffderivtest(nhidden=100_000, ninput=1_000)
    rng = Distributions.Gaussian(0, 0.01)

    W = rand(rng, nhidden, ninput)
    b = rand(rng, nhidden)
    θ = [W, b]
    x = rand(rng, ninput)
    f(θ) = tanh(sum(θ[1]*x .+ θ[2]))
    g(θ) = ForwardDiff.derivative(f, θ)

    g(θ)
end


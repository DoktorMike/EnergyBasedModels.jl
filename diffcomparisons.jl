using ReverseDiff
using ForwardDiff
using Zygote
using Distributions
using Flux
using Flux.Tracker


function zygotederivtest(nhidden=100_000, ninput=1_000)
    #rng = Distributions.Gaussian(0, 0.01)
    #g(x) = Zygote.derivative(f, x) 

    # Test case with single layer 10000 hidden nodes 100 inputs
    W = rand(nhidden, ninput) .- 0.5
    b = rand(nhidden) .- 0.5
    f(x) = tanh(sum(W*x .+ b))

    x = rand(ninput)
    g = Zygote.gradient(() -> f(x), Zygote.Params([W, b]))
    g[W], g[b]
end

function forwarddiffderivtest(nhidden=100_000, ninput=1_000)
    W = rand(nhidden, ninput) .- 0.5
    b = rand(nhidden) .- 0.5
    θ = vcat([W[x] for x in eachindex(W)], [b[x] for x in eachindex(b)])
    x = rand(ninput)
    indstop = nhidden*ninput
    f(θ) = tanh(sum(reshape(θ[1:indstop], nhidden, ninput)*x .+ θ[1+indstop:end]))
    ForwardDiff.gradient(f, θ)
end

function fluxderivtest(nhidden=100_000, ninput=1_000)
    W = param(rand(nhidden, ninput) .- 0.5)
    b = param(rand(nhidden) .- 0.5)
    x = rand(ninput)
    f(x) = tanh(sum(W*x .+ b))
    g = Tracker.gradient(() -> f(x), Params([W, b]))
    g[W], g[b]
end

function reversediffderivtest(nhidden=100_000, ninput=1_000)
    W = rand(nhidden, ninput) .- 0.5
    b = rand(nhidden) .- 0.5
    x = rand(ninput)
    f(W, b) = tanh(sum(W*x .+ b))
    ReverseDiff.gradient(f, (W, b))
end

function yaadderivtest(nhidden=100_000, ninput=1_000)
    W = Variable(rand(nhidden, ninput) .- 0.5)
    b = Variable(rand(nhidden) .- 0.5)
    x = rand(ninput)
    f(W, b) = tanh(sum(W*x .+ b))
    y = tr(f(W, b))
    backward(y)
end

W = Variable(rand(3, 2))
b = Variable(rand(3))
x = randn(2)


# Isolated

G(p) = prod(p)
G([1.5,1.0,3.0])

using ForwardDiff
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])

using Calculus
res3 = Calculus.gradient(G,[1.5,1.0,3.0])

using Flux
res4 = Flux.Tracker.gradient(G,[1.5,1.0,3.0])[1]

using ReverseDiff
res5 = ReverseDiff.gradient(G,[1.5,1.0,3.0])


# Gradients of distributions

#using ForwardDiff: gradient
using Flux
using Flux.Tracker
using Distributions
using Flux.Tracker: gradient
using Distributions: Gaussian, logpdf, gradlogpdf, MultivariateNormal

function logprob(x)
    a = Gaussian(0, 1)
    return logpdf.(a, x)
end

function logprob2(x, z)
    a = Gaussian(z[1], z[2])
    return logpdf.(a, x)
end

μ = [1, 1]
Σ = Float64[i==j ? rand() : 0 for i in 1:2, j in 1:2]
pz = MultivariateNormal(μ, Σ) 
                                      
gradient(x -> logprob2(x, [2,3]), 1)


# Multiple arguments taking gradients
φ(x, ω)=tanh(x'*ω)
φ([1,2,1,1], [0.1,-0.2,0.3,-0.4])
gradient(φ, [1,2,1], [1,1,1])


struct Affine{F, S, T}
    W::S
    b::T
    φ::F
end

function Affine(in::Integer, out::Integer; initW=randn, initb=randn)
    Affine(initW(out, in), initb(out), identity)
end

function Affine(in::Integer, out::Integer, φ::Function; initW=randn, initb=randn)
    Affine(initW(out, in), initb(out), φ)
end

function (a::Affine)(X::AbstractArray)
    W, b, φ = a.W, a.b, a.φ
    φ.(W*X .+ b)
end

l = Affine(10, 5)
l(rand(10, 40))

layers = [Affine(10, 5, x->1/(1+exp(-x))), 
          Affine(5, 3), 
          x->tanh.(x)]
foldl((x, m)->m(x), layers, init=randn(10))


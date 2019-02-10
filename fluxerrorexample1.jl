using Flux
using Flux.Tracker
using Distributions: Bernoulli, logpdf

mutable struct AffineB{F, S, T}
    W::S
    b::T

    μ::T
    logσ::T

    φ::F
end

z(μ, logσ) = μ + exp(logσ)*randn()

function initaffineb(in::Integer, out::Integer, μ, logσ)
    s = z.(μ, logσ)
    W, b = reshape(s[1:out*in], out, in), reshape(s[out*in+1:end], out)
    W, b
end

function initaffineb(in::Integer, out::Integer)
    μ, logσ = param(randn(out*in+out)), param(rand(out*in+out))
    W, b = initaffineb(in, out, μ, logσ)
    W, b, μ, logσ
end

function AffineB(in::Integer, out::Integer)
    W, b, μ, logσ = initaffineb(in, out)
    AffineB(W, b, μ, logσ, identity)
end

function AffineB(in::Integer, out::Integer, μ, logσ)
    W, b = initaffineb(in, out, μ, logσ)
    AffineB(W, b, μ, logσ, identity)
end

function (a::AffineB)(X::AbstractArray)
    W, b, φ = a.W, a.b, a.φ
    φ.(W*X .+ b)
end

function resample!(a::Chain)
    for i in 1:length(a)
        resample!(a[i])
    end
end

function resample!(a::AffineB)
    out, in = size(a.W)
    a.W, a.b = initaffineb(in, out, a.μ, a.logσ)
end

resample!(a::Any) = a

Flux.@treelike AffineB # This should allow me to call params on an AffineB type.

# Simple XOR logic problem
x = [0 0 1 1; 0 1 0 1]
y = [0 1 1 0]

#b = Chain(AffineB(2, 2))
#σ.(b(x)) # Works fine

a = Chain(AffineB(2, 2), x->σ.(x), AffineB(2, 1), x->σ.(x))
#a(x) # Dies

#b = Chain(Dense(2, 2, σ), Dense(2, 1, σ))
pars = params(a)
#a = AffineB(2, 1, pars[1], pars[2]) # This breaks
#a = AffineB(2, 1, a.μ, a.logσ) # This works

#loss(ŷ, y) = sum((ŷ.-y).^2)
loss(ŷ, y) = -sum(logpdf.(Bernoulli.(ŷ), y))

for b in 1:10000
    l = loss(a(x), y)
    Tracker.back!(l)
    for p in pars
        p.data .-= 0.1 .* Tracker.data(p.grad)
        Tracker.tracker(p).grad .= 0;
    end
    resample!(a)
end

@show loss(a(x), y)

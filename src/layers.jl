using Flux

"""
Structure representing an Affine transformation layer.
"""
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

Flux.@treelike Affine


### BAYES ###

z(μ, logσ) = μ + exp(logσ)*randn()

struct AffineB{F, S, T}
    W::S
    b::T

    μ::T
    logσ::T

    φ::F
end

function AffineB(in::Integer, out::Integer)
    μ = zeros(out*in+out)
    logσ = zeros(out*in+out)
    s = z.(μ, logσ)
    W, b = reshape(s[1:out*in], out, in), reshape(s[out*in+1:end], out)
    AffineB(W, b, μ, logσ, identity)
end


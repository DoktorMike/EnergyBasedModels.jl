using Flux: params, param

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

"""
    z(μ, logσ)

Draw a sample from the parameterized Gaussian `q` distribution by using the
reparameterization trick.
"""
z(μ, logσ) = μ + exp(logσ)*randn()

"""
Structure representing a Bayesian Affine transformation layer.
"""
struct AffineB{F, S, T}
    W::S
    b::T

    μ::T
    logσ::T

    φ::F
end

"""
    initaffineb(in::Integer, out::Integer)

Helper function for initializing a Bayesian Affine layers parameters.
"""
function initaffineb(in::Integer, out::Integer)
    μ = zeros(out*in+out)
    logσ = zeros(out*in+out)
    s = z.(μ, logσ)
    W, b = reshape(s[1:out*in], out, in), reshape(s[out*in+1:end], out)
    W, b, μ, logσ
end

function AffineB(in::Integer, out::Integer)
    W, b, μ, logσ = initaffineb(in, out)
    AffineB(param(W), param(b), param(μ), param(logσ), identity)
end

function AffineB(in::Integer, out::Integer, φ::Function)
    W, b, μ, logσ = initaffineb(in, out)
    AffineB(param(W), param(b), param(μ), param(logσ), φ)
end

function (a::AffineB)(X::AbstractArray)
    W, b, φ = a.W, a.b, a.φ
    φ.(W*X .+ b)
end

function resample!(a::AffineB)
    μ, logσ = a.μ, a.logσ
    index = 0
    for p ∈ params(a)
        ω = z.(μ[index+1:length(p)], logσ[index+1:length(p)])
        println(size(ω))
        println(size(p))
        p.data .= reshape(ω, size(p))
        index += length(p)
    end
end

Flux.@treelike AffineB

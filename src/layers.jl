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
    initaffineb(in::Integer, out::Integer, μ, logσ)

Helper function for initializing a Bayesian Affine layer where the variational
parameters `μ` and `logσ` are given.
"""
function initaffineb(in::Integer, out::Integer, μ, logσ)
    s = z.(μ, logσ)
    W, b = reshape(s[1:out*in], out, in), reshape(s[out*in+1:end], out)
    W, b
end

"""
    initaffineb(in::Integer, out::Integer)

Helper function for initializing a Bayesian Affine layers parameters.
"""
function initaffineb(in::Integer, out::Integer)
    μ, logσ = param(zeros(out*in+out)), param(zeros(out*in+out))
    W, b = initaffineb(in, out, μ, logσ)
    W, b, μ, logσ
end

function AffineB(in::Integer, out::Integer)
    W, b, μ, logσ = initaffineb(in, out)
    AffineB(W, b, μ, logσ, identity)
end

function AffineB(in::Integer, out::Integer, φ::Function)
    W, b, μ, logσ = initaffineb(in, out)
    AffineB(W, b, μ, logσ, φ)
end

function AffineB(in::Integer, out::Integer, μ, logσ)
    W, b = initaffineb(in, out, μ, logσ)
    AffineB(W, b, μ, logσ, identity)
end

function AffineB(in::Integer, out::Integer, φ::Function, μ, logσ)
    W, b = initaffineb(in, out, μ, logσ)
    AffineB(W, b, μ, logσ, φ)
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

function resample(a::AffineB)
    out, in = size(a.W)
    AffineB(in, out, a.μ, a.logσ)
end

Flux.@treelike AffineB

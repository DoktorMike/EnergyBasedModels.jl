using Pkg
Pkg.activate(".")
import EnergyBasedModels

using Distributions
using Plots
using UnicodePlots

println("Verify that the world still works");
a = rand(Distributions.Gaussian(10, 5), 100);
b = rand(Distributions.Gaussian(10, 5), 100);


unicodeplots()
plot(x=a, y=b)


struct Observation
    x::Vector
    y::Vector
end


# Model

y = Distributions.Gaussian(10, 5)


using Flux.Tracker: gradient, param
using Flux

a = param(1)
b = param([1,0,1])
c(x)=b'*x + a

x1 = [2,10,100]
c(x1)



# Simple Gradient Test Explicit parameters

using Flux
using Flux.Tracker: gradient

m(x, θ) = θ[1]*x[1]+ θ[2]*x[2]
loss(x, y, θ) = sum((y-m(x, θ))^2)

g(x, y, θ) = gradient(θ -> loss(x,y,θ), θ) # Only tracking θ
g([1,1], 1, [1,1])
g(x, y, θ) = gradient((x,y,θ) -> loss(x,y,θ), x,y,θ) # Tracking x, y and θ
g([1,1], 1, [1,1])


# Simple Gradient Test Implicit parameters but global

using Flux
using Flux.Tracker: gradient

θ = param([1, 1])
m(x) = θ[1]*x[1]+ θ[2]*x[2]
loss(x, y) = sum((y-m(x))^2)

g(x, y) = gradient(() -> loss(x,y), Params(θ)) # Only tracking θ
gs = g([1,1], 1)


# Simple Gradient Test Implicit parameters locally in object

using Flux
using Flux: params, @treelike
using Flux.Tracker: gradient, param

struct Model
    ω
    b
end

Model(in::Int, out::Int) = Model(param(rand(out, in)), param(rand(out)))
(m::Model)(x) = m.ω*x .+ m.b

@treelike Model

loss(y, ŷ) = sum((y .- ŷ).^2)

# Define data
x = [[1, 1] [2,2] [3,3]]
y = [1, 2, 3]

# Instantiate model
model = Model(2, 1)
θ = params(model) # theta consist of [ω, b]
gs = gradient(() -> loss(y, model(x)), Flux.Params(θ))

# Calculate loss
loss(y, model(x))

# Update model parameters
θ[1].data .+= -0.001 .* Flux.data(gs[θ[1]])
θ[2].data .+= -0.001 .* Flux.data(gs[θ[2]])

# Calculate loss
loss(y, model(x))


# Simple Gradient Test Implicit parameters locally in object using forward and
# back!

using Flux
using Flux: params, @treelike
using Flux.Tracker: gradient, param, back, back!

struct Model
    ω
    b
end

Model(in::Int, out::Int) = Model(param(rand(out, in)), param(rand(out)))
(m::Model)(x) = m.ω*x .+ m.b

@treelike Model

loss(y, ŷ) = sum((y .- ŷ).^2)

# Define data
x = [[1, 1] [2,2] [3,3]]
y = [1, 2, 3]

# Instantiate model
model = Model(2, 1)

# Calculate loss
l = loss(y, model(x))

θ = params(model) # theta consist of [ω, b]
back!(l)

# Update model parameters
θ[1].data .+= -0.001 .* θ[1].grad
θ[2].data .+= -0.001 .* θ[2].grad

# Calculate loss
l = loss(y, model(x))


# Simple Gradient Test Implicit parameters locally in object using forward and
# back

using Flux
using Flux: params, @treelike
using Flux.Tracker: gradient, param, forward, Params

struct Model
    ω
    b
end

Model(in::Int, out::Int) = Model(param(rand(out, in)), param(rand(out)))
(m::Model)(x) = m.ω*x .+ m.b

@treelike Model

loss(y, ŷ) = sum((y .- ŷ).^2)

# Define data
x = [[1, 1] [2,2] [3,3]]
y = [1, 2, 3]

# Instantiate model
model = Model(2, 1)
θ = params(model) # theta consist of [ω, b]

# Calculate loss via a forward pass and generate the backwards function
l, back = forward(() -> loss(y, model(x)), Params(θ))

# Calculate gradients via the back function with sensitivity 1
grads = back(1)

# Update model parameters
θ[1].data .+= -0.001 .* Flux.data(grads[θ[1]])
θ[2].data .+= -0.001 .* Flux.data(grads[θ[2]])

# Calculate loss again and see improvement
l = loss(y, model(x))


# Standard example from Flux documentation

using Flux
using Flux.Tracker
using Flux.Tracker: update!

W, b = param(2), param(3)

predict(x) = W*x + b
loss(x, y) = sum((y - predict(x))^2)

x, y = 4, 15
pars = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), pars)

update!(W, -0.1*grads[W])
loss(x, y)

# Second example from Flux documentation

using Flux
using Flux.Tracker
using Flux.Tracker: update!

W, b = param(rand(2, 5)), param(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((y .- predict(x)).^2)

x, y = rand(5), rand(2) # Dummy data
pars = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), pars)

update!(W, -0.1*grads[W])
loss(x, y)


# Random variable case: Doesn't update the Distributions!!!

using Flux
using Flux.Tracker
using Flux.Tracker: update!
using Distributions: Gaussian, logpdf

μ, logσ = param(rand(2)), param(rand(2))
rvs = [Gaussian(μ[i], exp(logσ[i])) for i in 1:length(μ)]
loss(y) = sum(logpdf.(rvs, y))

y = [1, 1]
pars = Params([μ, logσ])
grads = Tracker.gradient(() -> loss(y), pars)

loss(y)
update!(μ, 0.1*grads[μ])
update!(logσ, 0.1*grads[logσ])
rvs = [Gaussian(μ[i], exp(logσ[i])) for i in 1:length(μ)] # Why do I need to recreate this?
loss(y)

# Playing

struct Neuron{T, P, R}
    lif::T
    threshold::P
    f::R
end

Neuron() = Neuron(0, 0, identity)

neurons = [Neuron() for i in 1:1e5]
conns = randn(length(neurons), length(neurons))


# Zygote again

using Zygote
using Base:zero

struct Affine
    W
    b
end

Affine(in::Int, out::Int) = Affine(randn(out, in), randn(out))

(a::Affine)(x) = a.W*x .+ a.b
params(a::Affine) = [a.W, a.b]
zero(a::Affine) = Affine(zeros(out,in), zeros(randn(out)))

# Simple model
m = Affine(3, 2)
m(randn(3))
g = Zygote.gradient(()->sum(m(randn(3))), Zygote.Params([m.W, m.b]))
g[m.W], g[m.b]
Zygote.refresh() # Kill all gradients etc

# Chained model
layers = [Affine(3, 2), Affine(2, 1)]
model(x) = foldl((xx,m)->m(xx), layers, init=x)
g = Zygote.gradient(()->sum(model(randn(3))), Zygote.Params(map(params, layers)))
g = Zygote.gradient(()->sum(model(randn(3))), Zygote.Params([layers[1].W, layers[1].b, layers[2].W, layers[2].b]))
g = Zygote.gradient(()->sum(model(randn(3))), Zygote.Params([layers[1].W, layers[1].b]))


# FLUX TRACKER

using Flux
using Flux.Tracker
using DataFrames

struct Affine
    W
    b
end

Affine(in::Int, out::Int) = Affine(param(randn(out, in)), param(randn(out)))
(a::Affine)(x) = a.W*x .+ a.b
Flux.@treelike Affine

# Model Definition
layers = [Affine(2,2), Affine(2,1)]
model(x) = foldl((x,m) -> m(x), layers, init=x)
loss(y, ŷ) = sum((y.-ŷ).^2)
mygrad(y, ŷ) = gradient(()->loss(y, ŷ), Params(params(model)))

# Data
xordf = DataFrame(x1=[0,0,1,1], x2=[0,1,0,1], y=[0,1,1,1])
xormat = convert(Matrix, xordf)

# Gradient
#loss([1 1 1 1 1; 1 1 1 1 1], model(randn(3, 5)))
loss(xormat'[3,:], model(xormat'[1:2,:]))


# ____  _     _      ____               _ 
#|  _ \(_)___| |_   / ___|_ __ __ _  __| |
#| | | | / __| __| | |  _| '__/ _` |/ _` |
#| |_| | \__ \ |_  | |_| | | | (_| | (_| |
#|____/|_|___/\__|  \____|_|  \__,_|\__,_|
#                                         

# Gradients with respect to parameters
using Flux.Tracker
using Distributions: Gaussian, logpdf
θ = param(randn(2))
y = 0
ŷ = rand(Gaussian(θ[1], exp(θ[2])))
l = (ŷ-y)^2
Tracker.back!(l)
Flux.Tracker.zero_grad!(θ)

θ = param(randn(2))
y = 0
l = logpdf(Gaussian(θ[1], exp(θ[2])), y)
Tracker.back!(l)


# ____  _            _    _                __     _____ 
#| __ )| | __ _  ___| | _| |__   _____  __ \ \   / /_ _|
#|  _ \| |/ _` |/ __| |/ / '_ \ / _ \ \/ /  \ \ / / | | 
#| |_) | | (_| | (__|   <| |_) | (_) >  <    \ V /  | | 
#|____/|_|\__,_|\___|_|\_\_.__/ \___/_/\_\    \_/  |___|
#                                                       

# Using
using Distributions: Gaussian, logpdf, Bernoulli, MvNormal
using Flux.Tracker
using DataFrames

# Helpers
softmax(x) = exp.(x) ./ sum(exp.(x))
softplus(x) = log(1 + exp(x))
sigmoid(x) = 1 / (1 + exp(-x))
t(μ, logσ) = μ .+ softplus.(logσ).*randn(length(μ))
logq(θ, λ) = sum(logpdf.(Gaussian.(λ[1], softplus.(λ[2])), θ))
logp(θ) = sum(logpdf.(Gaussian(0, 1), θ))
mae(y, ŷ) = sum(abs.(y .- ŷ))/length(y)
sse(y, ŷ) = sum((y .- ŷ).^2)

function elbo(logpyz, y, x, logqz, λ, nsamples)
    e = 0
    for i in 1:nsamples
        z = t(λ[1], λ[2])
        e += logpyz(y,x,z) - logqz(z, λ)
    end
    e/nsamples
end

# Define model

function unpackpars(θ, in, out)
    offset = out*in
    reshape(θ[1:offset], out, in), θ[(offset+1):end]
end

function f(x, θ)
    ω = unpackpars(θ, 2, 1)
    sigmoid.(ω[1]*x .+ ω[2])
end

function logjointprob(y, x, θ)
    sum(logpdf.(Bernoulli.(f(x, θ)), y)) + logp(θ)
end

# Init λ parameters of the guides distributions
out, in = 1, 2
λ = Tracker.param(randn(out*in+out)), Tracker.param(rand(out*in+out).*-1)
θ = t(λ[1], λ[2]); # Resample parameters
#θ = Tracker.param(randn(out*in+out))

x = [0 0 1 1
     0 1 0 1];
y = [0 1 1 1];

N = 100
perfdf = DataFrame(Loss=rand(N), LogPrior=rand(N), LogLikelihood=rand(N),
                   LogQ=rand(N), Performance=rand(N))
# Main loop
for i in 1:N
    #θ = t(λ[1].data, λ[2].data); # Resample parameters
    #l = -elbo(logjointprob, y, x, logq, λ, 1)
    θ = t(λ[1], λ[2]); # Resample parameters
    l = -(sum(logpdf.(Bernoulli.(f(x, θ)), y)) + 0.1.*(logp(θ) - logq(θ, λ)))           # Does work
    Tracker.back!(l);
    for p in λ
        p.data .-= 0.1 .* Tracker.data(p.grad);
        Tracker.tracker(p).grad .= 0;
    end
    lp = logp(copy(θ.data))
    lq = logq(θ.data, (λ[1].data, λ[2].data))
    ll = sum(logpdf.(Bernoulli.(f(x, θ.data)), y))
    err = mae(f(x, θ.data)[:], y[:])
    perfdf[i, :Loss] = l.data 
    perfdf[i, :LogPrior] = lp
    perfdf[i, :LogLikelihood] = ll
    perfdf[i, :LogQ] = lq
    perfdf[i, :Performance] = err
    println("Performance: $err");
    #l = -(sum(logpdf.(Bernoulli.(f(x, θ)), y)))           # Does work
    #l = -(sum(logpdf.(Bernoulli.(f(x, θ)), y)) + logp(θ)) # Does not work

    # Backpropagate and update parameter and zero gradients
    #θ.data .-= 0.01 .* Tracker.data(θ.grad)
    #λ[1].data .-= 0.01 .* Tracker.data(λ[1].grad);
    #λ[2].data .-= 0.01 .* Tracker.data(λ[2].grad);
    #Tracker.tracker(λ[1]).grad .= 0;
    #Tracker.tracker(λ[2]).grad .= 0;
end

using UnicodePlots

lineplot(perfdf[:, :Loss], width=80)
lineplot(perfdf[:, :LogLikelihood], width=80, name="Likelihood")
lineplot(perfdf[:, :LogPrior], width=80, name="Prior")
lineplot(perfdf[:, :LogQ], width=80, name="Q")
lineplot(perfdf[:, :Performance], width=80, name="MAE")

# _____ _              ____             ___ 
#|  ___| |_   ___  __ | __ ) _   _  __ |__ \
#| |_  | | | | \ \/ / |  _ \| | | |/ _` |/ /
#|  _| | | |_| |>  <  | |_) | |_| | (_| |_| 
#|_|   |_|\__,_/_/\_\ |____/ \__,_|\__, (_) 
#                                  |___/    

using Flux
using Flux.Tracker

struct AffineB{F, S, T}
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

function (a::AffineB)(X::AbstractArray)
    W, b, φ = a.W, a.b, a.φ
    φ.(W*X .+ b)
end

Flux.@treelike AffineB # This should allow me to call params on an AffineB type.

# Simple OR logic problem
x = [0 0 1 1; 0 1 0 1]
y = [0 1 1 1]
a = AffineB(2, 1)

l = sum((a(x) .- y).^2) # Sum squared error
pars = params(a) # You have to collect the parameters BEFORE you do back! or it dies..
Tracker.back!(l) # Backpropagate

for p in pars
    p.data .-= 0.01 .* Tracker.data(p.grad)
    Tracker.tracker(p).grad .= 0;
end

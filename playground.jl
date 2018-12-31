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


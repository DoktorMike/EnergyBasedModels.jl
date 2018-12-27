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


# Simple Gradient Test Implicit parameters

using Flux
using Flux.Tracker

θ = param([1, 1])
m(x) = θ[1]*x[1]+ θ[2]*x[2]
loss(x, y) = sum((y-m(x))^2)

g(x, y) = gradient(() -> loss(x,y), Params(θ)) # Only tracking θ
g([1,1], 1)
g(x, y, θ) = gradient((x,y) -> loss(x,y), x,y,θ) # Tracking x, y and θ
g([1,1], 1, [1,1])

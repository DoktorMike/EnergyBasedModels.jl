using Distributions
using Plots

println("Verify that the world still works");
a = rand(Distributions.Gaussian(10, 5), 100);
b = rand(Distributions.Gaussian(10, 5), 100);

plot(x=a, y=b)


struct Observation
    x::Vector
    y::Vector
end


# Model

y = Distributions.Gaussian(10, 5)

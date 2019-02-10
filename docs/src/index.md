# Energy Based Models

## Functions

```@autodocs
Modules = [EnergyBasedModels, ]
Order   = [:function, :type]
```

## Examples

```@example
import Random # hide
Random.seed!(1) # hide
A = rand(3, 3)
b = [1, 2, 3]
A \ b
```

```@example
using EnergyBasedModels: Affine

l = Affine(10, 5)
l(rand(10, 40))

layers = [Affine(10, 5, x->1/(1+exp(-x))), 
          Affine(5, 3), 
          x->tanh.(x)]
foldl((x, m)->m(x), layers, init=randn(10))
```

We can also use fully Bayesian layers like this.

```@example
using EnergyBasedModels: AffineB, Affine, z, resample!, resample
using Flux: param, params, Params
using Flux.Tracker

x = [0 0 1 1; 0 1 0 1]
y = [0 1 1 1]
a = AffineB(2, 1)
pars = params(a)

for i in 1:10
    a = AffineB(2, 1, pars[1], pars[2])
    l = sum((a(x) .- y).^2)
    Tracker.back!(l)
    for p in pars
        p.data .-= 0.1 .* Tracker.data(p.grad);
        Tracker.tracker(p).grad .= 0;
    end
end

#a = resample(a)

```

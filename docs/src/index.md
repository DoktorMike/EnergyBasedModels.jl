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
using EnergyBasedModels: AffineB, Affine, z, resample!
using Flux: param, params, Params

a = AffineB(2, 1)
x = rand(2)
a(x)

resample!(a)
a(x)

```

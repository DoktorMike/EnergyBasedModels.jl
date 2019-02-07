### Broadcast error

This does not work.

```julia
using EnergyBasedModels: AffineB, Affine, z, resample!
using Flux: param, params, Params

a = AffineB(2, 1)
x = rand(2)
a(x)
```

Because this

```julia
identity.(a.W*[1,1] .+ a.b)
```

does not work. This, however


```julia
identity(a.W*[1,1] .+ a.b)
```

does.

module EnergyBasedModels

using Distributions
using Flux

include("layers.jl")
include("variationalinference.jl")

greet() = print("Hello World!")

end # module

using Test
using EnergyBasedModels: Affine
using Random

println("Testing...")

@testset "Affine layer test" begin
    Random.seed!(1)
    layers = [Affine(10, 5, x->tanh.(x)), 
              Affine(5, 3),
              x->tanh.(x)]
    o = foldl((x, m) -> m(x), layers, init=rand(10))
    @test o ≈ [-0.3766950303416259, 0.9987795785957421, 0.027565971701823188]
end


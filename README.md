# Implementation for the paper "Multivariate Dependencies in Multiplex Graphs: Theory and Estimation" by Skeja and Olhede (Skeja and Olhede, 2024)

## Overview
This implementation estimates and quantifies **multivariate dependencies in multiplex graphs** as developed by *Skeja and Olhede (2024)*.  
The estimation now operates **directly at the level of the probability cells**, using the latent blockmodel inference procedure of `NetworkHistogram.jl`.

---

## Run information measures on your own adjacency matrices (A1.csv, A2.csv, A3.csv)
```julia
using DelimitedFiles, LinearAlgebra, Random
include("multiplex_info.jl")       
Random.seed!(1234)                

load_csv(path) = Matrix{Int}(readdlm(path, ',', Int))

symmetrize01(A) = begin
    B = A .> 0                      
    B = max.(B, B')                  
    fill!(view(B, diagind(B)), 0)    
    Matrix{Int}(B)                  
end

same_size(As...) = begin
    n = size(As[1],1)
    @assert all(size(A) == (n,n) for A in As) "All layers must be n×n with the same n"
    n
end

A1 = symmetrize01(load_csv("A1.csv"))
A2 = symmetrize01(load_csv("A2.csv"))
A3 = symmetrize01(load_csv("A3.csv"))
n  = same_size(A1, A2, A3)


##Blockmodel approximation
### Acknowledgments
This step relies on `NetworkHistogram.jl` (Dufour and Grainger, 2023), using the inference implementation of Dufour and Olhede (2024).

using NetworkHistogram

estimator, history = graphhist(cat(A1, A2, A3, dims=3);
      starting_assignment_rule = OrderedStart(),
      maxitr = 1_000_000,
      stop_rule = PreviousBestValue(100))

# Construct the probability-cell tensor directly from the fitted estimator.
# Each slice Pcells[:, :, ℓ] corresponds to one of the 2^d (here 8) joint edge configurations.
Pcells = build_Pcells(estimator)

##Compute graphon information measures

tri, mi_12, mi_23, mi_13, cmi_12_3, cmi_13_2, cmi_23_1 =
    info_measures_from_cells(n, Pcells)

println("I(1;2;3)   = ", tri)
println("I(1;2)     = ", mi_12)
println("I(2;3)     = ", mi_23)
println("I(1;3)     = ", mi_13)
println("I(1;2 | 3) = ", cmi_12_3)
println("I(1;3 | 2) = ", cmi_13_2)
println("I(2;3 | 1) = ", cmi_23_1)


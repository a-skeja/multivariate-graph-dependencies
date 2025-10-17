# Implementation for the paper "Multivariate Dependencies in Multiplex Graphs: Theory and Estimation" by Skeja and Olhede (Skeja and Olhede, 2024)

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
```
## Blockmodel approximation 
```julia
### Acknowledgments
This step relies on `NetworkHistogram.jl` (Dufour and Grainger, 2023), using the implementation of Dufour and Olhede (2024).

out = graphhist(cat(A1, A2, A3, dims=3);
      starting_assignment_rule = OrderedStart(),
      maxitr = 1_000_000,
      stop_rule = PreviousBestValue(100))

    
est = out.graphhist

p   = NetworkHistogram.get_moment_representation(est)[1]

p_1, p_2, p_12 = p[:, :, 1], p[:, :, 2], p[:, :, 3]

p_3, p_13, p_23, p_123 = p[:, :, 4], p[:, :, 5], p[:, :, 6], p[:, :, 7]
```

## Compute graphon information measures; see multiplex_info.jl

```julia

I123, I12, I23, I13, I12_3, I13_2, I23_1 =
    graphon_info_measures(n, est, p_1, p_2, p_3, p_12, p_13, p_23, p_123)
```
## Print results

```julia

println("I(1;2;3)   = ", I123)
println("I(1;2)     = ", I12)
println("I(2;3)     = ", I23)
println("I(1;3)     = ", I13)
println("I(1;2 | 3) = ", I12_3)
println("I(1;3 | 2) = ", I13_2)
println("I(2;3 | 1) = ", I23_1)
```

## References
 Skeja, A., and Olhede, S. C. (2024). Quantifying multivariate graph dependencies: Theory and estimation for multiplex graphs. arXiv preprint arXiv:2405.14482.
 
 Dufour, C., and Olhede, S. C. (2024). Inference for decorated graphs and application to multiplex networks. arXiv preprint arXiv:2408.12339.
 
 Dufour, C., and Grainger, J. (2023). NetworkHistogram.jl (v0.5.1). Zenodo. https://doi.org/10.5281/zenodo.10212852

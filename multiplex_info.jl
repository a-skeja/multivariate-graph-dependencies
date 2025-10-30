
############################
#Implementation for the paper Multivariate Dependencies in Multiplex Graphs:Theory and Estimation by Skeja and Olhede (Skeja and Olhede,2025)#
############################
# Install the NetworkHistogram package from Dufour and Grainger (2023), that implements the code of Dufour and Olhede (2024)
# ---------- Environment ----------
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
for pkg in ("LaTeXStrings","Distributions","SpecialFunctions","Roots","QuadGK","Plots","NetworkHistogram")
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using LinearAlgebra, Statistics, Random
using SpecialFunctions, Roots, QuadGK
using NetworkHistogram
using Plots

# ---------- Example graphons  ----------
function beta_graphon(x, y)
    n = length(x)
    G = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        x1 = beta_inc_inv(0.55, 0.55, x[i])[1]
        y1 = beta_inc_inv(0.55, 0.55, y[j])[1]
        x2 = beta_inc_inv(0.45, 0.45, 1 - x[i])[1]
        y2 = beta_inc_inv(0.45, 0.45, 1 - y[j])[1]
        G[i, j] = 0.2 * (0.1 + 2.0*x1*y1 + 2.0*x2*y2)
    end
    G
end

function expo_graphon(x, y)
    n = length(x)
    G = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        G[i, j] = 0.2 * exp(-0.5 * (x[i] + y[j]))
    end
    G
end

# ---------- Utilities ----------
@inline xlogx(x::Float64) = (x <= 0.0 ? 0.0 : x * log(x))

# bit i (1-based) from ℓ (1-based cell index); LSB is layer 1
@inline bit_at(ℓ::Int, i::Int) = ((ℓ - 1) >> (i - 1)) & 0x01

@inline function H_from_cells(probs::NTuple{N,Float64}) where {N}
    s = 0.0
    @inbounds @simd for r in 1:N
        s += xlogx(probs[r])
    end
    -s
end

# ---------- Data generation (simple dependent pair + XOR third) ----------
function generate_net(n::Int)
    ξ = rand(n)
    slots = triu(ones(Bool, n, n), 1)
    u = rand(count(slots))
    P1 = expo_graphon(ξ, ξ)
    P2 = beta_graphon(ξ, ξ)
    A1 = zeros(Int, n, n); A2 = zeros(Int, n, n)
    A1[slots] = u .< P1[slots]
    A2[slots] = u .< P2[slots]
    A1 .+= A1'; A2 .+= A2'
    A3 = xor.(A1, A2)
    cat(A1, A2, A3; dims = 3)   # n×n×d with d=3
end

# --- helper: expand k×k block matrix Θ to n×n by node labels ---
@inline function block_expand(Θ::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    n = length(labels)
    P = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        P[i, j] = Θ[labels[i], labels[j]]
    end
    return P
end

# --- build Pcells from a NetworkHistogram.GraphHist (no smoothing) ---
# Using the inference method of Dufour and Olhede (2024) and implementation by Dufour and Grainger (2023)
function build_Pcells(est::NetworkHistogram.GraphHist)
    θ = est.θ
    n = length(est.node_labels)

    if ndims(θ) == 3
        # θ is k×k×L array: θ[:,:,ℓ] = block probs for cell ℓ
        k1, k2, L = size(θ); @assert k1 == k2
        Pcells = zeros(Float64, n, n, L)
        @inbounds for ℓ in 1:L
            Θℓ = @view θ[:, :, ℓ]             # k×k
            Pcells[:, :, ℓ] = block_expand(Θℓ, est.node_labels)
            # If your NH exports get_p_matrix, you could do:
            # Pcells[:, :, ℓ] = NetworkHistogram.get_p_matrix(Θℓ, est.node_labels)
        end
        return Pcells

    elseif ndims(θ) == 2 && eltype(θ) <: AbstractVector
        # θ is k×k matrix of length-L vectors
        k = size(θ, 1)
        L = length(θ[1,1])
        Pcells = zeros(Float64, n, n, L)
        @inbounds for ℓ in 1:L
            Θℓ = [θ[a,b][ℓ] for a in 1:k, b in 1:k]  # k×k for cell ℓ
            Pcells[:, :, ℓ] = block_expand(Θℓ, est.node_labels)
        end
        return Pcells

    else
        error("Unrecognized θ layout: size=$(size(θ)), eltype=$(eltype(θ))")
    end
end


# ---------- Information measures ----------
function info_measures_from_cells(n::Int, Pcells::Array{Float64,3})
    L = size(Pcells, 3)
    d = round(Int, log2(L))
    @assert d == 3 "This function expects d=3 (L=8). Extend bit loops for general d."

    H1=0.0; H2=0.0; H3=0.0
    H12=0.0; H13=0.0; H23=0.0
    H123=0.0
    inv_n2 = 1.0/(n^2)

    @inbounds for i in 1:n, j in 1:n
        # single-layer marginals
        p1=p2=p3=0.0
        for ℓ in 1:L
            α1 = bit_at(ℓ,1); α2 = bit_at(ℓ,2); α3 = bit_at(ℓ,3)
            pijℓ = Pcells[i,j,ℓ]
            p1 += (α1==1) ? pijℓ : 0.0
            p2 += (α2==1) ? pijℓ : 0.0
            p3 += (α3==1) ? pijℓ : 0.0
        end
        H1 += (-(xlogx(p1) + xlogx(1-p1))) * inv_n2
        H2 += (-(xlogx(p2) + xlogx(1-p2))) * inv_n2
        H3 += (-(xlogx(p3) + xlogx(1-p3))) * inv_n2

        # pairwise joint entropies
        p11=p10=p01=p00=0.0
        q11=q10=q01=q00=0.0
        r11=r10=r01=r00=0.0
        for ℓ in 1:L
            α1 = bit_at(ℓ,1); α2 = bit_at(ℓ,2); α3 = bit_at(ℓ,3)
            pijℓ = Pcells[i,j,ℓ]
            # (1,2)
            if α1==1 && α2==1; p11 += pijℓ
            elseif α1==1 && α2==0; p10 += pijℓ
            elseif α1==0 && α2==1; p01 += pijℓ
            else; p00 += pijℓ; end
            # (1,3)
            if α1==1 && α3==1; q11 += pijℓ
            elseif α1==1 && α3==0; q10 += pijℓ
            elseif α1==0 && α3==1; q01 += pijℓ
            else; q00 += pijℓ; end
            # (2,3)
            if α2==1 && α3==1; r11 += pijℓ
            elseif α2==1 && α3==0; r10 += pijℓ
            elseif α2==0 && α3==1; r01 += pijℓ
            else; r00 += pijℓ; end
        end
        H12 += (-(xlogx(p11)+xlogx(p10)+xlogx(p01)+xlogx(p00))) * inv_n2
        H13 += (-(xlogx(q11)+xlogx(q10)+xlogx(q01)+xlogx(q00))) * inv_n2
        H23 += (-(xlogx(r11)+xlogx(r10)+xlogx(r01)+xlogx(r00))) * inv_n2

        # full trivariate (8 cells)
        c1=c2=c3=c4=c5=c6=c7=c8=0.0
        c1=Pcells[i,j,1]; c2=Pcells[i,j,2]; c3=Pcells[i,j,3]; c4=Pcells[i,j,4]
        c5=Pcells[i,j,5]; c6=Pcells[i,j,6]; c7=Pcells[i,j,7]; c8=Pcells[i,j,8]
        H123 += H_from_cells((c1,c2,c3,c4,c5,c6,c7,c8)) * inv_n2
    end

    mi_12 = H1 + H2 - H12
    mi_13 = H1 + H3 - H13
    mi_23 = H2 + H3 - H23

    cmi_12_3 = H23 + H13 - H123 - H3
    cmi_13_2 = H23 + H12 - H123 - H2
    cmi_23_1 = H13 + H12 - H123 - H1

    tri = H1 + H2 + H3 - H12 - H13 - H23 + H123

    return tri, mi_12, mi_23, mi_13, cmi_12_3, cmi_13_2, cmi_23_1
end

# ---------- Demo run (NetworkHistogram only) ----------
let
    n = 500
    A = generate_net(n)                      # n×n×3

# ---------- Fit NetworkHistogram ----------
    estimator, history = graphhist(A;
        starting_assignment_rule = EigenStart(),
        maxitr = Int(1e7),
        stop_rule = PreviousBestValue(10_000))



# ---------- Build cell-probability tensor directly from estimated.θ ----------
Pcells = build_Pcells(estimator)

# ---------- Print information measures ----------
    tri, mi_12, mi_23, mi_13, cmi_12_3, cmi_13_2, cmi_23_1 =
        info_measures_from_cells(n, Pcells)

    println("I(1;2;3)   = ", tri)
    println("I(1;2)     = ", mi_12)
    println("I(2;3)     = ", mi_23)
    println("I(1;3)     = ", mi_13)
    println("I(1;2 | 3) = ", cmi_12_3)
    println("I(1;3 | 2) = ", cmi_13_2)
    println("I(2;3 | 1) = ", cmi_23_1)
end

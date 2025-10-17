
############################
#Implementation for the paper Multivariate Dependencies in Multiplex Graphs:Theory and Estimation by Skeja and Olhede (Skeja and Olhede,2025)#
############################
# Install the NetworkHistogram package from Dufour and Grainger (2023), that implements the code of Dufour and Olhede (2024)
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
for pkg in (
    "LaTeXStrings", "Distributions", "SpecialFunctions",
    "Roots", "QuadGK", "Plots", "NetworkHistogram"
)
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end
# --------------------------------------------------------------------

using LinearAlgebra, Statistics, Random, Distributions
using SpecialFunctions, Roots, QuadGK
using NetworkHistogram
using Plots
using Printf
using LaTeXStrings

# ---------------------- graphon definitions -------------------------

function beta_graphon(x, y)
    n = length(x)
    result = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            x_val1 = beta_inc_inv(0.55, 0.55, x[i])[1]
            y_val1 = beta_inc_inv(0.55, 0.55, y[j])[1]
            x_val2 = beta_inc_inv(0.45, 0.45, 1 - x[i])[1]
            y_val2 = beta_inc_inv(0.45, 0.45, 1 - y[j])[1]
            result[i, j] = 0.2 * (0.1 + (4 * 0.5 * x_val1 * y_val1) + (4 * 0.5 * x_val2 * y_val2))
        end
    end
    return result
end

function expo_graphon(x, y)
    n = length(x)
    result1 = zeros(n, n)
    for i in 1:n
        for j in 1:n
            result1[i, j] = 0.2 * exp(-0.5 * (x[i] + y[j]))
        end
    end
    return result1
end

# ---------------------- utilities -------------------------

function xlogx(x)
    if x == 0
        return 0
    elseif x < 0
        return 0
    else
        return x * log(x)
    end
end

# ---------------------- data generation -------------------------

function generate_net(n)
    xi = rand(n)  # latent variables

    numPairs = n * (n - 1) ÷ 2  # number of random die rolls

    P1 = expo_graphon(xi, xi)   # expectation of Bernoulli matrix
    A1 = zeros(Int, n, n)

    # symmetric independent realizations
    slots = triu(ones(Bool, n, n), 1)
    common_random_numbers = rand(numPairs)

    # fill A according to P
    A1[slots] = common_random_numbers .< P1[slots]
    A1 = A1 + A1'

    P2 = beta_graphon(xi, xi)
    A2 = zeros(Int, n, n)
    A2[slots] = common_random_numbers .< P2[slots]
    A2 = A2 + A2'

    return A1, A2
end

# ---------------------- graphon information measures -------------------------

function graphon_info_measures(n_nodes, est, p_1, p_2, p_3, p_12, p_13, p_23, p_123)
    P_hat1   = zeros(n_nodes, n_nodes)
    P_hat2   = zeros(n_nodes, n_nodes)
    P_hat3   = zeros(n_nodes, n_nodes)
    P_hat12  = zeros(n_nodes, n_nodes)
    P_hat13  = zeros(n_nodes, n_nodes)
    P_hat23  = zeros(n_nodes, n_nodes)
    P_hat123 = zeros(n_nodes, n_nodes)

    entropy1 = zeros(n_nodes, n_nodes)
    entropy2 = zeros(n_nodes, n_nodes)
    entropy3 = zeros(n_nodes, n_nodes)
    joint_entropy12 = zeros(n_nodes, n_nodes)
    joint_entropy13 = zeros(n_nodes, n_nodes)
    joint_entropy23 = zeros(n_nodes, n_nodes)
    jointentropy    = zeros(n_nodes, n_nodes)

    total_entropy1 = 0.0
    total_entropy2 = 0.0
    total_entropy3 = 0.0
    total_joint_entropy12 = 0.0
    total_joint_entropy13 = 0.0
    total_joint_entropy23 = 0.0
    total_joint_entropy   = 0.0

    for i in 1:n_nodes
        for j in 1:n_nodes
            P_hat12[i, j] = p_12[est.node_labels[i], est.node_labels[j]]
            P_hat13[i, j] = p_13[est.node_labels[i], est.node_labels[j]]
            P_hat23[i, j] = p_23[est.node_labels[i], est.node_labels[j]]

            P_hat1[i, j]  = p_1[est.node_labels[i], est.node_labels[j]]
            entropy1[i, j] = (-xlogx(P_hat1[i, j]) - xlogx(1 - P_hat1[i, j])) / (n_nodes)^2
            total_entropy1 += entropy1[i, j]

            P_hat2[i, j]  = p_2[est.node_labels[i], est.node_labels[j]]
            entropy2[i, j] = (-xlogx(P_hat2[i, j]) - xlogx(1 - P_hat2[i, j])) / (n_nodes)^2
            total_entropy2 += entropy2[i, j]

            P_hat3[i, j]  = p_3[est.node_labels[i], est.node_labels[j]]
            entropy3[i, j] = (-xlogx(P_hat3[i, j]) - xlogx(1 - P_hat3[i, j])) / (n_nodes)^2
            total_entropy3 += entropy3[i, j]

            joint_entropy12[i, j] =
                -(xlogx(P_hat12[i, j]) +
                  xlogx(P_hat1[i, j] - P_hat12[i, j]) +
                  xlogx(P_hat2[i, j] - P_hat12[i, j]) +
                  xlogx(1 - P_hat1[i, j] - P_hat2[i, j] + P_hat12[i, j])) / (n_nodes)^2
            total_joint_entropy12 += joint_entropy12[i, j]

            joint_entropy13[i, j] =
                -(xlogx(P_hat13[i, j]) +
                  xlogx(P_hat1[i, j] - P_hat13[i, j]) +
                  xlogx(P_hat3[i, j] - P_hat13[i, j]) +
                  xlogx(1 - P_hat1[i, j] - P_hat3[i, j] + P_hat13[i, j])) / (n_nodes)^2
            total_joint_entropy13 += joint_entropy13[i, j]

            joint_entropy23[i, j] =
                -(xlogx(P_hat23[i, j]) +
                  xlogx(P_hat2[i, j] - P_hat23[i, j]) +
                  xlogx(P_hat3[i, j] - P_hat23[i, j]) +
                  xlogx(1 - P_hat2[i, j] - P_hat3[i, j] + P_hat23[i, j])) / (n_nodes)^2
            total_joint_entropy23 += joint_entropy23[i, j]

            P_hat123[i, j] = p_123[est.node_labels[i], est.node_labels[j]]
            jointentropy[i, j] =
                -(xlogx(P_hat123[i, j]) +
                  xlogx(P_hat12[i, j] - P_hat123[i, j]) +
                  xlogx(P_hat13[i, j] - P_hat123[i, j]) +
                  xlogx(P_hat23[i, j] - P_hat123[i, j]) +
                  xlogx(P_hat1[i, j] - P_hat12[i, j] - P_hat13[i, j] + P_hat123[i, j]) +
                  xlogx(P_hat2[i, j] - P_hat12[i, j] - P_hat23[i, j] + P_hat123[i, j]) +
                  xlogx(P_hat3[i, j] - P_hat13[i, j] - P_hat23[i, j] + P_hat123[i, j]) +
                  xlogx(1 - P_hat1[i, j] - P_hat2[i, j] - P_hat3[i, j] + P_hat12[i, j] + P_hat13[i, j] + P_hat23[i, j] - P_hat123[i, j])) / (n_nodes)^2
            total_joint_entropy += jointentropy[i, j]
        end
    end

    mi_12 = total_entropy1 + total_entropy2 - total_joint_entropy12
    mi_13 = total_entropy1 + total_entropy3 - total_joint_entropy13
    mi_23 = total_entropy2 + total_entropy3 - total_joint_entropy23

    # conditional mutual informations
    cmi_12_3 = total_joint_entropy23 + total_joint_entropy13 - total_joint_entropy - total_entropy3
    cmi_13_2 = total_joint_entropy23 + total_joint_entropy12 - total_joint_entropy - total_entropy2
    cmi_23_1 = total_joint_entropy13 + total_joint_entropy12 - total_joint_entropy - total_entropy1

    # trivariate (interaction) information
    trivariate_mi = total_entropy1 + total_entropy2 + total_entropy3 -
                    total_joint_entropy12 - total_joint_entropy13 - total_joint_entropy23 +
                    total_joint_entropy

    # Return the seven values your downstream code actually uses.
    return trivariate_mi, mi_12, mi_23, mi_13, cmi_12_3, cmi_13_2, cmi_23_1
end

# ---------------------- Example test: two dependent graphs, with the third generated as their XOR ----------------------


# Use your adjacency matrices A1, A2, A3. An example is given below 
    n = 500

    A1, A2 = generate_net(n)
    A3 = xor.(A1, A2)

    out = graphhist(cat(A1, A2, A3, dims = 3);
        starting_assignment_rule = OrderedStart(),
        maxitr = 1_000_000,
        stop_rule = PreviousBestValue(100)
    )

    est = out.graphhist
    n_nodes = size(est.node_labels, 1)

    p = NetworkHistogram.get_moment_representation(est)[1]
    p_1  = p[:, :, 1]
    p_2  = p[:, :, 2]
    p_3  = p[:, :, 4]
    p_12 = p[:, :, 3]
    p_13 = p[:, :, 5]
    p_23 = p[:, :, 6]
    p_123 = p[:, :, 7]

    trivariate_mi, mi_12, mi_23, mi_13, cmi_12_3, cmi_13_2, cmi_23_1 =
        graphon_info_measures(n_nodes, est, p_1, p_2, p_3, p_12, p_13, p_23, p_123)

    println("I(1;2;3)   = ", trivariate_mi)
    println("I(1;2)     = ", mi_12)
    println("I(2;3)     = ", mi_23)
    println("I(1;3)     = ", mi_13)
    println("I(1;2 | 3) = ", cmi_12_3)
    println("I(1;3 | 2) = ", cmi_13_2)
    println("I(2;3 | 1) = ", cmi_23_1)

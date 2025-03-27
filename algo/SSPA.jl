using Distributions
using LinearAlgebra
using SparseArrays
using Random
using LinearAlgebra
using MAT
include("NNLS.jl")

function SSPA(X, r, p, options)
    """
    Smoothed Vertex Component Analysis (SVCA)

    Heuristic to solve the following problem:
    Given a matrix X, find a matrix W such that X ≈ WH for some H ≥ 0,
    under the assumption that each column of W has `p` columns of X close
    to it (called the `p` proximal latent points).

    # Parameters:
    - `X::Matrix{Float64}` : Input data matrix of size (m, n).
    - `r::Int` : Number of columns of W.
    - `p::Int` : Number of proximal latent points.
    - `options::Dict{String, Int}=Dict()` (optional):
        - `"lra"` (`Int`):  
            - `1` uses a low-rank approximation (LRA) of X in the selection step.  
            - `0` (default) does not.  
        - `"average"` (`Int`):  
            - `1` uses the mean for aggregation.  
            - `0` (default) uses the median.  

    # Returns:
    - `W::Matrix{Float64}` : The matrix such that X ≈ WH.
    - `K::Vector{Int}` : Indices of the selected data points (one column per iteration).

    This code is based on the paper *Smoothed Separable Nonnegative Matrix Factorization*  
    by N. Nadisic, N. Gillis, and C. Kervazo  
    [https://arxiv.org/abs/2110.05528](https://arxiv.org/abs/2110.05528)
"""

    if !haskey(options, :lra)
        options[:lra] = 0
    end
    
    if options[:lra] == 1
        Y, S, Z = svds(X, nsv=r)
        Z = S * Z'
    else
        Z = X
    end

    if !haskey(options, :average)
        options[:average] = 0
    end
    
    V = []
    normX2 = sum(X.^2, dims=1)
    normX2=Float64.(normX2)
    W = zeros(size(Z, 1), r)
    K = zeros(Int, r, p)
    
    for k in 1:r
        spa = argmax(normX2)
        diru = X[:, spa.I[2]]
        
        if k >= 2
            diru -= V * (V' * diru)
        end
        
        u = diru' * X
        b = sortperm(vec(u), rev=true)
        K[k, :] = b[1:p]
        
        if p == 1
            W[:, k] = Z[:, K[k, :]]
        else
            if options[:average] == 1
                W[:, k] = mean(Z[:, K[k, :]], dims=2)
            elseif options[:average] == 3
                W[:, k] = colaverage(Z[:, K[k, :]], 3)
            elseif options[:average] == 4
                W[:, k] = colaverage(Z[:, K[k, :]], 4)
            else
                W[:, k] = median(Z[:, K[k, :]], dims=2)
            end
        end
        
        V = updateorthbasis(V, W[:, k])
        normX2 .-= (transpose(V[:, end]) * X).^2
    end
    
    if options[:lra] == 1
        W = Y * W
    end
    
    return W, K
end

function svds(A, k)
    U, Σ, V = svd(A)
    Uk = U[:, 1:k]
    Σk = Diagonal(Σ[1:k])
    Vk = V[:, 1:k]
    return Uk, Σk, Vk
end

function updateorthbasis(V, v)
    if isempty(V)
        V = v / norm(v,2)
    else
        # Project new vector onto orthogonal complement, and normalize
        v -= V * (V' * v)
        v /= norm(v)
        V = hcat(V, v)
    end
    return V
end
function colaverage(W, type)
    if size(W, 2) == 1
        return W
    else
        if type == 1
            return mean(W, dims=2)
        elseif type == 2
            return median(W, dims=2)
        elseif type == 3
            u, s, v = svds(W, nsv=1)
            if sum(v[v .> 0]) < sum(v[v .< 0])
                u = -u
                v = -v
            end
            v = s .* v
            return u * mean(v)
        elseif type == 4
            u, v = L1LRAcd(W, 1, 100)
            return u * median(v)
        end
    end
end

function spa(X::AbstractMatrix{T}, r::Integer, epsilon::Float64=10e-9) where T <: AbstractFloat
    # Get dimensions
    m, n = size(X)
    col_sums = sum(X, dims=1)

    # Calculate the inverse of the column sums and transpose to get a row vector
    inverse_col_sums = 1. ./ col_sums'
    D = diagm(vec(inverse_col_sums))
    X = X * D
    
    # Set of selected indices
    K = zeros(Int, r)

    # Norm of columns of input X
    normX0 = sum.(abs2, eachcol(X))
    # Max of the columns norm
    nXmax = maximum(normX0)
    
    # Init residual
    normR = copy(normX0)
    
    # Init set of extracted columns
    U = Matrix{T}(undef, m, r)

    i = 1

    while i <= r && sqrt(maximum(normR) / nXmax) > epsilon    
        # Select column of X with largest l2-norm
        a = maximum(normR)
        
        # Check ties up to 1e-6 precision
        b = findall((a .- normR) / a .<= 1e-6)
        
        # In case of a tie, select column with largest norm of the input matrix
        _, d = findmax(normX0[b])
        b = b[d]
        
        # Save index of selected column, and column itself
        K[i] = b
        U[:, i] .= X[:, b]
        
        # Update residual
        for j in 1:i-1
            U[:, i] .= U[:, i] - U[:, j] * (U[:, j]' * U[:, i])
        end
        
        U[:, i] ./= norm(U[:, i])
        normR .-= (X' * U[:, i]).^2
        
        # Increment iterator
        i += 1
    end

    return K
end


# Orthogonal Symmetric Nonnegative Matrix Tri-Factorization (OtrisymNMF)

This repository provides a Julia implementation of the **Orthogonal Symmetric Nonnegative Matrix Tri-Factorization** (OtrisymNMF) algorithm, as proposed in the paper:

**Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis.**  
*"Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."*  
IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2024.  
Institute of Electrical and Electronics Engineers (IEEE), United States.

The algorithm aims to solve the following optimization problem:

$$
\min_{W \geq 0, S \geq 0} \|\|X - WSW^T\|\|_F^2 \quad \text{s.t.} \quad W^TW = I
$$

Where:
- $X$ is a given symmetric nonnegative matrix (e.g., adjacency matrix of an undirected graph).
- $W$ is a matrix representing the assignment of elements to \( r \) communities.
- $S$ is a central matrix describing interactions between communities.

## OtrisymNMF Julia Methods

### 1. **OtrisymNMF_CD(X, r, maxiter, epsi, init_algo, time_limit)**
   - **Parameters**:
     - `X::Matrix{Float64}` : Symmetric nonnegative matrix (Adjacency matrix of an undirected graph).
     - `r::Int` : Number of columns of W (Number of communities).
     - `maxiter::Int` : Maximum number of iterations (default: 1000).
     - `epsi::Float64` : Convergence tolerance (default: 1e-7).
     - `init_algo::String` : Initialization method (`"random"`,`"svca"`, `"k_means"`, `"sspa"`).
     - `time_limit::Int` : Time limit for each trial in seconds (default: 5).
   - **Returns**:
     - `W::Matrix{Float64}` : Assignment matrix.
     - `S::Matrix{Float64}` : Central matrix.
     - `error_best::Float64` : Relative error  $\|X - WSW^T\|_F / \|X\|_F$.

## Tests

The repository contains the tests of the paper in the `/Test/` directory:
- `Test_synt_MU.jl`
- `Test_synt_initializations.jl`
- `CBCL.jl`
- `TDT2.jl`


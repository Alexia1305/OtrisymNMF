
# OtrisymNMF - Orthogonal Symmetric Nonnegative Matrix Trifactorization

## Overview

The **Orthogonal Symmetric Nonnegative Matrix Trifactorization** (OtrisymNMF) is a method designed to decompose a symmetric nonnegative matrix \( X \geq 0 \) into two matrices \( W \geq 0 \) and \( S \geq 0 \), such that:

\[
X \approx W S W^T \quad \text{with} \quad W^T W = I
\]

This approach is useful for tasks like community detection, matrix factorization, and clustering. The **Coordinate Descent** (CD) algorithm is used for optimizing the matrices \( W \) and \( S \).

## Folder Structure

```
- algo/
  - OtrisymNMF/
    - OtrisymNMF_CD.m
```

## Function: `OtrisymNMF_CD`

The `OtrisymNMF_CD` function performs Orthogonal Symmetric Nonnegative Matrix Trifactorization using a **Coordinate Descent** approach. It solves the following optimization problem:

\[
\min_{W \geq 0, S \geq 0} \| X - W S W^T \|_F^2 \quad \text{subject to} \quad W^T W = I
\]

### Function Signature

```matlab
[w, v, S, error] = OtrisymNMF_CD(X, r, varargin)
```

### Inputs:

- **X**: A symmetric nonnegative matrix \( X \) of size \( n \times n \), representing the adjacency matrix of a graph or any other symmetric nonnegative matrix.
- **r**: An integer specifying the number of columns in matrix \( W \), i.e., the number of communities or clusters to be detected.

#### Optional Parameters (via `varargin`):
- **numTrials** (default: 1): Number of trials with different initializations.
- **maxiter** (default: 1000): Maximum number of iterations per trial.
- **delta** (default: 1e-7): Convergence tolerance. The iteration stops when the relative error between two consecutive updates is less than `delta` or when the error itself becomes smaller than `delta`.
- **time_limit** (default: 300 seconds): Maximum time limit in seconds for the heuristic.
- **init** (default: "SSPA"): Initialization method for the algorithm, with available options being `"random"`, `"SSPA"`, `"SVCA"`, and `"SPA"`.
- **verbosity** (default: 1): Verbosity level for output display. Use `1` to display messages and `0` for silent mode.

### Outputs:

- **v**: A vector of length \( n \) where `v(i)` gives the index of the nonzero columns of \( W \) for the i-th row.
- **w**: A vector of length \( n \) where `w(i)` gives the value of the nonzero element in the i-th row of \( W \).
- **S**: The central matrix \( S \) of size \( r \times r \).
- **error**: The relative error of the factorization, computed as:

\[
\text{error} = \frac{\| X - W S W^T \|_F}{\| X \|_F}
\]

## Example

An example script demonstrating how to use the `OtrisymNMF_CD` function is included in the script`Exemple.m`.




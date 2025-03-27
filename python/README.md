
# OtrisymNMF

Orthogonal Symmetric Nonnegative Matrix Trifactorization using Coordinate Descent.

Given a symmetric matrix X ≥ 0, OtrisymNMF finds matrices W ≥ 0 and S ≥ 0 such that:

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]

## Description

- **W** is represented by:
  - **v**: Indices of the nonzero columns of W for each row.
  - **w**: Values of the nonzero elements in each row of W.

- **Application to community detection**:
  - `X` is the adjacency matrix of an undirected graph.
  - `OtrisymNMF` detects `r` communities.
  - `v` assigns each node to a community.
  - `w` indicates the importance of a node within its community.
  - `S` describes interactions between the `r` communities.

Reference:  
*"Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."*  
2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2024.

## Installation

To use the **OtrisymNMF** package, download it and place it in your project directory.  
Then, import it as follows:

```python
import OtrisymNMF

## Example

An example notebook, **`Example.ipynb`**, is provided in the repository to help you get started with the OtrisymNMF method.

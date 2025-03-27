# OtrisymNMF: Orthogonal Symmetric Nonnegative Matrix Tri-Factorization

 This repository provides implementations in in three different programming languages: **Julia**, **MATLAB**, and **Python** of the **Orthogonal Symmetric Nonnegative Matrix Tri-Factorization** (OtrisymNMF) algorithm, as proposed in the paper:

**Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis.**  
*"Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."*  
IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2024.  
Institute of Electrical and Electronics Engineers (IEEE), United States.

The algorithm aims to solve the following optimization problem:

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]

Where:
- \( X \) is a given symmetric nonnegative matrix (e.g., adjacency matrix of an undirected graph).
- \( W \) is a matrix representing the assignment of elements to \( r \) communities.
- \( S \) is a central matrix describing interactions between communities.

Each language-specific implementation is located in its respective folder, with a corresponding README file for detailed instructions.
The Julia folder contains the tests described in the paper *Orthogonal Symmetric Nonnegative Matrix Tri-Factorization* (DOI: [10.1109/MLSP58920.2024.10734715](https://doi.org/10.1109/MLSP58920.2024.10734715)) 



## Citing

If you use this code for your research, please cite the following paper:
Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis. "Orthogonal Symmetric Nonnegative Matrix Tri-Factorization." IEEE International Workshop on Machine Learning for Signal Processing. Institute of Electrical and Electronic Engineers (IEEE), United States, 2024.

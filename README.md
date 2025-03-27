# OtrisymNMF: Orthogonal Symmetric Nonnegative Matrix Tri-Factorization

This repository contains implementations of our method for **Orthogonal Symmetric Nonnegative Matrix Tri-Factorization (OtrisymNMF)** in three different programming languages: **Julia**, **MATLAB**, and **Python**. Each language-specific implementation is located in its respective folder, with a corresponding README file for detailed instructions.
The julia folder contains the tests described in the paper *Orthogonal Symmetric Nonnegative Matrix Tri-Factorization* (DOI: [10.1109/MLSP58920.2024.10734715](https://doi.org/10.1109/MLSP58920.2024.10734715)) 

## Overview

The **Orthogonal Symmetric Nonnegative Matrix Tri-Factorization (OtrisymNMF)** decomposes a symmetric nonnegative matrix \( X \geq 0 \) into two matrices \( W \geq 0 \) and \( S \geq 0 \), such that:

\[
X \approx W S W^T \quad \text{with} \quad W^T W = I
\]


Our method solves the following optimization problem:

\[
\min_{W \geq 0, S \geq 0} \|X - WSW^T\|_F^2 \quad \text{s.t.} \quad W^TW = I
\]


## Citing

If you use this code for your research, please cite the following paper:
Dache, Alexandra, Arnaud Vandaele, and Nicolas Gillis. "Orthogonal Symmetric Nonnegative Matrix Tri-Factorization." IEEE International Workshop on Machine Learning for Signal Processing. Institute of Electrical and Electronic Engineers (IEEE), United States, 2024.


# OtrisymNMF - Orthogonal Symmetric Nonnegative Matrix Trifactorization

## Overview

The **Orthogonal Symmetric Nonnegative Matrix Trifactorization** (OtrisymNMF) is a method designed to decompose a symmetric nonnegative matrix \( X \geq 0 \) into two matrices \( W \geq 0 \) and \( S \geq 0 \), such that:

\[
X \approx W S W^T \quad \text{with} \quad W^T W = I
\]

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


An example script demonstrating how to use the `OtrisymNMF_CD` function is included in the script`Exemple.m`.




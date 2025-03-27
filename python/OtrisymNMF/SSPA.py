import numpy as np
from scipy.sparse.linalg import svds


def update_orth_basis(V, v):
    """
    Updates the orthonormal basis V by adding a new vector v while ensuring orthogonality.

    Parameters:
        V (numpy.ndarray): Current orthonormal basis (m x k) where columns are basis vectors.
        v (numpy.ndarray): New vector to be added (m,).

    Returns:
        numpy.ndarray: Updated orthonormal basis including v.
    """
    if V.size == 0:
        # If V is empty, normalize v and set it as the first basis vector
        V = v / np.linalg.norm(v).reshape(1, -1)
        V=V.T
    else:
        # Project v onto the orthogonal complement of V
        v = v - V @ (V.T @ v)
        # Normalize v
        v = v / np.linalg.norm(v)
        # Append to the basis
        V = np.column_stack((V, v))

    return V


def SSPA(X, r, p, options=None):
    """
    Smoothed Successive Projection Algorithm (SSPA)

    This heuristic algorithm finds a matrix W such that X ≈ WH for some H ≥ 0,
    under the assumption that each column of W has p columns of X close to it
    (called the p proximal latent points).

    Parameters:
        X (numpy.ndarray): Input data matrix of size (m, n).
        r (int): Number of columns of W.
        p (int): Number of proximal latent points.
        options (dict, optional):
            - 'lra' (int):
                1 uses a low-rank approximation (LRA) of X in the selection step,
                0 (default) does not.
            - 'average' (int):
                1 uses the mean for aggregation,
                0 (default) uses the median.

    Returns:
        W (numpy.ndarray): The matrix such that X ≈ WH.
        K (numpy.ndarray): Indices of the selected data points (one column per iteration).

        This code is based on the paper Smoothed Separable Nonnegative Matrix Factorization
        by N. Nadisic, N. Gillis, and C. Kervazo
        https://arxiv.org/abs/2110.05528
    """
    if options is None:
        options = {}

    # Set default options if not provided
    X=X.astype('float')
    lra = options.get('lra', 0)
    average = options.get('average', 0)

    # Low-rank approximation (LRA) of the input matrix if enabled
    if lra == 1:
        U, S, Vt = svds(X, k=r)  # Compute rank-r SVD of X
        Z = np.dot(np.diag(S), Vt)  # Reconstruct low-rank version
    else:
        Z = X.copy()

    V = np.array([])  # Initialize the orthogonal basis
    normX2 = np.sum(X ** 2, axis=0)  # Compute squared L2 norm of each column of X

    W = np.zeros((X.shape[0], r))  # Initialize W matrix
    K = np.zeros((r, p), dtype=int)  # Store indices of selected data points

    for k in range(r):
        # Select SPA direction (column with maximum norm)
        spb = np.argmax(normX2)
        diru = X[:, spb]

        # Ensure orthogonality to previously extracted columns
        if k >= 1:
            diru -= np.dot(V, np.dot(V.T, diru))

        # Compute inner product with data matrix
        u = np.dot(diru.T, X)

        # Sort values and select indices corresponding to largest values
        sorted_indices = np.argsort(-u)  # Descending order
        K[k, :] = sorted_indices[:p]  # Select top p indices

        # Compute new column for W
        if p == 1:
            W[:, k] = Z[:, K[k, 0]]
        else:
            if average == 1:
                W[:, k] = np.mean(Z[:, K[k, :]], axis=1).T
            else:
                W[:, k] = np.median(Z[:, K[k, :]], axis=1).T

        # Update orthogonal basis
        V = update_orth_basis(V, W[:, k])

        # Update squared L2 norm of columns
        normX2 -= (np.dot(V[:, -1].T, X)) ** 2

    # If low-rank approximation was used, project W back
    if lra == 1:
        W = np.dot(U, W)

    return W, K
def SVCA(X,r,p,options=None):
    """
        Smoothed Vertex Component Analysis(SVCA)

        Heuristic to solve the following problem:
        Given a matrix X, find a matrix W such that X~=WH for some H>=0,
        under the assumption that each column of W has p columns of X close
        to it (called the p proximal latent points).

        Parameters:
            X (numpy.ndarray): Input data matrix of size (m, n).
            r (int): Number of columns of W.
            p (int): Number of proximal latent points.
            options (dict, optional):
                - 'lra' (int):
                    1 uses a low-rank approximation (LRA) of X in the selection step,
                    0 (default) does not.
                - 'average' (int):
                    1 uses the mean for aggregation,
                    0 (default) uses the median.

        Returns:
            W (numpy.ndarray): The matrix such that X ≈ WH.
            K (numpy.ndarray): Indices of the selected data points (one column per iteration).

            This code is based on the paper Smoothed Separable Nonnegative Matrix Factorization
            by N. Nadisic, N. Gillis, and C. Kervazo
            https://arxiv.org/abs/2110.05528
        """
    if options is None:
        options = {}

        # Approximation de faible rang (LRA) de la matrice d'entrée
        # Par défaut, il n'y a pas d'approximation de faible rang
        # Set default options if not provided
    X = X.astype('float')
    if 'lra' not in options:
        options['lra'] = 0

        # Calcul des vecteurs singuliers
    U, S, Vt = svds(X, k=r)  # U contient les premiers r vecteurs singuliers de X
    # On peut utiliser des algorithmes plus rapides ici, comme mentionné dans le code MATLAB

    if options['lra'] == 1:
        X = np.dot(S, Vt)  # Remplace X par son approximation de faible rang

    # Agrégation par moyenne ou médiane (par défaut : médiane)
    if 'average' not in options:
        options['average'] = 0  # Médiane par défaut

    # Projection (I - VV^T) sur le complément orthogonal des colonnes de W
    V = np.empty((X.shape[0], 0))  # Matrice vide pour commencer les itérations de SVCA

    W = np.zeros((X.shape[0], r))  # Matrice W de la taille m * r
    K = np.zeros((r, p), dtype=int)  # Indices des points de données sélectionnés

    for k in range(r):
        # Direction aléatoire dans la colonne de U
        diru = np.dot(U, np.random.randn(r))

        # Projection de la direction aléatoire pour être orthogonale aux colonnes extraites de W
        if k >= 1:
            diru = diru - np.dot(V, np.dot(V.T, diru))

        # Produit scalaire avec la matrice de données
        u = np.dot(diru.T, X)

        # Trier les entrées et sélectionner la direction maximisant |u|
        b = np.argsort(u)

        # Vérifier la condition de médiane
        if np.abs(np.median(u[b[:p]])) < np.abs(np.median(u[b[-p:]])):
            b = b[::-1]  # Inverser si nécessaire

        # Sélectionner les indices correspondant aux plus grandes valeurs de u
        K[k, :] = b[:p]

        # Calcul de la "vertex"
        if p == 1:
            W[:, k] = X[:, K[k, :]]
        else:
            if options['average'] == 1:
                W[:, k] = np.mean(X[:, K[k, :]], axis=1)
            else:
                W[:, k] = np.median(X[:, K[k, :]], axis=1)

        # Mise à jour du projecteur
        V = update_orth_basis(V, W[:, k])

    if options['lra'] == 1:
        W = np.dot(U, W)  # Si l'approximation de faible rang est activée, on multiplie par U

    return W, K
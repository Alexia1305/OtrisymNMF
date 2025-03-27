import numpy as np
import time
import math
from scipy.sparse import find

from .SSPA import SSPA
from .SSPA import SVCA
from .Utils import orthNNLS
from scipy.sparse import issparse

def OtrisymNMF_CD(X, r, numTrials=1, maxiter=1000, delta=1e-3, time_limit=300, init_method=None, verbosity=1):
    """
    Orthogonal Symmetric Nonnegative Matrix Trifactorization using Coordinate Descent.
    Given a symmetric matrix X >= 0, finds matrices W >= 0 and S >= 0 such that X ≈ WSW' with W'W=I.
    W is represented by:
    - v: indices of the nonzero columns of W for each row.
    - w: values of the nonzero elements in each row of W.

    Application to community detection:
        - X is the adjacency matrix of an undirected graph.
        - OtrisymNMF detects r communities.
        - v assigns each node to a community.
        - w indicates the importance of a node within its community.
        - S describes interactions between the r communities.

    "Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."
    2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2024.

    Parameters:
        X : np.array, shape (n, n)
            Symmetric nonnegative matrix (Adjacency matrix of an undirected graph).
        r : int
            Number of columns of W (Number of communities).
        numTrials : int, default=1
            Number of trials with different initializations.
        maxiter : int, default=1000
            Maximum iterations for each trial.
        delta : float, default=1e-7
            Convergence tolerance.
        time_limit : int, default=300
            Time limit in seconds.
        init_method : str, default=None
            Initialization method ("random", "SSPA", "SVCA", "SPA").
        verbosity : int, default=1
            Verbosity level (1 for messages, 0 for silent mode).

    Returns:
        w_best : np.array, shape (n,)
            Values of the nonzero elements for each row.
        v_best : np.array, shape (n,)
            Indices of the nonzero columns of W.
        S_best : np.array, shape (r, r)
            Central matrix.
        error_best : float
            Relative error ||X - WSW'||_F / ||X||_F.
    """
    start_time = time.time()
    if issparse(X):
        X=X.toarray()
    n = X.shape[0]
    error_best = float('inf')

    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    for trial in range(numTrials):
        w = np.zeros(n)
        v = np.zeros(n, dtype=int)

        if init_method is None:
            init_algo = "SSPA" if trial == 0 else "SVCA"
        else:
            init_algo = init_method

        # Initialization
        if init_algo == "random":
            v = np.random.randint(0, r, size=n)
            w = np.random.rand(n)
            # Normalisation des colonnes de W
            nw = np.zeros(r)
            for i in range(n):
                nw[v[i]] += w[i] ** 2
            nw = np.sqrt(nw)
            w /= nw[v]
            S = np.random.rand(r, r)
            S = (S + S.T) / 2  # Symmetric
        else:
            # Placeholder for proper initialization functions
            W = initialize_W(X, r, method=init_algo)
            w, v = extract_w_v(W)
            # Normalisation des colonnes de W
            nw = np.zeros(r)
            for i in range(n):
                nw[v[i]] += w[i] ** 2
            nw = np.sqrt(nw)
            w /= nw[v]
            S = update_S(X, r, w, v)

        # Iterative update
        prev_error = compute_error(X, S, w, v)
        error = prev_error

        for iteration in range(maxiter):
            if time.time() - start_time > time_limit:
                print('Time limit passed')
                break

            w, v = update_W(X, S, w, v)
            S = update_S(X, r, w, v)

            prev_error = error
            error = compute_error(X, S, w, v)

            if error < delta or abs(prev_error - error) < delta:
                break
        if iteration == maxiter-1:
            print('Not converged')

        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error
            if error_best <= delta or time.time() - start_time > time_limit:
                break

        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with {init_algo}: Error {error:.4e} | Best: {error_best:.4e}')

    return w_best, v_best, S_best, error_best


# Placeholder functions (to be implemented)
def initialize_W(X, r, method="SSPA"):
    """ Initializes W based on the chosen method."""

    if method == "SSPA":
        n = X.shape[0]
        p=max(2,math.floor(0.1*n/r))

        options = {'average': 1}
        WO, K = SSPA(X, r, p,options=options )

        norm2x = np.sqrt(np.sum(X ** 2, axis=0))  # Calcul de la norme L2 sur chaque colonne de X
        Xn = X * (1 / (norm2x + 1e-16))  # Normalisation de X (évite la division par zéro)


        HO = orthNNLS(X, WO, Xn)

        # Transposition du résultat
        W = HO.T

    if method == "SVCA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))

        options = {'average': 1}
        WO, K = SVCA(X, r, p, options=options)

        norm2x = np.sqrt(np.sum(X ** 2, axis=0))  # Calcul de la norme L2 sur chaque colonne de X
        Xn = X * (1 / (norm2x + 1e-16))  # Normalisation de X (évite la division par zéro)

        HO = orthNNLS(X, WO, Xn)

        # Transposition du résultat
        W = HO.T


    return W


def extract_w_v(W):
    """ Extracts w and v from W."""
    w = np.max(W, axis=1)
    v = np.argmax(W, axis=1)
    return w, v


def update_W(X, S, w, v):
    """
    Parameters:
    - X: Non-negative symmetric matrix (numpy.ndarray of size n x n)
    - S: Central matrix (numpy.ndarray of size r x r)
    - w: Vector of non-zero coefficients (numpy.ndarray of size n)
    - v: Indices of non-zero columns of W (numpy.ndarray of size n)

    Returns:
    - Updated w
    - Updated v

    """
    n = X.shape[0]
    r = S.shape[0]

    # Pré-calculs pour éviter une double boucle sur "n"
    wp2 = np.zeros(r)
    for k in range(r):
        for p in range(n):
            wp2[k] += (w[p] * S[v[p], k]) ** 2

    # Mise à jour de W
    for i in range(n):
        vi_new, wi_new, f_new = -1, -1, np.inf

        for k in range(r):
            # Calcul des coefficients pour la minimisation
            c3 = S[k, k] ** 2
            c1 = 2 * (wp2[k] - (w[i] * S[v[i], k]) ** 2) - 2 * S[k, k] * X[i, i]
            c0 = -4 * sum(X[i, p] * w[p] * S[v[p], k] for p in np.nonzero(X[i, :])[0] if p != i)

            # Résolution des racines avec la méthode de Cardan
            roots = cardan(4 * c3, 0, 2 * c1, c0)

            # Trouver la meilleure solution positive
            x = np.sqrt(r/n)
            min_value = c3 * (x ** 4) + c1 * (x ** 2) + c0 * x
            for sol in roots:
                value = c3 * (sol ** 4) + c1 * (sol ** 2) + c0 * sol
                if sol > 0 and value < min_value:
                    x, min_value = sol, value

            if c3 * x ** 4 + c1 * x ** 2 + c0 * x < f_new:
                f_new, wi_new, vi_new = c3 * x ** 4 + c1 * x ** 2 + c0 * x, x, k

        # Mise à jour de wp2
        for k in range(r):
            wp2[k] = wp2[k] - (w[i] * S[v[i], k]) ** 2 + (wi_new * S[vi_new, k]) ** 2

        # Mise à jour des valeurs de w et v
        w[i], v[i] = wi_new, vi_new

    # Normalisation des colonnes de W
    nw = np.zeros(r)
    for i in range(n):
        nw[v[i]] += w[i] ** 2
    nw = np.sqrt(nw)
    w /= nw[v]

    return w, v





def cardan(a, b, c, d):
    """ Cardano formula to solve ax^3+bx^2+cx+d=0"""
    if a == 0:
        if b == 0:
            if c == 0:
                roots = []
                return roots
            else:
                root1 = -d / c
                roots = [root1]
                return roots

        delta = c ** 2 - 4 * b * d
        root1 = (-c + math.sqrt(delta)) / (2 * b)
        root2 = (-c - math.sqrt(delta)) / (2 * b)

        if root1 == root2:
            roots = [root1]
        else:
            roots = [root1, root2]
        return roots

    p = -(b ** 2 / (3 * a ** 2)) + c / a
    q = ((2 * b ** 3) / (27 * a ** 3)) - ((9 * c * b) / (27 * a ** 2)) + (d / a)
    delta = -(4 * p ** 3 + 27 * q ** 2)

    if delta < 0:
        u = (-q + math.sqrt(-delta / 27)) / 2
        v = (-q - math.sqrt(-delta / 27)) / 2

        if u < 0:
            u = -(-u) ** (1 / 3)
        elif u > 0:
            u = u ** (1 / 3)
        else:
            u = 0

        if v < 0:
            v = -(-v) ** (1 / 3)
        elif v > 0:
            v = v ** (1 / 3)
        else:
            v = 0

        root1 = u + v - (b / (3 * a))
        roots = [root1]
        return roots

    elif delta == 0:
        if p == 0 and q == 0:
            root1 = 0
            roots = [root1]
        else:
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            roots = [root1, root2]
        return roots

    else:
        epsilon = -1e-300
        phi = math.acos(-q / (2 * math.sqrt(-27 / (p ** 3 + epsilon))))
        z1 = 2 * math.sqrt(-p / 3) * math.cos(phi / 3)
        z2 = 2 * math.sqrt(-p / 3) * math.cos((phi + 2 * math.pi) / 3)
        z3 = 2 * math.sqrt(-p / 3) * math.cos((phi + 4 * math.pi) / 3)

        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))

        roots = [root1, root2, root3]
        return roots


def update_S(X, r, w, v):
    """ Update of S with the closed form S=WTXW"""
    # Initialize S with zeros
    S = np.zeros((r, r))

    # Get the row indices, column indices, and values of the non-zero elements in the sparse matrix X
    i, j, val = find(X)

    # Loop through the non-zero elements of X
    for k in range(len(val)):
        S[v[i[k]], v[j[k]]] += w[i[k]] * w[j[k]] * val[k]

    return S



def compute_error(X, S, w, v):
    """ Computes error ||X - WSW'||_F / ||X||_F."""
    error=0
    i, j, val = find(X)
    for k in range(len(val)) :
        error+=(val[k]-S[v[i[k]],v[j[k]]]*w[i[k]]*w[j[k]])**2
    error=np.sqrt(error)/np.linalg.norm(X, 'fro')
    return error

def Community_detection_SVCA(X, r, numTrials=1,verbosity=1):
    """
        Perform community detection using the SVCA (Smooth VCA).

        Parameters:
        - X: ndarray or sparse matrix (Adjacency matrix of the graph)
        - r: int (Number of communities)
        - numTrials: int (Number of trials to find the best decomposition, default=1)
        - verbosity: int (Level of verbosity for printing progress, default=1)

        Returns:
        - w_best: ndarray (Importance of each node in its community)
        - v_best: ndarray (Community index for each node)
        - S_best: ndarray (Interaction matrix between communities)
        - error_best: float (Reconstruction error of the best trial)
        """

    if issparse(X):
        X=X.toarray()
    n = X.shape[0]
    error_best = float('inf')
    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    for trial in range(numTrials):

        # Placeholder for proper initialization functions
        W = initialize_W(X, r, method="SVCA")
        w, v = extract_w_v(W)
        # Normalisation des colonnes de W
        nw = np.zeros(r)
        for i in range(n):
            nw[v[i]] += w[i] ** 2
        nw = np.sqrt(nw)
        w /= nw[v]
        S = update_S(X, r, w, v)


        error = compute_error(X, S, w, v)




        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error

        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with SVCA: Error {error:.4e} | Best: {error_best:.4e}')

    return w_best, v_best, S_best, error_best

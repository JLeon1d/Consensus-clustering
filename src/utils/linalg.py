"""Linear algebra utilities for eigenvalue decomposition and discretization."""

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh


def eig1(
    A: np.ndarray,
    c: int,
    is_max: bool = True,
    is_sym: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top or bottom c eigenvectors and eigenvalues.

    This function mimics the MATLAB eig1 function, computing either the
    largest or smallest eigenvalues and their corresponding eigenvectors.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (n x n)
    c : int
        Number of eigenvectors to compute
    is_max : bool, default=True
        If True, compute largest eigenvalues; if False, compute smallest
    is_sym : bool, default=True
        If True, assume matrix is symmetric

    Returns
    -------
    eigvec : np.ndarray
        Eigenvectors (n x c)
    eigval : np.ndarray
        Eigenvalues (c,)

    """
    n = A.shape[0]
    c = min(c, n)

    if is_sym:
        A = np.maximum(A, A.T)

    use_sparse = issparse(A) or n > 1000

    try:
        if use_sparse:
            # Use sparse eigenvalue solver
            if is_max:
                eigval, eigvec = eigsh(A, k=c, which="LA")
            else:
                eigval, eigvec = eigsh(A, k=c, which="SA")
        else:
            # Use dense eigenvalue solver
            A_dense = A.toarray() if issparse(A) else A
            eigval, eigvec = eigh(A_dense)

    except Exception:
        # Fallback to dense solver
        A_dense = A.toarray() if issparse(A) else A
        eigval, eigvec = eigh(A_dense)

    if is_max:
        idx = np.argsort(eigval)[::-1]  # Descending order
    else:
        idx = np.argsort(eigval)

    idx = idx[:c]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    return eigvec, eigval


def discretisation(eigenvectors: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """
    Discretize continuous eigenvectors to obtain cluster assignments.

    This implements the discretization method from:
    "Multiclass spectral clustering" by Stella Yu and Jianbo Shi, 2003.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Continuous eigenvectors (n_samples x n_clusters)
    max_iter : int, default=20
        Maximum number of iterations

    Returns
    -------
    discrete_eigenvectors : np.ndarray
        Discrete cluster assignment matrix (n_samples x n_clusters)
        Each row has exactly one 1 and rest 0s

    """
    n, k = eigenvectors.shape

    vm = np.maximum(np.sqrt(np.sum(eigenvectors**2, axis=1, keepdims=True)), np.finfo(float).eps)
    eigenvectors = eigenvectors / vm

    R = np.zeros((k, k))
    R[:, 0] = eigenvectors[np.random.randint(0, n), :]

    c = np.zeros(n)
    for j in range(1, k):
        c = c + np.abs(eigenvectors @ R[:, j - 1])
        i = np.argmin(c)
        R[:, j] = eigenvectors[i, :]

    last_objective = 0
    for iteration in range(max_iter):
        discrete_eigenvectors = discretise_eigenvector_data(eigenvectors @ R)
        U, S, Vt = np.linalg.svd(discrete_eigenvectors.T @ eigenvectors, full_matrices=False)
        ncut_value = 2 * (n - np.sum(S))

        if np.abs(ncut_value - last_objective) < np.finfo(float).eps or iteration >= max_iter - 1:
            break

        last_objective = ncut_value
        R = Vt.T @ U.T

    return discrete_eigenvectors


def discretise_eigenvector_data(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Convert continuous eigenvectors to discrete cluster assignments.

    Each sample is assigned to the cluster corresponding to its
    maximum eigenvector component.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Continuous eigenvectors (n_samples x n_clusters)

    Returns
    -------
    discrete : np.ndarray
        Binary cluster assignment matrix (n_samples x n_clusters)
    """
    n, k = eigenvectors.shape

    norms = np.sqrt(np.sum(eigenvectors**2, axis=1, keepdims=True))
    norms = np.maximum(norms, np.finfo(float).eps)
    eigenvectors_normalized = eigenvectors / norms

    cluster_assignments = np.argmax(np.abs(eigenvectors_normalized), axis=1)

    discrete = np.zeros((n, k))
    discrete[np.arange(n), cluster_assignments] = 1

    return discrete
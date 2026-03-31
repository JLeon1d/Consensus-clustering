"""Discrete optimization for cluster assignment matrices G."""

from typing import List

import numpy as np
from scipy.sparse import csr_matrix, issparse


def optimize_G(
    AX: np.ndarray,
    G: List[np.ndarray],
    F: List[np.ndarray],
    W: np.ndarray,
    alpha: np.ndarray,
    lambda_: float,
    max_iter: int = 5,
) -> List[np.ndarray]:
    """
    Optimize cluster assignment matrices G using discrete optimization.

    This function performs greedy discrete optimization to update each
    cluster assignment matrix G_i by iteratively reassigning each sample
    to the cluster that minimizes the objective function.

    Parameters
    ----------
    AX : np.ndarray
        Transformed data matrix A^k X (n x d)
    G : list of np.ndarray
        List of cluster assignment matrices (each n x c)
    F : list of np.ndarray
        List of cluster center matrices (each c x d)
    W : np.ndarray
        Weighted consensus matrix (n x n)
    alpha : np.ndarray
        Weight vector (m,)
    lambda_ : float
        Regularization parameter
    max_iter : int, default=5
        Maximum number of iterations for each G_i

    Returns
    -------
    G_updated : list of np.ndarray
        Updated cluster assignment matrices

    Notes
    -----
    Each G_i is a binary matrix where G_i[j, k] = 1 if sample j is
    assigned to cluster k, and 0 otherwise. Each row has exactly one 1.
    """
    n, d = AX.shape
    c = F[0].shape[0]
    m = len(F)

    G_updated = [G_i.copy() for G_i in G]

    for i in range(m):
        if issparse(G_updated[i]):
            G_updated[i] = G_updated[i].toarray()

    for i in range(m):
        idx = np.argmax(G_updated[i], axis=1)
        alphaW = alpha[i] * W.T

        GG_i = np.zeros((n, n))
        for j in range(m):
            if j != i:
                GG_i += alpha[j] * (G_updated[j] @ G_updated[j].T)

        for iteration in range(max_iter):
            changed = False

            for j in range(n):
                AXj = AX[j, :]

                obj = np.zeros(c)

                for k in range(c):
                    obj[k] = np.sum((AXj - F[i][k, :]) ** 2)

                G_j = G_updated[i].copy()
                G_j[j, :] = 0

                W_j1 = alphaW[j, :].copy()
                W_j1[j] = 0
                W_j2 = alphaW[:, j].copy()
                W_j2[j] = 0

                obj2 = np.sum(G_j * (W_j2[:, np.newaxis] + W_j1[:, np.newaxis]), axis=0)
                obj += -2 * lambda_ * obj2

                GG_i_j = GG_i[:, j].copy()
                GG_i_j[j] = 0
                obj3 = np.sum(G_j * GG_i_j[:, np.newaxis], axis=0)
                obj += 4 * lambda_ * alpha[i] * obj3

                obj4 = np.sum(G_j, axis=0) * 2 * alpha[i] * alpha[i] * lambda_
                obj += obj4

                min_idx = np.argmin(obj)

                if idx[j] != min_idx:
                    G_updated[i][j, :] = 0
                    G_updated[i][j, min_idx] = 1
                    idx[j] = min_idx
                    changed = True

            if not changed:
                break

    return G_updated
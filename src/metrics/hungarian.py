"""Hungarian algorithm for optimal assignment problem."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve the assignment problem using the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Square cost matrix (n x n)

    Returns
    -------
    assignment : np.ndarray
        Optimal assignment vector (n,) where assignment[i] = j means
        row i is assigned to column j
    total_cost : float
        Total cost of the optimal assignment

    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    n = cost_matrix.shape[0]
    assignment = np.zeros(n, dtype=int)
    assignment[row_ind] = col_ind + 1  # +1 for 1-indexing

    total_cost = cost_matrix[row_ind, col_ind].sum()

    return assignment, total_cost


def best_map(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """
    Permute labels of L2 to best match L1 using Hungarian algorithm.

    Parameters
    ----------
    L1 : np.ndarray
        Reference labels (n_samples,)
    L2 : np.ndarray
        Labels to permute (n_samples,)

    Returns
    -------
    new_L2 : np.ndarray
        Permuted labels that best match L1

    """
    L1 = L1.ravel()
    L2 = L2.ravel()

    if L1.shape != L2.shape:
        raise ValueError("L1 and L2 must have the same shape")

    L1_min = L1.min()
    L2_min = L2.min()
    L1 = L1 - L1_min
    L2 = L2 - L2_min

    n_class = max(L1.max(), L2.max()) + 1
    G = np.zeros((n_class, n_class))

    for i in range(n_class):
        for j in range(n_class):
            G[i, j] = np.sum((L1 == i) & (L2 == j))

    row_ind, col_ind = linear_sum_assignment(-G)

    new_L2 = np.zeros_like(L2)
    for i in range(n_class):
        new_L2[L2 == i] = col_ind[i]

    return new_L2 + L1_min

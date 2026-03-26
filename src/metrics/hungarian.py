"""Hungarian algorithm for optimal assignment problem."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve the assignment problem using the Hungarian algorithm.

    This is a wrapper around scipy's linear_sum_assignment that matches
    the MATLAB hungarian function interface.

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
    # scipy's linear_sum_assignment minimizes cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create assignment vector (1-indexed to match MATLAB)
    n = cost_matrix.shape[0]
    assignment = np.zeros(n, dtype=int)
    assignment[row_ind] = col_ind + 1  # +1 for 1-indexing

    # Calculate total cost
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

    # Shift labels to start from 1
    L1 = L1 - L1.min() + 1
    L2 = L2 - L2.min() + 1

    # Create bipartite graph
    n_class = max(L1.max(), L2.max())
    G = np.zeros((n_class, n_class))

    for i in range(1, n_class + 1):
        for j in range(1, n_class + 1):
            G[i - 1, j - 1] = np.sum((L1 == i) & (L2 == j))

    # Use Hungarian algorithm to find optimal assignment
    # We want to maximize overlap, so negate the matrix
    assignment, _ = hungarian(-G)

    # Apply permutation to L2
    new_L2 = np.zeros_like(L2)
    for i in range(1, n_class + 1):
        new_L2[L2 == i] = assignment[i - 1]

    return new_L2
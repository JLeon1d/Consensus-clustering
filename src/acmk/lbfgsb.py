"""L-BFGS-B optimization wrappers using scipy.optimize."""

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .objectives import obj_f1_d2, obj_f2


def lbfgsb_optimize_A(
    A_init: np.ndarray,
    DWD: np.ndarray,
    GF: List[np.ndarray],
    GF_all: np.ndarray,
    X: np.ndarray,
    Lambda1: np.ndarray,
    mu: float,
    k: int,
    m: int,
    max_iter: int = 20,
    pgtol: float = 1e-3,
) -> np.ndarray:
    """
    Optimize matrix A using L-BFGS-B algorithm.

    Parameters
    ----------
    A_init : np.ndarray
        Initial A matrix (n x n)
    DWD : np.ndarray
        Normalized W matrix
    GF : list of np.ndarray
        List of G_i * F_i matrices
    GF_all : np.ndarray
        Sum of all GF matrices
    X : np.ndarray
        Data matrix (n x d)
    Lambda1 : np.ndarray
        Lagrange multiplier (n x n)
    mu : float
        Penalty parameter
    k : int
        Power of A matrix
    m : int
        Number of base clusterings
    max_iter : int, default=20
        Maximum number of iterations
    pgtol : float, default=1e-3
        Gradient tolerance

    Returns
    -------
    A_result : np.ndarray
        Optimized A matrix (n x n)
    """
    n = A_init.shape[0]
    x_init = A_init.ravel()

    def objective(x):
        return obj_f1_d2(x, DWD, GF, GF_all, X, Lambda1, mu, k, m, n)

    bounds = None

    result = minimize(
        objective,
        x_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "ftol": 1e-10,
            "gtol": pgtol,
            "disp": False,
        },
    )

    A_result = result.x.reshape(n, n)

    return A_result


def lbfgsb_optimize_W(
    W_init: np.ndarray,
    A: np.ndarray,
    G: List[np.ndarray],
    V: np.ndarray,
    Lambda1: np.ndarray,
    Lambda2: np.ndarray,
    mu: float,
    alpha: np.ndarray,
    lambda_: float,
    m: int,
    max_iter: int = 20,
    pgtol: float = 1e-3,
) -> np.ndarray:
    """
    Optimize matrix W using L-BFGS-B algorithm with box constraints [0, 1].

    Parameters
    ----------
    W_init : np.ndarray
        Initial W matrix (n x n)
    A : np.ndarray
        Affinity matrix (n x n)
    G : list of np.ndarray
        List of cluster assignment matrices
    V : np.ndarray
        Auxiliary variable (n x n)
    Lambda1 : np.ndarray
        Lagrange multiplier for A constraint (n x n)
    Lambda2 : np.ndarray
        Lagrange multiplier for W constraint (n x n)
    mu : float
        Penalty parameter
    alpha : np.ndarray
        Weight vector (m,)
    lambda_ : float
        Regularization parameter
    m : int
        Number of base clusterings
    max_iter : int, default=20
        Maximum number of iterations
    pgtol : float, default=1e-3
        Gradient tolerance

    Returns
    -------
    W_result : np.ndarray
        Optimized W matrix (n x n)
    """
    n = W_init.shape[0]
    x_init = W_init.ravel()

    def objective(x):
        return obj_f2(x, A, G, V, Lambda1, Lambda2, mu, alpha, lambda_, m, n)

    bounds = [(0.0, 1.0) for _ in range(n * n)]

    result = minimize(
        objective,
        x_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "ftol": 1e-10,
            "gtol": pgtol,
            "disp": False,
        },
    )

    W_result = result.x.reshape(n, n)

    return W_result
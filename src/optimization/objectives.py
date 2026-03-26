"""Objective functions for ACMK optimization."""

from typing import List, Tuple

import numpy as np


def obj_f1_d2(
    x: np.ndarray,
    DWD: np.ndarray,
    GF: List[np.ndarray],
    GF_all: np.ndarray,
    X: np.ndarray,
    Lambda1: np.ndarray,
    mu: float,
    k: int,
    m: int,
    n: int,
) -> Tuple[float, np.ndarray]:
    """
    Objective function for optimizing matrix A.

    This function computes the objective value and gradient for the
    A matrix optimization subproblem in the ACMK algorithm.

    Parameters
    ----------
    x : np.ndarray
        Flattened A matrix (n*n,)
    DWD : np.ndarray
        Normalized W matrix: D^(-1/2) W D^(-1/2)
    GF : list of np.ndarray
        List of G_i * F_i matrices for each base clustering
    GF_all : np.ndarray
        Sum of all GF matrices
    X : np.ndarray
        Data matrix (n x d)
    Lambda1 : np.ndarray
        Lagrange multiplier matrix (n x n)
    mu : float
        Penalty parameter
    k : int
        Power of A matrix
    m : int
        Number of base clusterings
    n : int
        Number of samples

    Returns
    -------
    obj : float
        Objective function value
    grad : np.ndarray
        Gradient (flattened, n*n)
    """
    A = x.reshape(n, n)

    AX = [X]
    for i in range(k):
        AX.append(A @ AX[-1])

    AAX = AX[k]

    XAA = []
    if k > 1:
        XAA.append(AAX.T @ A)
        for i in range(1, k - 1):
            XAA.append(XAA[-1] @ A)

    GFA = [GF_all]
    for r in range(1, k):
        GFA.append(GFA[-1] @ A)

    obj1 = 0.0
    for i in range(m):
        diff = AAX - GF[i]
        obj1 += np.sum(diff**2)

    DWD_constraint = (np.eye(n) + DWD) / 2
    obj2 = np.sum(Lambda1 * A)
    obj3 = np.sum((A - DWD_constraint) ** 2)

    obj = obj1 + obj2 + mu / 2 * obj3

    grad1 = AX[k] @ AAX.T
    for r in range(k - 1):
        grad1 += AX[k - r - 1] @ XAA[r]
    grad1 = 2 * grad1.T * m

    grad2 = AX[0] @ GFA[k - 1]
    for r in range(k - 1):
        grad2 += AX[r + 1] @ GFA[k - r - 1]
    grad2 = grad2.T

    grad12 = grad1 - 2 * grad2
    grad = grad12 + Lambda1 + mu * A - mu * DWD_constraint

    grad = grad.ravel()

    return obj, grad


def obj_f2(
    x: np.ndarray,
    A: np.ndarray,
    G: List[np.ndarray],
    V: np.ndarray,
    Lambda1: np.ndarray,
    Lambda2: np.ndarray,
    mu: float,
    alpha: np.ndarray,
    lambda_: float,
    m: int,
    n: int,
) -> Tuple[float, np.ndarray]:
    """
    Objective function for optimizing matrix W.

    This function computes the objective value and gradient for the
    W matrix optimization subproblem in the ACMK algorithm.

    Parameters
    ----------
    x : np.ndarray
        Flattened W matrix (n*n,)
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
    n : int
        Number of samples

    Returns
    -------
    obj : float
        Objective function value
    grad : np.ndarray
        Gradient (flattened, n*n)
    """
    W = x.reshape(n, n)

    GG = np.zeros((n, n))
    for v in range(m):
        GG += alpha[v] * (G[v] @ G[v].T)

    obj1 = lambda_ * np.sum((W - GG) ** 2)

    dd = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), np.finfo(float).eps))
    DWD1 = W * dd[:, np.newaxis]
    DWD2 = W * dd[np.newaxis, :]
    DWD = DWD1 * dd[np.newaxis, :]

    obj2 = -np.sum(Lambda1 * DWD) / 2

    obj3 = np.sum(Lambda2 * W)

    obj4 = mu / 2 * np.sum((A - 0.5 * (np.eye(n) + DWD)) ** 2)

    obj5 = mu / 2 * np.sum((W - V) ** 2)

    obj = obj1 + obj2 + obj3 + obj4 + obj5

    grad1 = 2 * lambda_ * (W - GG)

    grad21 = Lambda1 * dd[:, np.newaxis] * dd[np.newaxis, :]
    grad21 = -0.5 * grad21

    tmp1 = (Lambda1 * DWD1).sum(axis=1) + (Lambda1 * DWD2).sum(axis=0)
    tmp1 = tmp1 * dd**3
    grad2 = grad21 + 0.25 * tmp1[:, np.newaxis]

    grad5 = mu * (W - V)

    dd2 = 1.0 / np.maximum(W.sum(axis=1), np.finfo(float).eps)
    DWD21 = W * dd2[:, np.newaxis]
    DWD22 = W * dd2[np.newaxis, :]
    DWD23 = DWD21 * dd2[np.newaxis, :]

    tmp2 = (DWD21 * W).sum(axis=1) + (DWD22 * W).sum(axis=0)
    tmp2 = tmp2 * dd2**2
    grad41 = 2 * DWD23 - tmp2[:, np.newaxis]

    B = A - np.eye(n) / 2
    grad421 = B * dd[:, np.newaxis] * dd[np.newaxis, :]

    tmp3 = (B * DWD1).sum(axis=1) + (B * DWD2).sum(axis=0)
    tmp3 = tmp3 * dd**3
    grad42 = grad421 - 0.5 * tmp3[:, np.newaxis]

    grad4 = mu / 8 * grad41 - mu / 2 * grad42

    grad = grad1 + grad2 + Lambda2 + grad4 + grad5

    grad = grad.ravel()

    return obj, grad
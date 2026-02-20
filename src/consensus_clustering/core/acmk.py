"""Main ACMK (Adaptive Consensus Multiple Kernel) clustering algorithm."""

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import issparse

from ..clustering.kmeans import litekmeans
from ..optimization.lbfgsb import lbfgsb_optimize_A, lbfgsb_optimize_W
from ..optimization.optimize_g import optimize_G
from ..utils.linalg import discretisation, eig1


class ACMK:
    """
    Adaptive Consensus Multiple Kernel (ACMK) clustering algorithm.

    This algorithm performs consensus clustering by combining multiple base
    clusterings through an ADMM optimization framework.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    m_base : int
        Number of base clusterings
    lambda_ : float, default=0.1
        Regularization parameter
    k_power : int, default=3
        Power of affinity matrix A
    max_iter : int, default=20
        Maximum number of ADMM iterations
    mu_init : float, default=1.0
        Initial penalty parameter
    rho : float, default=1.05
        Penalty parameter increase rate
    verbose : bool, default=False
        Whether to print progress information

    Attributes
    ----------
    labels_spectral_ : np.ndarray
        Cluster labels from spectral clustering
    labels_kmeans_ : np.ndarray
        Cluster labels from k-means on transformed space
    W_ : np.ndarray
        Final weighted consensus matrix
    A_ : np.ndarray
        Final affinity matrix
    G_ : list of np.ndarray
        Final cluster assignment matrices
    F_ : list of np.ndarray
        Final cluster center matrices
    alpha_ : np.ndarray
        Final weight vector for base clusterings
    objective_history_ : list
        History of objective function values

    Examples
    --------
    >>> from consensus_clustering.clustering.base_generation import generate_base_clusterings
    >>> X = np.random.randn(100, 10)
    >>> base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)
    >>> acmk = ACMK(n_clusters=5, m_base=10, lambda_=0.1)
    >>> acmk.fit(X, **base_data)
    >>> labels = acmk.predict(method='spectral')
    """

    def __init__(
        self,
        n_clusters: int,
        m_base: int,
        lambda_: float = 0.1,
        k_power: int = 3,
        max_iter: int = 20,
        mu_init: float = 1.0,
        rho: float = 1.05,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.m_base = m_base
        self.lambda_ = lambda_
        self.k_power = k_power
        self.max_iter = max_iter
        self.mu_init = mu_init
        self.rho = rho
        self.verbose = verbose

        # Attributes set during fit
        self.labels_spectral_: Optional[np.ndarray] = None
        self.labels_kmeans_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None
        self.G_: Optional[List[np.ndarray]] = None
        self.F_: Optional[List[np.ndarray]] = None
        self.alpha_: Optional[np.ndarray] = None
        self.objective_history_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        W: np.ndarray,
        G: List[np.ndarray],
        F: List[np.ndarray],
        **kwargs
    ) -> "ACMK":
        """
        Fit the ACMK model.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)
        W : np.ndarray
            Initial consensus matrix (n_samples x n_samples)
        G : list of np.ndarray
            Initial cluster assignment matrices
        F : list of np.ndarray
            Initial cluster center matrices

        Returns
        -------
        self : ACMK
            Fitted estimator
        """
        n, d = X.shape
        m = self.m_base
        c = self.n_clusters
        k = self.k_power

        alpha = np.ones(m) / m
        V = W.copy()
        A = W.copy()
        G = [G_i.toarray() if issparse(G_i) else G_i for G_i in G]
        Lambda1 = np.zeros((n, n))
        Lambda2 = np.zeros((n, n))
        mu = self.mu_init

        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter}")

            GF = [G[j] @ F[j] for j in range(m)]
            GF_all = sum(GF)

            dd = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), np.finfo(float).eps))
            DWD = W * dd[:, np.newaxis] * dd[np.newaxis, :]

            A = lbfgsb_optimize_A(
                A, DWD, GF, GF_all, X, Lambda1, mu, k, m, max_iter=20, pgtol=1e-3
            )

            W = lbfgsb_optimize_W(
                W, A, G, V, Lambda1, Lambda2, mu, alpha, self.lambda_, m, max_iter=20, pgtol=1e-3
            )

            dd = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), np.finfo(float).eps))
            DWD = W * dd[:, np.newaxis] * dd[np.newaxis, :]

            AX = X.copy()
            for _ in range(k):
                AX = A @ AX

            G = optimize_G(AX, G, F, W, alpha, self.lambda_, max_iter=5)

            for j in range(m):
                GG = G[j].T @ G[j]
                gg = 1.0 / np.maximum(np.diag(GG), np.finfo(float).eps)
                GX = G[j].T @ AX
                F[j] = GX * gg[:, np.newaxis]

            V = W + Lambda2 / mu
            V = (V + V.T) / 2

            H = np.zeros((m, m))
            for j in range(m):
                for p in range(m):
                    tmp = G[j].T @ G[p]
                    tmp = tmp @ G[p].T
                    H[j, p] = np.sum(G[j].T * tmp)
            H = H + H.T

            f = np.zeros(m)
            for j in range(m):
                tmp2 = W.T @ G[j]
                f[j] = np.sum(tmp2 * G[j])
            f = -2.0 * f

            result = minimize(
                lambda a: 0.5 * a @ H @ a + f @ a,
                alpha,
                method="SLSQP",
                bounds=[(0, 1) for _ in range(m)],
                constraints={"type": "eq", "fun": lambda a: np.sum(a) - 1},
                options={"disp": False},
            )
            alpha = result.x

            Lambda1 = Lambda1 + mu * (A - 0.5 * (np.eye(n) + DWD))
            Lambda2 = Lambda2 + mu * (W - V)
            mu = mu * self.rho

            if self.verbose:
                print(f"  mu = {mu:.4f}, alpha = {alpha}")

        self.W_ = W
        self.A_ = A
        self.G_ = G
        self.F_ = F
        self.alpha_ = alpha

        self._compute_final_labels(X, A, k, n, c)

        return self

    def _compute_final_labels(self, X: np.ndarray, A: np.ndarray, k: int, n: int, c: int):
        """Compute final cluster labels using spectral clustering and k-means."""
        L = 2 * (np.eye(n) - A)
        L = (L + L.T) / 2
        eigvec, _ = eig1(L, c + 1, is_max=False, is_sym=True)
        eigvec = eigvec[:, 1:]

        Y = discretisation(eigvec)
        self.labels_spectral_ = np.argmax(Y, axis=1)

        AX = X.copy()
        for _ in range(k):
            AX = A @ AX

        self.labels_kmeans_, _, _ = litekmeans(AX, self.n_clusters, n_init=20)

    def predict(self, method: str = "spectral") -> np.ndarray:
        """
        Get cluster labels.

        Parameters
        ----------
        method : str, default='spectral'
            Method to use: 'spectral' or 'kmeans'

        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        if method == "spectral":
            return self.labels_spectral_
        elif method == "kmeans":
            return self.labels_kmeans_
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit_predict(
        self,
        X: np.ndarray,
        W: np.ndarray,
        G: List[np.ndarray],
        F: List[np.ndarray],
        method: str = "spectral",
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        W : np.ndarray
            Initial consensus matrix
        G : list of np.ndarray
            Initial cluster assignment matrices
        F : list of np.ndarray
            Initial cluster center matrices
        method : str, default='spectral'
            Method to use: 'spectral' or 'kmeans'

        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        self.fit(X, W, G, F, **kwargs)
        return self.predict(method=method)
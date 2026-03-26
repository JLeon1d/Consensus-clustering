"""SDGCA (Similarity and Dissimilarity Guided Co-association) clustering algorithm.

This module implements the SDGCA algorithm for ensemble clustering as described in:
"Similarity and Dissimilarity Guided Co-association Matrix Construction for Ensemble Clustering"
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import issparse


class SDGCA:
    """
    Similarity and Dissimilarity Guided Co-association (SDGCA) clustering algorithm.

    This algorithm performs ensemble clustering by constructing a refined co-association
    matrix using similarity and dissimilarity guidance through an ADMM optimization framework.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    lambda_param : float, default=0.09
        Regularization parameter for NWCA computation
    eta : float, default=0.75
        Threshold for high-confidence co-association (HC)
    theta : float, default=0.65
        Threshold for must-link affinity (MLA)
    max_iter : int, default=300
        Maximum number of ADMM iterations
    tau : float, default=0.8
        Threshold for dissimilarity matrix
    k_rw : int, default=20
        Number of iterations for random walk
    beta_rw : float, default=1.0
        Beta parameter for random walk
    mu1_init : float, default=1.0
        Initial penalty parameter for similarity
    mu2_init : float, default=1.0
        Initial penalty parameter for dissimilarity
    rho1 : float, default=1.1
        Penalty parameter increase rate for similarity
    rho2 : float, default=1.1
        Penalty parameter increase rate for dissimilarity
    tol : float, default=1e-3
        Convergence tolerance
    verbose : bool, default=False
        Whether to print progress information

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels from hierarchical clustering
    W_ : np.ndarray
        Final refined co-association matrix
    S_ : np.ndarray
        Final similarity guidance matrix
    D_ : np.ndarray
        Final dissimilarity guidance matrix
    NWCA_ : np.ndarray
        Normalized weighted co-association matrix
    CA_ : np.ndarray
        Basic co-association matrix
    """

    def __init__(
        self,
        n_clusters: int,
        lambda_param: float = 0.09,
        eta: float = 0.75,
        theta: float = 0.65,
        max_iter: int = 300,
        tau: float = 0.8,
        k_rw: int = 20,
        beta_rw: float = 1.0,
        mu1_init: float = 1.0,
        mu2_init: float = 1.0,
        rho1: float = 1.1,
        rho2: float = 1.1,
        tol: float = 1e-3,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.lambda_param = lambda_param
        self.eta = eta
        self.theta = theta
        self.max_iter = max_iter
        self.tau = tau
        self.k_rw = k_rw
        self.beta_rw = beta_rw
        self.mu1_init = mu1_init
        self.mu2_init = mu2_init
        self.rho1 = rho1
        self.rho2 = rho2
        self.tol = tol
        self.verbose = verbose

        # Attributes set during fit
        self.labels_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.D_: Optional[np.ndarray] = None
        self.NWCA_: Optional[np.ndarray] = None
        self.CA_: Optional[np.ndarray] = None

    def fit(self, base_clusterings: np.ndarray) -> "SDGCA":
        """
        Fit the SDGCA model.

        Parameters
        ----------
        base_clusterings : np.ndarray
            Base clustering matrix of shape (n_samples, n_base_clusterings)
            where each column contains cluster labels for one base clustering

        Returns
        -------
        self : SDGCA
            Fitted estimator
        """
        # Convert sparse to dense if needed
        if issparse(base_clusterings):
            base_clusterings = base_clusterings.toarray()

        n_samples, m_base = base_clusterings.shape

        # Get all segments (cluster indicators)
        bcs, base_cls_segs = self._get_all_segs(base_clusterings)

        # Compute basic co-association matrix
        self.CA_ = (base_cls_segs.T @ base_cls_segs) / m_base

        # Compute NECI (Normalized Entropy of Cluster Indicator)
        neci = self._compute_neci(bcs, base_cls_segs, self.lambda_param)

        # Compute NWCA (Normalized Weighted Co-Association)
        self.NWCA_ = self._compute_nwca(base_cls_segs, neci, m_base)

        # Check if we should use simple NWCA or full SDGCA
        if self.eta > 1:
            # Use NWCA directly
            self.W_ = self.NWCA_
            self.S_ = None
            self.D_ = None
        else:
            # Compute high-confidence co-association (HC)
            HC = self.CA_.copy()
            HC[HC < self.eta] = 0
            L = np.diag(HC.sum(axis=1)) - HC

            # Compute must-link affinity (MLA)
            MLA = self.CA_.copy()
            MLA[MLA < self.theta] = 0

            # Compute similarity guidance (ML)
            ML = self._compute_s(self.NWCA_, MLA)

            # Compute dissimilarity guidance (CL)
            CL = self._compute_d(bcs, base_cls_segs)

            # Apply constraint: ML and CL should not overlap
            ML[CL > 0] = 0

            # Optimize S and D using ADMM
            self.S_, self.D_ = self._optimize_sdgca(L, ML, CL)

            # Compute final refined co-association matrix
            self.W_ = self._compute_w(self.S_, self.D_, self.NWCA_)

        # Get final clustering result
        self.labels_ = self._get_clustering_result(self.W_, self.n_clusters)

        return self

    def _get_all_segs(
        self, base_clusterings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert base clusterings to segment representation.

        Parameters
        ----------
        base_clusterings : np.ndarray
            Base clustering matrix (n_samples, n_base_clusterings)

        Returns
        -------
        bcs : np.ndarray
            Renumbered base clusterings
        base_cls_segs : np.ndarray
            Binary cluster indicator matrix (n_total_clusters, n_samples)
        """
        n_samples, m_base = base_clusterings.shape
        bcs = base_clusterings.copy()

        # Get number of clusters in each base clustering
        n_cls_orig = bcs.max(axis=0)

        # Renumber clusters to be unique across all base clusterings
        cumsum_cls = np.cumsum(n_cls_orig)
        offsets = np.concatenate([[0], cumsum_cls[:-1]])
        bcs = bcs + offsets[np.newaxis, :]

        # Total number of clusters across all base clusterings
        n_total_cls = cumsum_cls[-1]

        # Create binary indicator matrix
        base_cls_segs = np.zeros((n_total_cls, n_samples))
        for i in range(n_samples):
            for j in range(m_base):
                cluster_id = int(bcs[i, j]) - 1  # Convert to 0-indexed and ensure Python int
                base_cls_segs[cluster_id, i] = 1

        return bcs, base_cls_segs

    def _compute_neci(
        self, bcs: np.ndarray, base_cls_segs: np.ndarray, para_theta: float
    ) -> np.ndarray:
        """
        Compute Normalized Entropy of Cluster Indicator (NECI).

        Parameters
        ----------
        bcs : np.ndarray
            Renumbered base clusterings
        base_cls_segs : np.ndarray
            Binary cluster indicator matrix
        para_theta : float
            Parameter for NECI computation

        Returns
        -------
        neci : np.ndarray
            NECI values for each cluster
        """
        m_base = bcs.shape[1]
        entropies = self._get_all_cls_entropy(bcs, base_cls_segs)
        neci = np.exp(-entropies / para_theta / m_base)
        return neci

    def _get_all_cls_entropy(
        self, bcs: np.ndarray, base_cls_segs: np.ndarray
    ) -> np.ndarray:
        """
        Compute entropy for all clusters.

        Parameters
        ----------
        bcs : np.ndarray
            Renumbered base clusterings
        base_cls_segs : np.ndarray
            Binary cluster indicator matrix

        Returns
        -------
        entropies : np.ndarray
            Entropy values for each cluster
        """
        n_total_cls = base_cls_segs.shape[0]
        entropies = np.zeros(n_total_cls)
        norm_k = self._compute_norm_k(bcs)

        for i in range(n_total_cls):
            # Get samples in this cluster
            mask = base_cls_segs[i, :] != 0
            if not mask.any():
                continue
            part_bcs = bcs[mask, :]
            entropies[i] = self._get_one_cls_entropy(part_bcs, norm_k[i])

        return entropies

    def _get_one_cls_entropy(self, part_bcs: np.ndarray, norm_k_i: float) -> float:
        """
        Compute entropy for one cluster.

        Parameters
        ----------
        part_bcs : np.ndarray
            Base clusterings for samples in this cluster
        norm_k_i : float
            Normalization factor

        Returns
        -------
        entropy : float
            Entropy value
        """
        if norm_k_i == 0:
            return 0.0

        entropy = 0.0
        for i in range(part_bcs.shape[1]):
            col = np.sort(part_bcs[:, i])
            unique_vals, counts = np.unique(col, return_counts=True)

            if len(unique_vals) <= 1:
                continue

            probs = counts / counts.sum()
            entropy -= np.sum(probs * np.log2(probs + 1e-10))

        entropy = entropy / np.log2(norm_k_i + 1e-10)
        return entropy

    def _compute_norm_k(self, bcs: np.ndarray) -> np.ndarray:
        """
        Compute normalization factors for entropy.

        Parameters
        ----------
        bcs : np.ndarray
            Renumbered base clusterings

        Returns
        -------
        norm_k : np.ndarray
            Normalization factors
        """
        clust = bcs.max(axis=0)
        norm_k = np.zeros(int(clust[-1]), dtype=int)

        # First base clustering
        norm_k[: int(clust[0])] = int(clust[0])

        # Remaining base clusterings
        for i in range(len(clust) - 1):
            start = int(clust[i])
            end = int(clust[i + 1])
            norm_k[start:end] = int(clust[i + 1]) - int(clust[i])

        return norm_k

    def _compute_nwca(
        self, base_cls_segs: np.ndarray, neci: np.ndarray, m_base: int
    ) -> np.ndarray:
        """
        Compute Normalized Weighted Co-Association matrix.

        Parameters
        ----------
        base_cls_segs : np.ndarray
            Binary cluster indicator matrix
        neci : np.ndarray
            NECI values
        m_base : int
            Number of base clusterings

        Returns
        -------
        nwca : np.ndarray
            Normalized weighted co-association matrix
        """
        # Weight each cluster by its NECI
        weighted_segs = base_cls_segs * neci[:, np.newaxis]

        # Compute weighted co-association
        nwca = (weighted_segs.T @ base_cls_segs) / m_base

        # Normalize
        max_val = nwca.max()
        if max_val > 0:
            nwca = nwca / max_val

        # Set diagonal to 1
        np.fill_diagonal(nwca, 1.0)

        return nwca

    def _compute_s(self, nwca: np.ndarray, mla: np.ndarray) -> np.ndarray:
        """
        Compute similarity guidance matrix (ML).

        Parameters
        ----------
        nwca : np.ndarray
            Normalized weighted co-association matrix
        mla : np.ndarray
            Must-link affinity matrix

        Returns
        -------
        ml : np.ndarray
            Similarity guidance matrix
        """
        ml = nwca.copy()
        ml[mla == 0] = 0

        # Normalize
        min_val = ml.min()
        ml = ml - min_val
        max_val = ml.max()
        if max_val > 0:
            ml = ml / max_val

        # Scale and shift
        ml = ml / 5.0 + 0.8
        ml[ml == 0.8] = 0

        return ml

    def _compute_d(self, bcs: np.ndarray, base_cls_segs: np.ndarray) -> np.ndarray:
        """
        Compute dissimilarity guidance matrix (CL).

        Parameters
        ----------
        bcs : np.ndarray
            Renumbered base clusterings
        base_cls_segs : np.ndarray
            Binary cluster indicator matrix

        Returns
        -------
        cl : np.ndarray
            Dissimilarity guidance matrix
        """
        n_samples, m_base = bcs.shape

        # Compute Jaccard similarity between clusters
        sim_of_cluster = self._jaccard_similarity(base_cls_segs)

        # Compute random walk on cluster similarity
        rw_of_cluster = self._random_walk_of_cluster(sim_of_cluster)

        # Set diagonal to 1
        np.fill_diagonal(rw_of_cluster, 1.0)

        # Compute dissimilarity
        dis_of_cluster = 1 - rw_of_cluster

        # Aggregate dissimilarity across base clusterings
        D = np.zeros((n_samples, n_samples))
        for m in range(m_base):
            cluster_ids = (bcs[:, m] - 1).astype(int)  # Convert to 0-indexed Python ints
            for i in range(n_samples):
                for j in range(n_samples):
                    D[i, j] += dis_of_cluster[int(cluster_ids[i]), int(cluster_ids[j])]

        D = D / m_base
        D[D < self.tau] = 0

        return D

    def _jaccard_similarity(self, a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Jaccard similarity between rows of matrices.

        Parameters
        ----------
        a : np.ndarray
            First matrix
        b : np.ndarray, optional
            Second matrix (if None, compute similarity within a)

        Returns
        -------
        similarity : np.ndarray
            Jaccard similarity matrix
        """
        if b is None:
            b = a

        # Compute intersection (dot product for binary matrices)
        intersection = a @ b.T

        # Compute union
        a_squared = (a ** 2).sum(axis=1, keepdims=True)
        b_squared = (b ** 2).sum(axis=1, keepdims=True)
        union = a_squared + b_squared.T - intersection

        # Avoid division by zero
        union = np.maximum(union, 1e-10)

        similarity = intersection / union
        return similarity

    def _random_walk_of_cluster(self, W: np.ndarray) -> np.ndarray:
        """
        Compute random walk similarity on cluster graph.

        Parameters
        ----------
        W : np.ndarray
            Similarity matrix between clusters

        Returns
        -------
        R : np.ndarray
            Random walk similarity matrix
        """
        n = W.shape[0]

        # Remove diagonal
        W = W - np.diag(np.diag(W))

        # Compute transition matrix
        row_sums = W.sum(axis=1)
        D_inv = np.zeros_like(W)
        non_zero = row_sums > 0
        D_inv[non_zero, non_zero] = 1.0 / row_sums[non_zero]

        W_tilde = D_inv @ W

        # Compute random walk
        tmp_O = W_tilde.copy()
        O_tilde = W_tilde @ W_tilde.T

        for i in range(self.k_rw - 1):
            tmp_O = tmp_O @ W_tilde
            O_tilde = O_tilde + self.beta_rw * (tmp_O @ tmp_O.T)

        # Normalize
        O_i = np.diag(O_tilde)[:, np.newaxis]
        denominator = np.sqrt(O_i @ O_i.T)
        denominator = np.maximum(denominator, 1e-10)
        R = O_tilde / denominator

        # Handle isolated nodes
        isolated_idx = row_sums < 1e-9
        if isolated_idx.any():
            R[isolated_idx, :] = 0
            R[:, isolated_idx] = 0

        return R

    def _optimize_sdgca(
        self, L: np.ndarray, ML: np.ndarray, CL: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize similarity (S) and dissimilarity (D) matrices using ADMM.

        Parameters
        ----------
        L : np.ndarray
            Laplacian matrix
        ML : np.ndarray
            Must-link (similarity guidance) matrix
        CL : np.ndarray
            Cannot-link (dissimilarity guidance) matrix

        Returns
        -------
        S : np.ndarray
            Optimized similarity matrix
        D : np.ndarray
            Optimized dissimilarity matrix
        """
        n = L.shape[0]
        I = np.eye(n)

        # Initialize variables
        S = np.zeros((n, n))
        D = np.zeros((n, n))
        Y1 = np.zeros((n, n))
        Y2 = np.zeros((n, n))
        F1 = np.zeros((n, n))
        F2 = np.zeros((n, n))

        mu1 = self.mu1_init
        mu2 = self.mu2_init

        # Precompute inverse matrices
        inv1 = np.linalg.inv(2 * L + 2 * mu1 * I)
        inv2 = np.linalg.inv(2 * L + 2 * mu2 * I)

        for iteration in range(self.max_iter):
            S_old = S.copy()
            D_old = D.copy()

            # Update S
            S = inv1 @ (2 * mu1 * F1 - D.T - Y1)

            # Update F1
            F1 = Y1 / (2 * mu1) + S
            F1[ML > 0] = 0
            F1 = np.clip(F1, 0, 1)
            F1 = F1 + ML
            F1 = (F1 + F1.T) / 2

            # Update D
            D = inv2 @ (2 * mu2 * F2 - S.T - Y2)

            # Update F2
            F2 = Y2 / (2 * mu2) + D
            F2[CL > 0] = 0
            F2 = np.clip(F2, 0, 1)
            F2 = F2 + CL
            F2 = (F2 + F2.T) / 2

            # Update dual variables
            Y1 = Y1 + mu1 * (S - F1)
            Y2 = Y2 + mu2 * (D - F2)

            # Update penalty parameters
            mu1 = min(mu1 * self.rho1, 1e6)
            mu2 = min(mu2 * self.rho2, 1e6)

            # Recompute inverse matrices with new mu
            inv1 = np.linalg.inv(2 * L + 2 * mu1 * I)
            inv2 = np.linalg.inv(2 * L + 2 * mu2 * I)

            # Check convergence
            errors = [
                np.linalg.norm(S - S_old, "fro"),
                np.linalg.norm(D - D_old, "fro"),
                np.linalg.norm(S - F1, "fro"),
                np.linalg.norm(D - F2, "fro"),
            ]

            if self.verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, max error: {max(errors):.6f}")

            if max(errors) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        return S, D

    def _compute_w(
        self, S: np.ndarray, D: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        """
        Compute final refined co-association matrix.

        Parameters
        ----------
        S : np.ndarray
            Similarity guidance matrix
        D : np.ndarray
            Dissimilarity guidance matrix
        W : np.ndarray
            Initial co-association matrix (NWCA)

        Returns
        -------
        W_star : np.ndarray
            Refined co-association matrix
        """
        # Clip and symmetrize
        S = np.clip(S, 0, 1)
        S = (S + S.T) / 2
        D = np.clip(D, 0, 1)
        D = (D + D.T) / 2

        # Compute two components
        W1 = 1 - (1 - S + D) * (1 - W)
        W2 = (1 + S - D) * W

        # Apply flag-based selection
        flag = S - D
        W1[flag < 0] = 0
        W2[flag >= 0] = 0

        W_star = W1 + W2
        return W_star

    def _get_clustering_result(
        self, CA: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """
        Get final clustering result using hierarchical clustering.

        Parameters
        ----------
        CA : np.ndarray
            Co-association matrix
        n_clusters : int
            Number of clusters

        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        # Clip values
        CA = np.clip(CA, 0, 1)

        # Make symmetric
        CA = np.maximum(CA, CA.T)

        # Convert to distance
        np.fill_diagonal(CA, 0)
        s = squareform(CA)
        d = 1 - s

        # Hierarchical clustering
        Z = linkage(d, method="average")
        labels = fcluster(Z, n_clusters, criterion="maxclust")

        return labels

    def predict(self) -> np.ndarray:
        """
        Get cluster labels.

        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.labels_

    def fit_predict(self, base_clusterings: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        base_clusterings : np.ndarray
            Base clustering matrix

        Returns
        -------
        labels : np.ndarray
            Cluster labels
        """
        self.fit(base_clusterings)
        return self.predict()

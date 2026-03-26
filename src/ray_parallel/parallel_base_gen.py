"""Parallel base clustering generation using Ray."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
from scipy.sparse import csr_matrix

from ..clustering.kmeans import litekmeans
from ..metrics import clustering_measure


def _run_single_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int,
    seed: Optional[int],
    y_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Run a single k-means clustering.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    n_clusters : int
        Number of clusters
    n_init : int
        Number of k-means initializations
    seed : int or None
        Random seed
    y_true : np.ndarray or None
        True labels for evaluation

    Returns
    -------
    labels : np.ndarray
        Cluster labels
    centers : np.ndarray
        Cluster centers
    G_dense : np.ndarray
        Dense cluster assignment matrix
    metrics : dict or None
        Evaluation metrics if y_true provided
    """

    labels, centers, _ = litekmeans(
        X, n_clusters=n_clusters, max_iter=100, n_init=n_init, random_state=seed
    )

    n_samples = X.shape[0]
    G_dense = np.zeros((n_samples, n_clusters))
    for j in range(n_samples):
        G_dense[j, labels[j]] = 1

    metrics = None
    if y_true is not None:
        metrics = clustering_measure(y_true, labels)

    return labels, centers, G_dense, metrics


def generate_base_clusterings_parallel(
    X: np.ndarray,
    n_clusters: int,
    m_base: int = 10,
    n_init: int = 1,
    random_state: Optional[int] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Generate multiple base clusterings using k-means in parallel with Ray.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    m_base : int, default=10
        Number of base clusterings to generate
    n_init : int, default=1
        Number of k-means initializations per base clustering
    random_state : int or None, default=None
        Random seed for reproducibility
    y_true : np.ndarray or None, default=None
        True labels for evaluation (optional)

    Returns
    -------
    base_data : dict
        Dictionary containing:
        - 'W': Initial consensus matrix (n_samples x n_samples)
        - 'G': List of cluster assignment matrices (m_base matrices)
        - 'F': List of cluster center matrices (m_base matrices)
        - 'labels': List of cluster labels (m_base arrays)
        - 'metrics': Evaluation metrics if y_true provided (optional)
    """

    n_samples = X.shape[0]

    X_ref = ray.put(X)
    y_true_ref = ray.put(y_true) if y_true is not None else None

    remote_kmeans = ray.remote(_run_single_kmeans)

    seeds = [None if random_state is None else random_state + i for i in range(m_base)]

    futures = [
        remote_kmeans.remote(X_ref, n_clusters, n_init, seed, y_true_ref)
        for seed in seeds
    ]

    results = ray.get(futures)

    W = np.zeros((n_samples, n_samples))
    G_list = []
    F_list = []
    labels_list = []
    metrics_list = [] if y_true is not None else None

    for labels, centers, G_dense, metrics in results:
        G = csr_matrix(G_dense)
        G_list.append(G)
        F_list.append(centers)
        labels_list.append(labels)

        W += G_dense @ G_dense.T

        if y_true is not None and metrics is not None:
            metrics_list.append(metrics)

    W = W / m_base

    base_data = {
        "W": W,
        "G": G_list,
        "F": F_list,
        "labels": labels_list,
    }

    if y_true is not None:
        base_data["metrics"] = metrics_list

    return base_data
"""Lightweight k-means clustering implementation."""

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


class LiteKMeans:
    """
    Lightweight k-means clustering implementation.

    This class provides a simple interface matching the MATLAB litekmeans
    function, with support for multiple initializations and replicates.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    max_iter : int, default=100
        Maximum number of iterations
    n_init : int, default=1
        Number of times to run k-means with different initializations
    tol : float, default=1e-4
        Convergence tolerance
    random_state : int or None, default=None
        Random seed for reproducibility

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each sample
    cluster_centers_ : np.ndarray
        Coordinates of cluster centers
    inertia_ : float
        Sum of squared distances to closest cluster center
    n_iter_ : int
        Number of iterations run

    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        n_init: int = 1,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "LiteKMeans":
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)

        Returns
        -------
        self : LiteKMeans
            Fitted estimator
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=self.random_state,
            algorithm="lloyd",
        )

        kmeans.fit(X)

        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.n_iter_ = kmeans.n_iter_

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each sample
        """
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray
            New data of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each sample
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")

        distances = np.sum((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2, axis=2)

        labels = np.argmin(distances, axis=1)

        return labels


def litekmeans(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    n_init: int = 1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform k-means clustering (functional interface).

    This function provides a simple functional interface matching the
    MATLAB litekmeans function.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    max_iter : int, default=100
        Maximum number of iterations
    n_init : int, default=1
        Number of times to run k-means with different initializations
    random_state : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (n_samples,)
    centers : np.ndarray
        Cluster centers (n_clusters, n_features)
    inertia : float
        Sum of squared distances to closest cluster center

    """
    kmeans = LiteKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )

    kmeans.fit(X)

    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_
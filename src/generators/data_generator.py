from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs


class ClusterDataGenerator:
    """
    Generate synthetic clustered data with controllable clusterability.

    Supports multiple generation modes for clustering benchmarks.

    Parameters
    ----------
    random_state : int or None, default=None
        Random seed for reproducibility
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_clusters: int = 5,
        mode: Literal['blobs', 'simple_blobs', 'anisotropic', 'varied'] = 'blobs',
        clusterability: float = 0.7,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clustered data with specified clusterability.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate
        n_features : int, default=10
            Number of features (dimensions)
        n_clusters : int, default=5
            Number of clusters
        mode : str, default='blobs'
            Generation mode:
            - 'blobs': Gaussian clusters generated with the custom intermixing procedure
            - 'simple_blobs': Isotropic Gaussian blobs (good for general testing)
            - 'anisotropic': Anisotropic Gaussian blobs (elongated clusters)
            - 'varied': Blobs with varied variances (unequal cluster sizes)
        clusterability : float, default=0.7
            Clusterability level (0.0 to 1.0):
            - 1.0: Perfect clusters (high separation)
            - 0.7: Good clusters (moderate separation)
            - 0.5: Weak clusters (low separation)
            - 0.3: Very weak clusters (barely separable)
            - 0.0: Random data (no cluster structure)
        **kwargs : dict
            Additional parameters passed to generation functions

        Returns
        -------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)
        y : np.ndarray
            True cluster labels of shape (n_samples,)

        Notes
        -----
        Clusterability is controlled by:
        - For 'blobs': squeeze parameter alpha (lower = more clusterable)
        - For 'simple_blobs': cluster_std (lower = more clusterable)
        """
        if not 0.0 <= clusterability <= 1.0:
            raise ValueError("clusterability must be between 0.0 and 1.0")
        
        if mode == 'blobs':
            return self._generate_blobs(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'simple_blobs':
            return self._generate_simple_blobs(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'anisotropic':
            return self._generate_anisotropic(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'varied':
            return self._generate_varied(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _generate_simple_blobs(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple isotropic Gaussian blobs."""
        cluster_std = 0.5 + (1.0 - clusterability) * 4.5
        
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=self.random_state,
            **kwargs
        )
        
        return X, y
    
    def _generate_anisotropic(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate anisotropic (elongated) Gaussian blobs."""
        cluster_std = 0.5 + (1.0 - clusterability) * 4.5
        
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=self.random_state,
            **kwargs
        )
        
        transformation = self.rng.randn(n_features, n_features)
        X = np.dot(X, transformation)
        
        return X, y
    
    def _generate_varied(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate blobs with varied cluster sizes and variances."""
        cluster_sizes = self.rng.randint(
            n_samples // (n_clusters * 2),
            n_samples // n_clusters * 2,
            size=n_clusters
        )
        cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n_samples).astype(int)
        cluster_sizes[-1] = n_samples - cluster_sizes[:-1].sum()
        
        base_std = 0.5 + (1.0 - clusterability) * 4.5
        cluster_stds = base_std * (0.5 + self.rng.rand(n_clusters))
        
        X_list = []
        y_list = []
        
        for i in range(n_clusters):
            X_cluster, _ = make_blobs(
                n_samples=cluster_sizes[i],
                n_features=n_features,
                centers=1,
                cluster_std=cluster_stds[i],
                random_state=self.random_state + i if self.random_state else None,
                **kwargs
            )
            X_list.append(X_cluster)
            y_list.append(np.full(cluster_sizes[i], i))
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        indices = self.rng.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y

    def _clusterability_to_alpha(self, clusterability: float) -> float:
        """Map clusterability in [0, 1] to squeeze parameter alpha in (0, 1)."""
        return 0.1 + (1.0 - clusterability) * 0.8

    def _generate_blobs(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Gaussian clusters using the custom intermixing procedure."""
        min_cluster_size = kwargs.pop('min_cluster_size', None)
        alpha = kwargs.pop('alpha', None)

        if kwargs:
            unexpected = ', '.join(sorted(kwargs.keys()))
            raise TypeError(f'Unexpected keyword arguments for blobs mode: {unexpected}')

        if alpha is None:
            alpha = self._clusterability_to_alpha(clusterability)
        if not 0.0 < alpha < 1.0:
            raise ValueError('alpha must be in the open interval (0, 1)')

        if min_cluster_size is None:
            min_cluster_size = max(1, n_samples // (2 * n_clusters))
        if min_cluster_size <= 0:
            raise ValueError('min_cluster_size must be positive')
        if n_clusters * min_cluster_size > n_samples:
            raise ValueError('n_clusters * min_cluster_size must not exceed n_samples')

        cluster_sizes = np.full(n_clusters, min_cluster_size, dtype=int)
        remaining = n_samples - n_clusters * min_cluster_size
        if remaining > 0:
            extra = self.rng.multinomial(remaining, np.full(n_clusters, 1.0 / n_clusters))
            cluster_sizes += extra

        center_low = alpha - 1.0
        center_high = 1.0 - alpha
        centers = self.rng.uniform(center_low, center_high, size=(n_clusters, n_features))
        stds = self.rng.uniform(0.05, 0.10, size=(n_clusters, n_features))

        X_parts = []
        y_parts = []
        for cluster_idx in range(n_clusters):
            noise = self.rng.normal(
                loc=0.0,
                scale=stds[cluster_idx],
                size=(cluster_sizes[cluster_idx], n_features),
            )
            X_cluster = centers[cluster_idx] + noise
            X_parts.append(X_cluster)
            y_parts.append(np.full(cluster_sizes[cluster_idx], cluster_idx, dtype=int))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        return X, y
    
    

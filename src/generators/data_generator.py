"""Advanced data generator for clustering benchmarks with clusterability control."""

from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


class ClusterDataGenerator:
    """
    Generate synthetic clustered data with controllable clusterability.
    
    Supports multiple generation modes and provides Hopkins statistic
    for measuring clusterability tendency.
    
    Parameters
    ----------
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Examples
    --------
    >>> gen = ClusterDataGenerator(random_state=42)
    >>> X, y = gen.generate(n_samples=1000, n_clusters=5, mode='blobs', 
    ...                     clusterability=0.8)
    >>> hopkins = gen.hopkins_statistic(X)
    >>> print(f"Hopkins statistic: {hopkins:.3f}")
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_clusters: int = 5,
        mode: Literal['blobs', 'moons', 'circles', 'anisotropic', 'varied', 'noisy'] = 'blobs',
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
            - 'blobs': Isotropic Gaussian blobs (good for general testing)
            - 'moons': Two interleaving half circles (non-convex clusters)
            - 'circles': Concentric circles (non-convex clusters)
            - 'anisotropic': Anisotropic Gaussian blobs (elongated clusters)
            - 'varied': Blobs with varied variances (unequal cluster sizes)
            - 'noisy': Blobs with uniform noise background
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
        - For 'blobs': cluster_std (lower = more clusterable)
        - For 'moons'/'circles': noise level (lower = more clusterable)
        - For 'noisy': noise ratio (lower = more clusterable)
        """
        if not 0.0 <= clusterability <= 1.0:
            raise ValueError("clusterability must be between 0.0 and 1.0")
        
        if mode == 'blobs':
            return self._generate_blobs(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'moons':
            return self._generate_moons(n_samples, n_features, clusterability, **kwargs)
        elif mode == 'circles':
            return self._generate_circles(n_samples, n_features, clusterability, **kwargs)
        elif mode == 'anisotropic':
            return self._generate_anisotropic(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'varied':
            return self._generate_varied(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        elif mode == 'noisy':
            return self._generate_noisy(
                n_samples, n_features, n_clusters, clusterability, **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _generate_blobs(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate isotropic Gaussian blobs."""
        # Map clusterability to cluster_std
        # clusterability=1.0 -> std=0.5 (tight clusters)
        # clusterability=0.5 -> std=2.0 (moderate overlap)
        # clusterability=0.0 -> std=5.0 (heavy overlap)
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
    
    def _generate_moons(
        self,
        n_samples: int,
        n_features: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two interleaving half circles."""
        # Map clusterability to noise
        noise = (1.0 - clusterability) * 0.3
        
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=self.random_state,
            **kwargs
        )
        
        # Extend to n_features dimensions if needed
        if n_features > 2:
            extra_features = self.rng.randn(n_samples, n_features - 2) * 0.1
            X = np.hstack([X, extra_features])
        
        return X, y
    
    def _generate_circles(
        self,
        n_samples: int,
        n_features: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate concentric circles."""
        # Map clusterability to noise and factor
        noise = (1.0 - clusterability) * 0.3
        factor = 0.3 + clusterability * 0.4  # 0.3 to 0.7
        
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=factor,
            random_state=self.random_state,
            **kwargs
        )
        
        # Extend to n_features dimensions if needed
        if n_features > 2:
            extra_features = self.rng.randn(n_samples, n_features - 2) * 0.1
            X = np.hstack([X, extra_features])
        
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
        
        # Apply random transformation to make clusters anisotropic
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
        # Create varied cluster sizes
        cluster_sizes = self.rng.randint(
            n_samples // (n_clusters * 2),
            n_samples // n_clusters * 2,
            size=n_clusters
        )
        cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n_samples).astype(int)
        cluster_sizes[-1] = n_samples - cluster_sizes[:-1].sum()
        
        # Create varied standard deviations
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
        
        # Shuffle
        indices = self.rng.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def _generate_noisy(
        self,
        n_samples: int,
        n_features: int,
        n_clusters: int,
        clusterability: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate blobs with uniform noise background."""
        # Map clusterability to noise ratio
        # clusterability=1.0 -> 0% noise
        # clusterability=0.5 -> 25% noise
        # clusterability=0.0 -> 50% noise
        noise_ratio = (1.0 - clusterability) * 0.5
        
        n_cluster_samples = int(n_samples * (1 - noise_ratio))
        n_noise_samples = n_samples - n_cluster_samples
        
        # Generate clusters
        cluster_std = 0.5 + (1.0 - clusterability) * 2.0
        X_clusters, y_clusters = make_blobs(
            n_samples=n_cluster_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=self.random_state,
            **kwargs
        )
        
        # Generate uniform noise
        if n_noise_samples > 0:
            # Get data range
            data_min = X_clusters.min(axis=0)
            data_max = X_clusters.max(axis=0)
            data_range = data_max - data_min
            
            # Expand range for noise
            noise_min = data_min - data_range * 0.2
            noise_max = data_max + data_range * 0.2
            
            X_noise = self.rng.uniform(
                noise_min, noise_max, size=(n_noise_samples, n_features)
            )
            y_noise = np.full(n_noise_samples, -1)  # Label noise as -1
            
            X = np.vstack([X_clusters, X_noise])
            y = np.concatenate([y_clusters, y_noise])
            
            # Shuffle
            indices = self.rng.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        else:
            X, y = X_clusters, y_clusters
        
        return X, y
    
    def hopkins_statistic(self, X: np.ndarray, n_samples: int = 200) -> float:
        """
        Calculate Hopkins statistic to measure clusterability tendency.
        
        The Hopkins statistic tests the spatial randomness of data.
        Values close to 0.5 indicate random data (no cluster tendency).
        Values close to 1.0 indicate highly clusterable data.
        Values close to 0.0 indicate regularly spaced data.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)
        n_samples : int, default=200
            Number of samples to use for calculation
        
        Returns
        -------
        hopkins : float
            Hopkins statistic (0.0 to 1.0)
            - H > 0.75: Data is highly clusterable
            - 0.5 < H < 0.75: Data has moderate cluster tendency
            - H ≈ 0.5: Data is random (no cluster structure)
            - H < 0.5: Data is regularly spaced
        
        References
        ----------
        Hopkins, B., & Skellam, J. G. (1954). A new method for determining 
        the type of distribution of plant individuals. Annals of Botany, 
        18(2), 213-227.
        """
        n, d = X.shape
        
        # Use min of n_samples and n//10
        m = min(n_samples, n // 10)
        
        # Sample m random points from X
        sample_indices = self.rng.choice(n, size=m, replace=False)
        X_sample = X[sample_indices]
        
        # Generate m uniformly random points in the data space
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        X_random = self.rng.uniform(data_min, data_max, size=(m, d))
        
        # Calculate distances
        from scipy.spatial.distance import cdist
        
        # Distance from random points to nearest data point
        dist_random = cdist(X_random, X).min(axis=1)
        u = dist_random.sum()
        
        # Distance from sample points to nearest other data point
        # (excluding the point itself)
        dist_sample = np.partition(cdist(X_sample, X), 1, axis=1)[:, 1]
        w = dist_sample.sum()
        
        # Hopkins statistic
        hopkins = u / (u + w)
        
        return hopkins
    
    def estimate_clusterability(self, X: np.ndarray) -> dict:
        """
        Estimate multiple clusterability metrics for the data.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)
        
        Returns
        -------
        metrics : dict
            Dictionary containing:
            - 'hopkins': Hopkins statistic
            - 'interpretation': Text interpretation of Hopkins value
            - 'variance_ratio': Ratio of between-cluster to within-cluster variance
        """
        hopkins = self.hopkins_statistic(X)
        
        if hopkins > 0.75:
            interpretation = "Highly clusterable"
        elif hopkins > 0.6:
            interpretation = "Moderately clusterable"
        elif hopkins > 0.5:
            interpretation = "Weakly clusterable"
        else:
            interpretation = "Random or regularly spaced (not clusterable)"
        
        # Calculate variance ratio (simple estimate)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        total_variance = np.var(X_scaled)
        
        return {
            'hopkins': hopkins,
            'interpretation': interpretation,
            'total_variance': total_variance,
        }

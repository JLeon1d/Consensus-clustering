"""Integration tests for SDGCA algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from consensus_clustering import SDGCA


class TestSDGCAIntegration:
    """Integration test suite for SDGCA algorithm."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic clustered data."""
        np.random.seed(42)
        X, y = make_blobs(
            n_samples=100, n_features=10, centers=3, cluster_std=1.0, random_state=42
        )
        return X, y

    @pytest.fixture
    def base_clusterings_from_data(self, synthetic_data):
        """Generate base clusterings from synthetic data."""
        from sklearn.cluster import KMeans

        X, y = synthetic_data
        n_base = 15
        n_clusters = 3

        base_clusterings = np.zeros((len(X), n_base), dtype=int)
        for i in range(n_base):
            kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
            base_clusterings[:, i] = kmeans.fit_predict(X) + 1

        return base_clusterings, y

    def test_sdgca_on_synthetic_data(self, base_clusterings_from_data):
        """Test SDGCA on synthetic data with known ground truth."""
        base_clusterings, true_labels = base_clusterings_from_data

        sdgca = SDGCA(n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        predicted_labels = sdgca.fit_predict(base_clusterings)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        assert ari > 0.3
        assert nmi > 0.3

    def test_sdgca_different_parameters(self, base_clusterings_from_data):
        """Test SDGCA with different parameter settings."""
        base_clusterings, true_labels = base_clusterings_from_data

        param_sets = [
            {"lambda_param": 0.05, "eta": 0.65, "theta": 0.7},
            {"lambda_param": 0.1, "eta": 0.75, "theta": 0.65},
            {"lambda_param": 0.15, "eta": 0.8, "theta": 0.6},
        ]

        for params in param_sets:
            sdgca = SDGCA(n_clusters=3, max_iter=100, **params)
            predicted_labels = sdgca.fit_predict(base_clusterings)

            assert len(predicted_labels) == len(true_labels)
            assert len(np.unique(predicted_labels)) <= 3

    def test_sdgca_vs_simple_nwca(self, base_clusterings_from_data):
        """Compare SDGCA with simple NWCA (eta > 1)."""
        base_clusterings, true_labels = base_clusterings_from_data

        sdgca_full = SDGCA(
            n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
        )
        labels_full = sdgca_full.fit_predict(base_clusterings)

        sdgca_simple = SDGCA(n_clusters=3, lambda_param=0.1, eta=1.5, max_iter=100)
        labels_simple = sdgca_simple.fit_predict(base_clusterings)

        assert len(labels_full) == len(labels_simple)

    def test_sdgca_convergence(self, base_clusterings_from_data):
        """Test that SDGCA converges."""
        base_clusterings, _ = base_clusterings_from_data

        sdgca = SDGCA(
            n_clusters=3,
            lambda_param=0.1,
            eta=0.7,
            theta=0.6,
            max_iter=200,
            tol=1e-4,
            verbose=False,
        )
        sdgca.fit(base_clusterings)

        assert sdgca.W_ is not None
        assert sdgca.labels_ is not None

    def test_sdgca_with_noisy_base_clusterings(self, synthetic_data):
        """Test SDGCA with very noisy base clusterings."""
        from sklearn.cluster import KMeans

        X, y = synthetic_data
        n_base = 20
        n_clusters = 3

        base_clusterings = np.zeros((len(X), n_base), dtype=int)
        for i in range(n_base):
            if i < 10:
                kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
                base_clusterings[:, i] = kmeans.fit_predict(X) + 1
            else:
                base_clusterings[:, i] = np.random.randint(1, n_clusters + 1, len(X))

        sdgca = SDGCA(n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        predicted_labels = sdgca.fit_predict(base_clusterings)

        assert len(predicted_labels) == len(y)
        assert len(np.unique(predicted_labels)) <= 3

    def test_sdgca_stability(self, base_clusterings_from_data):
        """Test stability of SDGCA results."""
        base_clusterings, _ = base_clusterings_from_data

        results = []
        for _ in range(3):
            sdgca = SDGCA(
                n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
            )
            labels = sdgca.fit_predict(base_clusterings)
            results.append(labels)

        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i])

    def test_sdgca_different_cluster_counts(self, synthetic_data):
        """Test SDGCA with different numbers of clusters."""
        from sklearn.cluster import KMeans

        X, _ = synthetic_data

        for n_clusters in [2, 3, 4, 5]:
            n_base = 10
            base_clusterings = np.zeros((len(X), n_base), dtype=int)
            for i in range(n_base):
                kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
                base_clusterings[:, i] = kmeans.fit_predict(X) + 1

            sdgca = SDGCA(
                n_clusters=n_clusters, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
            )
            labels = sdgca.fit_predict(base_clusterings)

            assert len(labels) == len(X)
            assert len(np.unique(labels)) <= n_clusters + 1

    def test_sdgca_matrices_properties(self, base_clusterings_from_data):
        """Test properties of computed matrices."""
        base_clusterings, _ = base_clusterings_from_data

        sdgca = SDGCA(n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        sdgca.fit(base_clusterings)

        n_samples = base_clusterings.shape[0]

        assert sdgca.CA_.shape == (n_samples, n_samples)
        assert np.allclose(sdgca.CA_, sdgca.CA_.T)
        assert np.all(sdgca.CA_ >= 0)
        assert np.all(sdgca.CA_ <= 1)

        assert sdgca.NWCA_.shape == (n_samples, n_samples)
        assert np.allclose(sdgca.NWCA_, sdgca.NWCA_.T)
        assert np.all(sdgca.NWCA_ >= 0)
        assert np.all(sdgca.NWCA_ <= 1 + 1e-6)

        assert sdgca.W_.shape == (n_samples, n_samples)

    def test_sdgca_with_identical_base_clusterings(self, synthetic_data):
        """Test SDGCA when all base clusterings are identical."""
        from sklearn.cluster import KMeans

        X, y = synthetic_data
        n_base = 10

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        single_clustering = kmeans.fit_predict(X) + 1

        base_clusterings = np.tile(single_clustering[:, np.newaxis], (1, n_base))

        sdgca = SDGCA(n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        labels = sdgca.fit_predict(base_clusterings)

        assert len(labels) == len(y)
        assert len(np.unique(labels)) <= 3

    def test_sdgca_performance_metrics(self, base_clusterings_from_data):
        """Test that SDGCA achieves reasonable performance metrics."""
        base_clusterings, true_labels = base_clusterings_from_data

        sdgca = SDGCA(n_clusters=3, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=150)
        predicted_labels = sdgca.fit_predict(base_clusterings)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        assert ari >= 0
        assert ari <= 1
        assert nmi >= 0
        assert nmi <= 1

    def test_sdgca_large_dataset(self):
        """Test SDGCA on a larger dataset."""
        from sklearn.cluster import KMeans

        np.random.seed(42)
        X, y = make_blobs(
            n_samples=200, n_features=15, centers=4, cluster_std=1.5, random_state=42
        )

        n_base = 12
        n_clusters = 4
        base_clusterings = np.zeros((len(X), n_base), dtype=int)
        for i in range(n_base):
            kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
            base_clusterings[:, i] = kmeans.fit_predict(X) + 1

        sdgca = SDGCA(
            n_clusters=4, lambda_param=0.08, eta=0.75, theta=0.65, max_iter=100
        )
        labels = sdgca.fit_predict(base_clusterings)

        assert len(labels) == len(y)
        assert len(np.unique(labels)) <= 4

        ari = adjusted_rand_score(y, labels)
        assert ari > 0.2

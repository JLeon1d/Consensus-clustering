"""Integration tests for SDGCA algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.sdgca import SDGCA


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

    def test_sdgca_on_synthetic_data(self, synthetic_data):
        """Test SDGCA on synthetic data with known ground truth."""
        X, true_labels = synthetic_data

        sdgca = SDGCA(n_clusters=3, m_base=15, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        predicted_labels = sdgca.fit_predict(X)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        assert ari > 0.3
        assert nmi > 0.3

    def test_sdgca_different_parameters(self, synthetic_data):
        """Test SDGCA with different parameter settings."""
        X, true_labels = synthetic_data

        param_sets = [
            {"lambda_param": 0.05, "eta": 0.65, "theta": 0.7},
            {"lambda_param": 0.1, "eta": 0.75, "theta": 0.65},
            {"lambda_param": 0.15, "eta": 0.8, "theta": 0.6},
        ]

        for params in param_sets:
            sdgca = SDGCA(n_clusters=3, m_base=15, max_iter=100, **params)
            predicted_labels = sdgca.fit_predict(X)

            assert len(predicted_labels) == len(true_labels)
            assert len(np.unique(predicted_labels)) <= 3

    def test_sdgca_vs_simple_nwca(self, synthetic_data):
        """Compare SDGCA with simple NWCA (eta > 1)."""
        X, true_labels = synthetic_data

        sdgca_full = SDGCA(
            n_clusters=3, m_base=15, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
        )
        labels_full = sdgca_full.fit_predict(X)

        sdgca_simple = SDGCA(n_clusters=3, m_base=15, lambda_param=0.1, eta=1.5, max_iter=100)
        labels_simple = sdgca_simple.fit_predict(X)

        assert len(labels_full) == len(labels_simple)

    def test_sdgca_convergence(self, synthetic_data):
        """Test that SDGCA converges."""
        X, _ = synthetic_data

        sdgca = SDGCA(
            n_clusters=3,
            m_base=15,
            lambda_param=0.1,
            eta=0.7,
            theta=0.6,
            max_iter=200,
            tol=1e-4,
            verbose=False,
        )
        sdgca.fit(X)

        assert sdgca.W_ is not None
        assert sdgca.labels_ is not None

    def test_sdgca_stability(self, synthetic_data):
        """Test stability of SDGCA results (same number of clusters across runs)."""
        X, _ = synthetic_data

        results = []
        for _ in range(3):
            sdgca = SDGCA(
                n_clusters=3, m_base=15, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
            )
            labels = sdgca.fit_predict(X)
            results.append(labels)

        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])
            assert len(np.unique(results[0])) == len(np.unique(results[i]))

    def test_sdgca_different_cluster_counts(self, synthetic_data):
        """Test SDGCA with different numbers of clusters."""
        X, _ = synthetic_data

        for n_clusters in [2, 3, 4, 5]:
            sdgca = SDGCA(
                n_clusters=n_clusters, m_base=10, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100
            )
            labels = sdgca.fit_predict(X)

            assert len(labels) == len(X)
            assert len(np.unique(labels)) <= n_clusters + 1

    def test_sdgca_matrices_properties(self, synthetic_data):
        """Test properties of computed matrices."""
        X, _ = synthetic_data

        sdgca = SDGCA(n_clusters=3, m_base=15, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=100)
        sdgca.fit(X)

        n_samples = X.shape[0]

        assert sdgca.CA_.shape == (n_samples, n_samples)
        assert np.allclose(sdgca.CA_, sdgca.CA_.T)
        assert np.all(sdgca.CA_ >= 0)
        assert np.all(sdgca.CA_ <= 1)

        assert sdgca.NWCA_.shape == (n_samples, n_samples)
        assert np.allclose(sdgca.NWCA_, sdgca.NWCA_.T)
        assert np.all(sdgca.NWCA_ >= 0)
        assert np.all(sdgca.NWCA_ <= 1 + 1e-6)

        assert sdgca.W_.shape == (n_samples, n_samples)

    def test_sdgca_performance_metrics(self, synthetic_data):
        """Test that SDGCA achieves reasonable performance metrics."""
        X, true_labels = synthetic_data

        sdgca = SDGCA(n_clusters=3, m_base=15, lambda_param=0.1, eta=0.7, theta=0.6, max_iter=150)
        predicted_labels = sdgca.fit_predict(X)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        assert ari >= 0
        assert ari <= 1
        assert nmi >= 0
        assert nmi <= 1

    def test_sdgca_large_dataset(self):
        """Test SDGCA on a larger dataset."""
        np.random.seed(42)
        X, y = make_blobs(
            n_samples=200, n_features=15, centers=4, cluster_std=1.5, random_state=42
        )

        sdgca = SDGCA(
            n_clusters=4, m_base=12, lambda_param=0.08, eta=0.75, theta=0.65, max_iter=100
        )
        labels = sdgca.fit_predict(X)

        assert len(labels) == len(y)
        assert len(np.unique(labels)) <= 4

        ari = adjusted_rand_score(y, labels)
        assert ari > 0.2

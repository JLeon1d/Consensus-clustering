"""Integration tests for ACMK algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from consensus_clustering import ACMK, clustering_measure, generate_base_clusterings


class TestACMKIntegration:
    """Integration tests for full ACMK pipeline."""

    @pytest.mark.integration
    def test_acmk_basic_pipeline(self, small_synthetic_data):
        """Test basic ACMK pipeline with synthetic data."""
        X, y_true = small_synthetic_data
        n_clusters = len(np.unique(y_true))

        # Generate base clusterings
        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=5, random_state=42
        )

        # Run ACMK
        acmk = ACMK(
            n_clusters=n_clusters, m_base=5, lambda_=0.1, k_power=3, max_iter=5, verbose=False
        )

        acmk.fit(X, **base_data)

        # Get predictions
        labels_spectral = acmk.predict(method="spectral")
        labels_kmeans = acmk.predict(method="kmeans")

        # Check outputs
        assert labels_spectral.shape == (X.shape[0],)
        assert labels_kmeans.shape == (X.shape[0],)
        assert len(np.unique(labels_spectral)) <= n_clusters
        assert len(np.unique(labels_kmeans)) <= n_clusters

        # Evaluate
        metrics_spectral = clustering_measure(y_true, labels_spectral)
        metrics_kmeans = clustering_measure(y_true, labels_kmeans)

        # Should have reasonable performance
        assert metrics_spectral["acc"] > 0.3  # At least better than random
        assert metrics_kmeans["acc"] > 0.3

    @pytest.mark.integration
    def test_acmk_fit_predict(self, small_synthetic_data):
        """Test fit_predict method."""
        X, y_true = small_synthetic_data
        n_clusters = len(np.unique(y_true))

        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=5, random_state=42
        )

        acmk = ACMK(n_clusters=n_clusters, m_base=5, lambda_=0.1, max_iter=5)

        labels = acmk.fit_predict(X, **base_data, method="spectral")

        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) <= n_clusters

    @pytest.mark.integration
    def test_acmk_with_different_parameters(self, tiny_data):
        """Test ACMK with different parameter settings."""
        X, y_true = tiny_data
        n_clusters = len(np.unique(y_true))

        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=3, random_state=42
        )

        # Test with different lambda values
        for lambda_val in [0.01, 0.1, 1.0]:
            acmk = ACMK(
                n_clusters=n_clusters,
                m_base=3,
                lambda_=lambda_val,
                k_power=2,
                max_iter=3,
                verbose=False,
            )

            acmk.fit(X, **base_data)
            labels = acmk.predict()

            assert labels.shape == (X.shape[0],)
            assert acmk.alpha_ is not None
            assert len(acmk.alpha_) == 3
            assert np.allclose(acmk.alpha_.sum(), 1.0)  # Weights sum to 1

    @pytest.mark.integration
    def test_base_clustering_generation(self, small_synthetic_data):
        """Test base clustering generation."""
        X, y_true = small_synthetic_data
        n_clusters = len(np.unique(y_true))

        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=10, random_state=42, y_true=y_true
        )

        # Check structure
        assert "W" in base_data
        assert "G" in base_data
        assert "F" in base_data
        assert "labels" in base_data
        assert "metrics" in base_data

        # Check dimensions
        assert base_data["W"].shape == (X.shape[0], X.shape[0])
        assert len(base_data["G"]) == 10
        assert len(base_data["F"]) == 10
        assert len(base_data["labels"]) == 10
        assert len(base_data["metrics"]) == 10

        # Check W is symmetric and normalized
        assert np.allclose(base_data["W"], base_data["W"].T)
        assert base_data["W"].min() >= 0
        assert base_data["W"].max() <= 1

        # Check metrics
        for metrics in base_data["metrics"]:
            assert "acc" in metrics
            assert "nmi" in metrics
            assert "purity" in metrics
            assert 0 <= metrics["acc"] <= 1
            assert 0 <= metrics["nmi"] <= 1
            assert 0 <= metrics["purity"] <= 1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_acmk_convergence(self, synthetic_data):
        """Test that ACMK converges on larger dataset."""
        X, y_true = synthetic_data
        n_clusters = len(np.unique(y_true))

        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=10, random_state=42
        )

        acmk = ACMK(
            n_clusters=n_clusters,
            m_base=10,
            lambda_=0.1,
            k_power=3,
            max_iter=20,
            verbose=False,
        )

        acmk.fit(X, **base_data)
        labels = acmk.predict(method="spectral")

        metrics = clustering_measure(y_true, labels)

        # Should achieve reasonable performance on well-separated data
        assert metrics["acc"] > 0.5
        assert metrics["nmi"] > 0.4

    @pytest.mark.integration
    def test_acmk_attributes(self, tiny_data):
        """Test that ACMK sets all expected attributes."""
        X, y_true = tiny_data
        n_clusters = len(np.unique(y_true))

        base_data = generate_base_clusterings(
            X, n_clusters=n_clusters, m_base=3, random_state=42
        )

        acmk = ACMK(n_clusters=n_clusters, m_base=3, lambda_=0.1, max_iter=3)
        acmk.fit(X, **base_data)

        # Check all attributes are set
        assert acmk.W_ is not None
        assert acmk.A_ is not None
        assert acmk.G_ is not None
        assert acmk.F_ is not None
        assert acmk.alpha_ is not None
        assert acmk.labels_spectral_ is not None
        assert acmk.labels_kmeans_ is not None

        # Check dimensions
        assert acmk.W_.shape == (X.shape[0], X.shape[0])
        assert acmk.A_.shape == (X.shape[0], X.shape[0])
        assert len(acmk.G_) == 3
        assert len(acmk.F_) == 3
        assert acmk.alpha_.shape == (3,)
        assert acmk.labels_spectral_.shape == (X.shape[0],)
        assert acmk.labels_kmeans_.shape == (X.shape[0],)
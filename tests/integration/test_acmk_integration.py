"""Integration tests for ACMK algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.acmk import ACMK
from src.metrics.clustering_measure import clustering_measure


class TestACMKIntegration:
    """Integration tests for full ACMK pipeline."""

    @pytest.mark.integration
    def test_acmk_basic_pipeline(self, small_synthetic_data):
        """Test basic ACMK pipeline with synthetic data."""
        X, y_true = small_synthetic_data
        n_clusters = len(np.unique(y_true))

        acmk = ACMK(
            n_clusters=n_clusters, m_base=5, lambda_=0.1, k_power=3, max_iter=5, verbose=False
        )
        acmk.fit(X)

        labels_spectral = acmk.predict(method="spectral")
        labels_kmeans = acmk.predict(method="kmeans")

        assert labels_spectral.shape == (X.shape[0],)
        assert labels_kmeans.shape == (X.shape[0],)
        assert len(np.unique(labels_spectral)) <= n_clusters
        assert len(np.unique(labels_kmeans)) <= n_clusters

        metrics_spectral = clustering_measure(y_true, labels_spectral)
        metrics_kmeans = clustering_measure(y_true, labels_kmeans)

        assert 0 <= metrics_spectral["acc"] <= 1
        assert 0 <= metrics_kmeans["acc"] <= 1

    @pytest.mark.integration
    def test_acmk_fit_predict(self, small_synthetic_data):
        """Test fit_predict method."""
        X, y_true = small_synthetic_data
        n_clusters = len(np.unique(y_true))

        acmk = ACMK(n_clusters=n_clusters, m_base=5, lambda_=0.1, max_iter=5)
        labels = acmk.fit_predict(X, method="spectral")

        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) <= n_clusters

    @pytest.mark.integration
    def test_acmk_with_different_parameters(self, tiny_data):
        """Test ACMK with different parameter settings."""
        X, y_true = tiny_data
        n_clusters = len(np.unique(y_true))

        for lambda_val in [0.01, 0.1, 1.0]:
            acmk = ACMK(
                n_clusters=n_clusters,
                m_base=3,
                lambda_=lambda_val,
                k_power=2,
                max_iter=3,
                verbose=False,
            )
            acmk.fit(X)
            labels = acmk.predict()

            assert labels.shape == (X.shape[0],)
            assert acmk.alpha_ is not None
            assert len(acmk.alpha_) == 3
            assert np.allclose(acmk.alpha_.sum(), 1.0)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_acmk_convergence(self, synthetic_data):
        """Test that ACMK runs to completion on larger dataset."""
        X, y_true = synthetic_data
        n_clusters = len(np.unique(y_true))

        acmk = ACMK(
            n_clusters=n_clusters,
            m_base=10,
            lambda_=0.1,
            k_power=3,
            max_iter=20,
            verbose=False,
        )
        acmk.fit(X)
        labels = acmk.predict(method="spectral")

        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) <= n_clusters
        assert 0 <= clustering_measure(y_true, labels)["acc"] <= 1

    @pytest.mark.integration
    def test_acmk_attributes(self, tiny_data):
        """Test that ACMK sets all expected attributes."""
        X, y_true = tiny_data
        n_clusters = len(np.unique(y_true))

        acmk = ACMK(n_clusters=n_clusters, m_base=3, lambda_=0.1, max_iter=3)
        acmk.fit(X)

        assert acmk.W_ is not None
        assert acmk.A_ is not None
        assert acmk.G_ is not None
        assert acmk.F_ is not None
        assert acmk.alpha_ is not None
        assert acmk.labels_spectral_ is not None
        assert acmk.labels_kmeans_ is not None

        assert acmk.W_.shape == (X.shape[0], X.shape[0])
        assert acmk.A_.shape == (X.shape[0], X.shape[0])
        assert len(acmk.G_) == 3
        assert len(acmk.F_) == 3
        assert acmk.alpha_.shape == (3,)
        assert acmk.labels_spectral_.shape == (X.shape[0],)
        assert acmk.labels_kmeans_.shape == (X.shape[0],)

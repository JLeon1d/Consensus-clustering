"""Unit tests for SDGCA algorithm."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from src.sdgca import SDGCA


class TestSDGCA:
    """Test suite for SDGCA algorithm."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data for testing."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=50, n_features=5, centers=3, random_state=42)
        return X

    @pytest.fixture
    def consistent_data(self):
        """Create well-separated synthetic data for testing."""
        X, y = make_blobs(n_samples=60, n_features=5, centers=3, cluster_std=0.5, random_state=0)
        return X, y

    def test_fit_simple(self, simple_data):
        """Test fitting SDGCA on simple data."""
        sdgca = SDGCA(n_clusters=3, m_base=5, max_iter=50)
        sdgca.fit(simple_data)

        assert sdgca.labels_ is not None
        assert len(sdgca.labels_) == simple_data.shape[0]
        assert sdgca.W_ is not None
        assert sdgca.CA_ is not None
        assert sdgca.NWCA_ is not None

        assert np.all(sdgca.labels_ >= 1)
        assert np.all(sdgca.labels_ <= 3)

    def test_fit_consistent(self, consistent_data):
        """Test fitting SDGCA on consistent data."""
        X, y = consistent_data
        sdgca = SDGCA(n_clusters=3, m_base=8, max_iter=100)
        sdgca.fit(X)

        assert sdgca.labels_ is not None
        assert len(sdgca.labels_) == len(y)

        unique_labels = np.unique(sdgca.labels_)
        assert len(unique_labels) == 3

    def test_fit_predict(self, simple_data):
        """Test fit_predict method."""
        sdgca = SDGCA(n_clusters=3, m_base=5, max_iter=50)
        labels = sdgca.fit_predict(simple_data)

        assert labels is not None
        assert len(labels) == simple_data.shape[0]
        assert np.all(labels >= 1)
        assert np.all(labels <= 3)

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        sdgca = SDGCA(n_clusters=3)
        with pytest.raises(ValueError, match="Model has not been fitted"):
            sdgca.predict()

    def test_get_all_segs(self):
        """Test _get_all_segs method."""
        sdgca = SDGCA(n_clusters=3)
        np.random.seed(42)
        n_samples = 50
        n_base = 10
        base_clusterings = np.random.randint(1, 4, (n_samples, n_base))

        bcs, base_cls_segs = sdgca._get_all_segs(base_clusterings)

        assert bcs.shape == base_clusterings.shape
        assert base_cls_segs.shape[1] == n_samples

        assert np.all((base_cls_segs == 0) | (base_cls_segs == 1))

        assert np.all(base_cls_segs.sum(axis=0) == n_base)

    def test_compute_neci(self):
        """Test NECI computation."""
        sdgca = SDGCA(n_clusters=3, lambda_param=0.1)
        np.random.seed(42)
        base_clusterings = np.random.randint(1, 4, (50, 10))
        bcs, base_cls_segs = sdgca._get_all_segs(base_clusterings)
        neci = sdgca._compute_neci(bcs, base_cls_segs, sdgca.lambda_param)

        assert len(neci) == base_cls_segs.shape[0]

        assert np.all(neci >= 0)
        assert np.all(neci <= 1)

    def test_compute_nwca(self):
        """Test NWCA computation."""
        sdgca = SDGCA(n_clusters=3, lambda_param=0.1)
        np.random.seed(42)
        n_samples = 50
        n_base = 10
        base_clusterings = np.random.randint(1, 4, (n_samples, n_base))
        bcs, base_cls_segs = sdgca._get_all_segs(base_clusterings)
        neci = sdgca._compute_neci(bcs, base_cls_segs, sdgca.lambda_param)
        nwca = sdgca._compute_nwca(base_cls_segs, neci, n_base)

        assert nwca.shape == (n_samples, n_samples)

        assert np.allclose(nwca, nwca.T)

        assert np.allclose(np.diag(nwca), 1.0)

        assert np.all(nwca >= 0)
        assert np.all(nwca <= 1)

    def test_jaccard_similarity(self):
        """Test Jaccard similarity computation."""
        sdgca = SDGCA(n_clusters=3)

        a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        sim = sdgca._jaccard_similarity(a)

        assert sim.shape == (3, 3)

        assert np.allclose(np.diag(sim), 1.0)

        assert np.allclose(sim, sim.T)

        assert np.all(sim >= 0)
        assert np.all(sim <= 1)

    def test_random_walk_of_cluster(self):
        """Test random walk computation."""
        sdgca = SDGCA(n_clusters=3, k_rw=5, beta_rw=1.0)

        W = np.array([[0, 0.8, 0.2], [0.8, 0, 0.6], [0.2, 0.6, 0]])

        R = sdgca._random_walk_of_cluster(W)

        assert R.shape == W.shape

        assert np.all(R >= 0)
        assert np.all(R <= 1 + 1e-6)

    def test_optimize_sdgca(self):
        """Test SDGCA optimization."""
        sdgca = SDGCA(n_clusters=3, max_iter=50, tol=1e-2)

        n = 10
        L = np.eye(n) * 2 - np.ones((n, n)) * 0.1
        ML = np.random.rand(n, n) * 0.5
        ML = (ML + ML.T) / 2
        CL = np.random.rand(n, n) * 0.3
        CL = (CL + CL.T) / 2

        ML[CL > 0] = 0

        S, D = sdgca._optimize_sdgca(L, ML, CL)

        assert S.shape == (n, n)
        assert D.shape == (n, n)

        assert np.all(S >= -2e-3)
        assert np.all(S <= 1 + 2e-3)
        assert np.all(D >= -2e-3)
        assert np.all(D <= 1 + 2e-3)

    def test_compute_w(self):
        """Test final co-association matrix computation."""
        sdgca = SDGCA(n_clusters=3)

        n = 10
        S = np.random.rand(n, n) * 0.5
        S = (S + S.T) / 2
        D = np.random.rand(n, n) * 0.3
        D = (D + D.T) / 2
        W = np.random.rand(n, n) * 0.8
        W = (W + W.T) / 2

        W_star = sdgca._compute_w(S, D, W)

        assert W_star.shape == (n, n)

        assert np.all(W_star >= -1e-6)
        assert np.all(W_star <= 1 + 1e-6)

    def test_eta_greater_than_one(self, simple_data):
        """Test SDGCA with eta > 1 (uses NWCA directly)."""
        sdgca = SDGCA(n_clusters=3, m_base=5, eta=1.5, max_iter=50)
        sdgca.fit(simple_data)

        assert sdgca.labels_ is not None
        assert sdgca.W_ is not None
        assert sdgca.S_ is None
        assert sdgca.D_ is None

    def test_different_cluster_numbers(self):
        """Test SDGCA with different numbers of clusters."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=40, n_features=5, centers=5, random_state=42)

        for n_clusters in [2, 3, 5]:
            sdgca = SDGCA(n_clusters=n_clusters, m_base=5, max_iter=50)
            labels = sdgca.fit_predict(X)

            assert len(labels) == len(X)
            assert len(np.unique(labels)) <= n_clusters + 1

    def test_verbose_mode(self, simple_data, capsys):
        """Test verbose output."""
        sdgca = SDGCA(n_clusters=3, m_base=5, max_iter=100, verbose=True)
        sdgca.fit(simple_data)

        captured = capsys.readouterr()

    def test_reproducibility(self, simple_data):
        """Test that results are reproducible in structure (same number of clusters)."""
        sdgca1 = SDGCA(n_clusters=3, m_base=5, max_iter=50)
        labels1 = sdgca1.fit_predict(simple_data)

        sdgca2 = SDGCA(n_clusters=3, m_base=5, max_iter=50)
        labels2 = sdgca2.fit_predict(simple_data)

        assert len(labels1) == len(labels2)
        assert len(np.unique(labels1)) == len(np.unique(labels2))

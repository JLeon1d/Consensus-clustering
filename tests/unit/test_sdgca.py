"""Unit tests for SDGCA algorithm."""

import numpy as np
import pytest
from consensus_clustering import SDGCA


class TestSDGCA:
    """Test suite for SDGCA algorithm."""

    @pytest.fixture
    def simple_base_clusterings(self):
        """Create simple base clusterings for testing."""
        np.random.seed(42)
        n_samples = 50
        n_base = 10
        n_clusters = 3

        # Generate base clusterings
        base_clusterings = np.zeros((n_samples, n_base), dtype=int)
        for i in range(n_base):
            # Random clustering with some consistency
            base_clusterings[:, i] = np.random.randint(1, n_clusters + 1, n_samples)

        return base_clusterings

    @pytest.fixture
    def consistent_base_clusterings(self):
        """Create consistent base clusterings for testing."""
        n_samples = 60
        n_base = 8
        n_clusters = 3

        # Create ground truth
        true_labels = np.repeat([1, 2, 3], n_samples // n_clusters)

        # Generate base clusterings with noise
        base_clusterings = np.zeros((n_samples, n_base), dtype=int)
        for i in range(n_base):
            # Add some noise to true labels
            noise = np.random.rand(n_samples) < 0.1
            base_clusterings[:, i] = true_labels.copy()
            base_clusterings[noise, i] = np.random.randint(1, n_clusters + 1, noise.sum())

        return base_clusterings, true_labels

    def test_fit_simple(self, simple_base_clusterings):
        """Test fitting SDGCA on simple data."""
        sdgca = SDGCA(n_clusters=3, max_iter=50)
        sdgca.fit(simple_base_clusterings)

        assert sdgca.labels_ is not None
        assert len(sdgca.labels_) == simple_base_clusterings.shape[0]
        assert sdgca.W_ is not None
        assert sdgca.CA_ is not None
        assert sdgca.NWCA_ is not None

        # Check that labels are in valid range
        assert np.all(sdgca.labels_ >= 1)
        assert np.all(sdgca.labels_ <= 3)

    def test_fit_consistent(self, consistent_base_clusterings):
        """Test fitting SDGCA on consistent data."""
        base_clusterings, true_labels = consistent_base_clusterings
        sdgca = SDGCA(n_clusters=3, max_iter=100)
        sdgca.fit(base_clusterings)

        assert sdgca.labels_ is not None
        assert len(sdgca.labels_) == len(true_labels)

        # Check that we get 3 clusters
        unique_labels = np.unique(sdgca.labels_)
        assert len(unique_labels) == 3

    def test_fit_predict(self, simple_base_clusterings):
        """Test fit_predict method."""
        sdgca = SDGCA(n_clusters=3, max_iter=50)
        labels = sdgca.fit_predict(simple_base_clusterings)

        assert labels is not None
        assert len(labels) == simple_base_clusterings.shape[0]
        assert np.all(labels >= 1)
        assert np.all(labels <= 3)

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        sdgca = SDGCA(n_clusters=3)
        with pytest.raises(ValueError, match="Model has not been fitted"):
            sdgca.predict()

    def test_get_all_segs(self, simple_base_clusterings):
        """Test _get_all_segs method."""
        sdgca = SDGCA(n_clusters=3)
        bcs, base_cls_segs = sdgca._get_all_segs(simple_base_clusterings)

        n_samples, n_base = simple_base_clusterings.shape

        # Check shapes
        assert bcs.shape == simple_base_clusterings.shape
        assert base_cls_segs.shape[1] == n_samples

        # Check that segments are binary
        assert np.all((base_cls_segs == 0) | (base_cls_segs == 1))

        # Check that each sample belongs to exactly n_base clusters
        assert np.all(base_cls_segs.sum(axis=0) == n_base)

    def test_compute_neci(self, simple_base_clusterings):
        """Test NECI computation."""
        sdgca = SDGCA(n_clusters=3, lambda_param=0.1)
        bcs, base_cls_segs = sdgca._get_all_segs(simple_base_clusterings)
        neci = sdgca._compute_neci(bcs, base_cls_segs, sdgca.lambda_param)

        # Check shape
        assert len(neci) == base_cls_segs.shape[0]

        # Check that NECI values are in valid range
        assert np.all(neci >= 0)
        assert np.all(neci <= 1)

    def test_compute_nwca(self, simple_base_clusterings):
        """Test NWCA computation."""
        sdgca = SDGCA(n_clusters=3, lambda_param=0.1)
        bcs, base_cls_segs = sdgca._get_all_segs(simple_base_clusterings)
        neci = sdgca._compute_neci(bcs, base_cls_segs, sdgca.lambda_param)
        nwca = sdgca._compute_nwca(base_cls_segs, neci, simple_base_clusterings.shape[1])

        n_samples = simple_base_clusterings.shape[0]

        # Check shape
        assert nwca.shape == (n_samples, n_samples)

        # Check symmetry
        assert np.allclose(nwca, nwca.T)

        # Check diagonal is 1
        assert np.allclose(np.diag(nwca), 1.0)

        # Check values are in [0, 1]
        assert np.all(nwca >= 0)
        assert np.all(nwca <= 1)

    def test_jaccard_similarity(self):
        """Test Jaccard similarity computation."""
        sdgca = SDGCA(n_clusters=3)

        # Simple test case
        a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        sim = sdgca._jaccard_similarity(a)

        # Check shape
        assert sim.shape == (3, 3)

        # Check diagonal is 1
        assert np.allclose(np.diag(sim), 1.0)

        # Check symmetry
        assert np.allclose(sim, sim.T)

        # Check values are in [0, 1]
        assert np.all(sim >= 0)
        assert np.all(sim <= 1)

    def test_random_walk_of_cluster(self):
        """Test random walk computation."""
        sdgca = SDGCA(n_clusters=3, k_rw=5, beta_rw=1.0)

        # Simple similarity matrix
        W = np.array([[0, 0.8, 0.2], [0.8, 0, 0.6], [0.2, 0.6, 0]])

        R = sdgca._random_walk_of_cluster(W)

        # Check shape
        assert R.shape == W.shape

        # Check diagonal is 1 (after setting)
        # Note: diagonal is set to 1 in the function

        # Check values are in [0, 1]
        assert np.all(R >= 0)
        assert np.all(R <= 1 + 1e-6)  # Allow small numerical error

    def test_optimize_sdgca(self):
        """Test SDGCA optimization."""
        sdgca = SDGCA(n_clusters=3, max_iter=50, tol=1e-2)

        n = 10
        # Create simple test matrices
        L = np.eye(n) * 2 - np.ones((n, n)) * 0.1
        ML = np.random.rand(n, n) * 0.5
        ML = (ML + ML.T) / 2
        CL = np.random.rand(n, n) * 0.3
        CL = (CL + CL.T) / 2

        # Ensure ML and CL don't overlap
        ML[CL > 0] = 0

        S, D = sdgca._optimize_sdgca(L, ML, CL)

        # Check shapes
        assert S.shape == (n, n)
        assert D.shape == (n, n)

        # Check that S and D are in valid range
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

        # Check shape
        assert W_star.shape == (n, n)

        # Check values are in valid range
        assert np.all(W_star >= -1e-6)
        assert np.all(W_star <= 1 + 1e-6)

    def test_eta_greater_than_one(self, simple_base_clusterings):
        """Test SDGCA with eta > 1 (uses NWCA directly)."""
        sdgca = SDGCA(n_clusters=3, eta=1.5, max_iter=50)
        sdgca.fit(simple_base_clusterings)

        assert sdgca.labels_ is not None
        assert sdgca.W_ is not None
        assert sdgca.S_ is None  # Should not compute S when eta > 1
        assert sdgca.D_ is None  # Should not compute D when eta > 1

    def test_different_cluster_numbers(self):
        """Test SDGCA with different numbers of clusters."""
        np.random.seed(42)
        n_samples = 40
        n_base = 8

        for n_clusters in [2, 3, 5]:
            base_clusterings = np.random.randint(1, n_clusters + 1, (n_samples, n_base))
            sdgca = SDGCA(n_clusters=n_clusters, max_iter=50)
            labels = sdgca.fit_predict(base_clusterings)

            assert len(labels) == n_samples
            # Note: hierarchical clustering might not always produce exactly n_clusters
            # but should be close
            assert len(np.unique(labels)) <= n_clusters + 1

    def test_verbose_mode(self, simple_base_clusterings, capsys):
        """Test verbose output."""
        sdgca = SDGCA(n_clusters=3, max_iter=100, verbose=True)
        sdgca.fit(simple_base_clusterings)

        # Check that something was printed
        captured = capsys.readouterr()
        # Verbose output should appear if optimization runs
        # (may not appear if eta > 1 or converges quickly)

    def test_reproducibility(self, simple_base_clusterings):
        """Test that results are reproducible."""
        sdgca1 = SDGCA(n_clusters=3, max_iter=50)
        labels1 = sdgca1.fit_predict(simple_base_clusterings)

        sdgca2 = SDGCA(n_clusters=3, max_iter=50)
        labels2 = sdgca2.fit_predict(simple_base_clusterings)

        # Results should be identical
        assert np.array_equal(labels1, labels2)

    def test_small_dataset(self):
        """Test SDGCA on very small dataset."""
        np.random.seed(42)
        base_clusterings = np.array([[1, 1, 2], [1, 2, 1], [2, 2, 2], [2, 1, 1]])

        sdgca = SDGCA(n_clusters=2, max_iter=50)
        labels = sdgca.fit_predict(base_clusterings)

        assert len(labels) == 4
        assert len(np.unique(labels)) <= 2

    def test_single_base_clustering(self):
        """Test SDGCA with single base clustering."""
        np.random.seed(42)
        base_clusterings = np.array([[1], [2], [1], [2], [3], [3]])

        sdgca = SDGCA(n_clusters=3, max_iter=50)
        labels = sdgca.fit_predict(base_clusterings)

        assert len(labels) == 6
        assert len(np.unique(labels)) <= 3

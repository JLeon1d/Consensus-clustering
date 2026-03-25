"""Unit tests for clustering metrics."""

import numpy as np
import pytest

from consensus_clustering.metrics.clustering_measure import (
    accuracy,
    clustering_measure,
    clustering_purity,
    normalized_mutual_info,
)
from consensus_clustering.metrics.hungarian import best_map, hungarian


class TestHungarian:
    """Tests for Hungarian algorithm."""

    def test_hungarian_simple(self):
        """Test Hungarian algorithm on simple cost matrix."""
        cost = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        assignment, total_cost = hungarian(cost)

        # Check that we get a valid assignment
        assert len(assignment) == 3
        assert total_cost > 0

    def test_best_map_perfect_match(self):
        """Test best_map with perfectly matching labels."""
        L1 = np.array([1, 1, 2, 2, 3, 3])
        L2 = np.array([1, 1, 2, 2, 3, 3])

        new_L2 = best_map(L1, L2)
        assert np.array_equal(new_L2, L1)

    def test_best_map_permuted(self):
        """Test best_map with permuted labels."""
        L1 = np.array([1, 1, 2, 2, 3, 3])
        L2 = np.array([2, 2, 3, 3, 1, 1])

        new_L2 = best_map(L1, L2)
        # After optimal permutation, should match better
        accuracy_before = np.mean(L1 == L2)
        accuracy_after = np.mean(L1 == new_L2)
        assert accuracy_after >= accuracy_before


class TestClusteringMetrics:
    """Tests for clustering evaluation metrics."""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect clustering."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        acc = accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_accuracy_permuted(self):
        """Test accuracy with permuted labels."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 0, 0, 2, 2])

        acc = accuracy(y_true, y_pred)
        # Should still be 1.0 after optimal matching
        assert acc == 1.0

    def test_nmi_perfect(self):
        """Test NMI with perfect clustering."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        nmi = normalized_mutual_info(y_true, y_pred)
        assert nmi == pytest.approx(1.0, abs=1e-6)

    def test_nmi_random(self):
        """Test NMI with random clustering."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.random.randint(0, 3, size=9)

        nmi = normalized_mutual_info(y_true, y_pred)
        # NMI should be between 0 and 1
        assert 0 <= nmi <= 1

    def test_purity_perfect(self):
        """Test purity with perfect clustering."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        purity = clustering_purity(y_true, y_pred)
        assert purity == 1.0

    def test_purity_imperfect(self):
        """Test purity with imperfect clustering."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        purity = clustering_purity(y_true, y_pred)
        # 2 correct in cluster 0, 3 correct in cluster 1 = 5/6
        assert purity == pytest.approx(5 / 6, abs=1e-6)

    def test_clustering_measure(self):
        """Test clustering_measure returns all metrics."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 0, 0, 2, 2])

        metrics = clustering_measure(y_true, y_pred)

        assert "acc" in metrics
        assert "nmi" in metrics
        assert "purity" in metrics

        # All metrics should be between 0 and 1
        assert 0 <= metrics["acc"] <= 1
        assert 0 <= metrics["nmi"] <= 1
        assert 0 <= metrics["purity"] <= 1

    def test_clustering_measure_with_fixture(self, tiny_data):
        """Test clustering_measure with fixture data."""
        X, y_true = tiny_data

        # Create some clustering (just use true labels for now)
        y_pred = y_true.copy()

        metrics = clustering_measure(y_true, y_pred)

        # Should be perfect
        assert metrics["acc"] == 1.0
        assert metrics["nmi"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["purity"] == 1.0
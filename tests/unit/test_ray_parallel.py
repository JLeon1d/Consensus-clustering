"""Tests for Ray parallel processing functionality."""

import numpy as np
import pytest

from src.clustering.base_generation import generate_base_clusterings
from src.utils.ray_utils import (
    get_ray_status,
    init_ray_if_needed,
    is_ray_available,
    shutdown_ray_if_initialized,
)


class TestRayUtils:
    def test_is_ray_available(self):
        result = is_ray_available()
        assert isinstance(result, bool)

    def test_get_ray_status(self):
        status = get_ray_status()
        assert isinstance(status, dict)
        assert 'available' in status
        assert 'initialized' in status
        assert 'num_cpus' in status
        assert isinstance(status['available'], bool)
        assert isinstance(status['initialized'], bool)

    def test_init_ray_if_needed_disabled(self):
        result = init_ray_if_needed(use_ray=False)
        assert result is False

    @pytest.mark.skipif(not is_ray_available(), reason="Ray not installed")
    def test_init_and_shutdown_ray(self):
        shutdown_ray_if_initialized()

        result = init_ray_if_needed(use_ray=True)
        assert result is True

        status = get_ray_status()
        assert status['initialized'] is True

        shutdown_ray_if_initialized()

        status = get_ray_status()
        assert status['initialized'] is False


@pytest.mark.skipif(not is_ray_available(), reason="Ray not installed")
class TestParallelBaseClustering:
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.n_clusters = 3
        self.m_base = 5

    def teardown_method(self):
        shutdown_ray_if_initialized()

    def test_parallel_base_generation_basic(self):
        result = generate_base_clusterings(
            self.X,
            n_clusters=self.n_clusters,
            m_base=self.m_base,
            random_state=42,
            use_ray=True,
        )

        assert 'W' in result
        assert 'G' in result
        assert 'F' in result
        assert 'labels' in result

        assert result['W'].shape == (100, 100)
        assert len(result['G']) == self.m_base
        assert len(result['F']) == self.m_base
        assert len(result['labels']) == self.m_base

        for i in range(self.m_base):
            assert result['G'][i].shape == (100, self.n_clusters)
            assert result['F'][i].shape == (self.n_clusters, 10)
            assert result['labels'][i].shape == (100,)
            assert np.all((result['labels'][i] >= 0) & (result['labels'][i] < self.n_clusters))

    def test_parallel_vs_sequential_consistency(self):
        result_seq = generate_base_clusterings(
            self.X,
            n_clusters=self.n_clusters,
            m_base=self.m_base,
            random_state=42,
            use_ray=False,
        )

        result_par = generate_base_clusterings(
            self.X,
            n_clusters=self.n_clusters,
            m_base=self.m_base,
            random_state=42,
            use_ray=True,
        )

        assert result_seq['W'].shape == result_par['W'].shape
        assert len(result_seq['G']) == len(result_par['G'])
        assert len(result_seq['F']) == len(result_par['F'])
        assert len(result_seq['labels']) == len(result_par['labels'])

        np.testing.assert_allclose(result_seq['W'], result_par['W'], rtol=1e-5)

    def test_parallel_with_true_labels(self):
        y_true = np.random.randint(0, self.n_clusters, size=100)

        result = generate_base_clusterings(
            self.X,
            n_clusters=self.n_clusters,
            m_base=self.m_base,
            random_state=42,
            y_true=y_true,
            use_ray=True,
        )

        assert 'metrics' in result
        assert len(result['metrics']) == self.m_base

        for metrics in result['metrics']:
            assert 'acc' in metrics
            assert 'nmi' in metrics
            assert 'purity' in metrics

    def test_parallel_fallback_when_ray_unavailable(self):
        shutdown_ray_if_initialized()

        result = generate_base_clusterings(
            self.X,
            n_clusters=self.n_clusters,
            m_base=self.m_base,
            random_state=42,
            use_ray=True,
        )

        assert 'W' in result
        assert 'G' in result
        assert 'F' in result
        assert 'labels' in result


class TestRayIntegration:
    def test_ray_utils_importable(self):
        from src.utils import ray_utils

        assert hasattr(ray_utils, 'is_ray_available')
        assert hasattr(ray_utils, 'init_ray_if_needed')
        assert hasattr(ray_utils, 'shutdown_ray_if_initialized')
        assert hasattr(ray_utils, 'get_ray_status')

    def test_ray_status_without_initialization(self):
        shutdown_ray_if_initialized()

        status = get_ray_status()
        assert status['initialized'] is False
        assert status['num_cpus'] is None

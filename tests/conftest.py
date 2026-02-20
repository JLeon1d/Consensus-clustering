"""Pytest configuration and fixtures."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "ray: mark test as requiring Ray (deselect with '-m \"not ray\"')"
    )


@pytest.fixture
def synthetic_data():
    """Generate synthetic clustering data."""
    X, y = make_blobs(
        n_samples=300, n_features=10, centers=5, cluster_std=1.0, random_state=42
    )
    return X, y


@pytest.fixture
def small_synthetic_data():
    """Generate small synthetic data for quick tests."""
    X, y = make_blobs(n_samples=50, n_features=5, centers=3, cluster_std=0.5, random_state=42)
    return X, y


@pytest.fixture
def tiny_data():
    """Generate tiny data for unit tests."""
    np.random.seed(42)
    X = np.random.randn(20, 4)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    return X, y


@pytest.fixture
def ray_context():
    """
    Initialize and shutdown Ray for tests.
    
    This fixture is only used by tests marked with @pytest.mark.ray.
    Tests without Ray can run independently.
    """
    try:
        import ray
        
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True, logging_level="ERROR")
        yield
        ray.shutdown()
    except ImportError:
        pytest.skip("Ray is not installed")


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip Ray tests if Ray is not installed.
    
    This allows running tests without Ray by either:
    - Not having Ray installed (tests will be skipped)
    - Using: pytest -m "not ray"
    """
    try:
        import ray  # noqa: F401
        ray_available = True
    except ImportError:
        ray_available = False
    
    if not ray_available:
        skip_ray = pytest.mark.skip(reason="Ray is not installed")
        for item in items:
            if "ray" in item.keywords:
                item.add_marker(skip_ray)
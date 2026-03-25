"""Example tests showing Ray test marking."""

import pytest
import ray


class TestRayMarking:
    """Example tests to demonstrate Ray test marking."""

    def test_without_ray(self):
        """This test runs without Ray."""
        assert 1 + 1 == 2

    @pytest.mark.ray
    def test_with_ray(self, ray_context):
        """
        This test requires Ray and will be skipped if Ray is not installed.

        Run with: pytest -m ray
        Skip with: pytest -m "not ray"
        """

        @ray.remote
        def add(a, b):
            return a + b

        result = ray.get(add.remote(1, 2))
        assert result == 3

    @pytest.mark.ray
    @pytest.mark.slow
    def test_ray_and_slow(self, ray_context):
        """
        This test is both Ray-dependent and slow.

        Skip with: pytest -m "not ray and not slow"
        """

        @ray.remote
        def multiply(a, b):
            return a * b

        result = ray.get(multiply.remote(3, 4))
        assert result == 12
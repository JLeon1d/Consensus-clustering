"""Example tests showing Ray test marking."""

import pytest
import ray


class TestRayMarking:
    def test_without_ray(self):
        assert 1 + 1 == 2

    @pytest.mark.ray
    def test_with_ray(self, ray_context):
        @ray.remote
        def add(a, b):
            return a + b

        result = ray.get(add.remote(1, 2))
        assert result == 3

    @pytest.mark.ray
    @pytest.mark.slow
    def test_ray_and_slow(self, ray_context):
        @ray.remote
        def multiply(a, b):
            return a * b

        result = ray.get(multiply.remote(3, 4))
        assert result == 12

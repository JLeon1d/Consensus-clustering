import os
import sys
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.acmk import ACMK
from src.sdgca import SDGCA
from src.clustering.base_generation import generate_base_clusterings
from src.generators.data_generator import ClusterDataGenerator
from src.utils.ray_utils import init_ray_if_needed


def run_base_clustering_benchmark(
    n: int,
    k: int,
    m: int,
    use_ray: bool,
    clusterability: float = 0.9,
    verbose: bool = True
) -> Dict:
    """Run base clustering benchmark (k-means only, no consensus step)."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Base Clustering Benchmark: n={n}, k={k}, m={m}")
        print(f"Mode: {'Ray Parallel' if use_ray else 'Sequential'}")
        print(f"Clusterability: {clusterability:.2f}")
        print(f"{'='*70}")

    if use_ray:
        init_ray_if_needed(use_ray=True)

    if verbose:
        print("Generating synthetic data...")
    generator = ClusterDataGenerator(random_state=42)
    X, _ = generator.generate(
        n_samples=n,
        n_features=50,
        n_clusters=k,
        mode='blobs',
        clusterability=clusterability
    )

    if verbose:
        print("Generating base clusterings...")
    t0 = time.time()
    generate_base_clusterings(X, n_clusters=k, m_base=m, use_ray=use_ray)
    elapsed = time.time() - t0

    if verbose:
        print(f"  Base clustering time: {elapsed:.2f}s")

    return {
        'algorithm': 'base_clustering',
        'n_samples': n,
        'n_clusters': k,
        'm_base': m,
        'use_ray': use_ray,
        'algorithm_time': elapsed,
        'total_time': elapsed,
    }


def run_acmk_benchmark(
    n: int,
    k: int,
    m: int,
    max_iter: int,
    use_ray: bool,
    clusterability: float = 0.7,
    verbose: bool = True
) -> Dict:
    """Run ACMK benchmark with specified parameters."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"ACMK Benchmark: n={n}, k={k}, m={m}, max_iter={max_iter}")
        print(f"Mode: {'Ray Parallel' if use_ray else 'Sequential'}")
        print(f"Clusterability: {clusterability:.2f}")
        print(f"{'='*70}")

    if verbose:
        print("Generating synthetic data...")
    generator = ClusterDataGenerator(random_state=42)
    X, y_true = generator.generate(
        n_samples=n,
        n_features=50,
        n_clusters=k,
        mode='blobs',
        clusterability=clusterability
    )

    if verbose:
        print("Running ACMK...")
    acmk_start = time.time()
    acmk = ACMK(
        n_clusters=k,
        m_base=m,
        max_iter=max_iter,
        use_ray=use_ray,
        verbose=False
    )
    acmk.fit(X)
    acmk_time = time.time() - acmk_start

    if verbose:
        print(f"\n  ACMK time: {acmk_time:.2f}s")

    return {
        'algorithm': 'ACMK',
        'n_samples': n,
        'n_clusters': k,
        'm_base': m,
        'max_iter': max_iter,
        'use_ray': use_ray,
        'algorithm_time': acmk_time,
        'total_time': acmk_time
    }


def run_sdgca_benchmark(
    n: int,
    k: int,
    m: int,
    max_iter: int,
    use_ray: bool,
    clusterability: float = 0.7,
    verbose: bool = True
) -> Dict:
    """Run SDGCA benchmark with specified parameters."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"SDGCA Benchmark: n={n}, k={k}, m={m}, max_iter={max_iter}")
        print(f"Mode: {'Ray Parallel' if use_ray else 'Sequential'}")
        print(f"Clusterability: {clusterability:.2f}")
        print(f"{'='*70}")

    if verbose:
        print("Generating synthetic data...")
    generator = ClusterDataGenerator(random_state=42)
    X, y_true = generator.generate(
        n_samples=n,
        n_features=50,
        n_clusters=k,
        mode='blobs',
        clusterability=clusterability
    )

    if verbose:
        print("Running SDGCA...")
    sdgca_start = time.time()
    sdgca = SDGCA(
        n_clusters=k,
        m_base=m,
        max_iter=max_iter,
        use_ray=use_ray,
        verbose=verbose
    )
    sdgca.fit(X)
    sdgca_time = time.time() - sdgca_start

    if verbose:
        print(f"\n  SDGCA time: {sdgca_time:.2f}s")

    return {
        'algorithm': 'SDGCA',
        'n_samples': n,
        'n_clusters': k,
        'm_base': m,
        'max_iter': max_iter,
        'use_ray': use_ray,
        'algorithm_time': sdgca_time,
        'total_time': sdgca_time
    }

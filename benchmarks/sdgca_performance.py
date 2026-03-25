"""Performance benchmarks for SDGCA algorithm."""

import time
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from consensus_clustering import SDGCA


def generate_base_clusterings_simple(
    X: np.ndarray, n_clusters: int, n_base: int, random_state: int = 42
) -> np.ndarray:
    """Generate base clusterings using k-means with different initializations."""
    n_samples = X.shape[0]
    base_clusterings = np.zeros((n_samples, n_base), dtype=int)

    for i in range(n_base):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=10)
        base_clusterings[:, i] = kmeans.fit_predict(X) + 1

    return base_clusterings


def benchmark_sdgca_scaling(
    n_samples_list: List[int] = [50, 100, 200, 300],
    n_features: int = 10,
    n_clusters: int = 3,
    n_base: int = 15,
) -> Dict:
    """Benchmark SDGCA with different dataset sizes."""
    results = {
        "n_samples": [],
        "time": [],
        "ari": [],
        "nmi": [],
    }

    for n_samples in n_samples_list:
        print(f"\nBenchmarking with {n_samples} samples...")

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=1.0,
            random_state=42,
        )

        base_clusterings = generate_base_clusterings_simple(X, n_clusters, n_base)

        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=0.1,
            eta=0.7,
            theta=0.6,
            max_iter=100,
            verbose=False,
        )

        start_time = time.time()
        labels = sdgca.fit_predict(base_clusterings)
        elapsed_time = time.time() - start_time

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results["n_samples"].append(n_samples)
        results["time"].append(elapsed_time)
        results["ari"].append(ari)
        results["nmi"].append(nmi)

        print(f"  Time: {elapsed_time:.3f}s, ARI: {ari:.3f}, NMI: {nmi:.3f}")

    return results


def benchmark_sdgca_base_clusterings(
    n_samples: int = 100,
    n_features: int = 10,
    n_clusters: int = 3,
    n_base_list: List[int] = [5, 10, 20, 30],
) -> Dict:
    """Benchmark SDGCA with different numbers of base clusterings."""
    results = {
        "n_base": [],
        "time": [],
        "ari": [],
        "nmi": [],
    }

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )

    for n_base in n_base_list:
        print(f"\nBenchmarking with {n_base} base clusterings...")

        base_clusterings = generate_base_clusterings_simple(X, n_clusters, n_base)

        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=0.1,
            eta=0.7,
            theta=0.6,
            max_iter=100,
            verbose=False,
        )

        start_time = time.time()
        labels = sdgca.fit_predict(base_clusterings)
        elapsed_time = time.time() - start_time

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results["n_base"].append(n_base)
        results["time"].append(elapsed_time)
        results["ari"].append(ari)
        results["nmi"].append(nmi)

        print(f"  Time: {elapsed_time:.3f}s, ARI: {ari:.3f}, NMI: {nmi:.3f}")

    return results


def benchmark_sdgca_parameters(
    n_samples: int = 100,
    n_features: int = 10,
    n_clusters: int = 3,
    n_base: int = 15,
) -> Dict:
    """Benchmark SDGCA with different parameter settings."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )

    base_clusterings = generate_base_clusterings_simple(X, n_clusters, n_base)

    param_sets = [
        {"lambda_param": 0.05, "eta": 0.65, "theta": 0.7, "name": "Set 1"},
        {"lambda_param": 0.09, "eta": 0.75, "theta": 0.65, "name": "Set 2 (default)"},
        {"lambda_param": 0.15, "eta": 0.8, "theta": 0.6, "name": "Set 3"},
        {"lambda_param": 0.1, "eta": 1.5, "theta": 0.6, "name": "NWCA only"},
    ]

    results = {
        "param_set": [],
        "time": [],
        "ari": [],
        "nmi": [],
    }

    for params in param_sets:
        name = params.pop("name")
        print(f"\nBenchmarking {name}...")

        sdgca = SDGCA(n_clusters=n_clusters, max_iter=100, verbose=False, **params)

        start_time = time.time()
        labels = sdgca.fit_predict(base_clusterings)
        elapsed_time = time.time() - start_time

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results["param_set"].append(name)
        results["time"].append(elapsed_time)
        results["ari"].append(ari)
        results["nmi"].append(nmi)

        print(f"  Time: {elapsed_time:.3f}s, ARI: {ari:.3f}, NMI: {nmi:.3f}")

    return results


def benchmark_sdgca_convergence(
    n_samples: int = 100,
    n_features: int = 10,
    n_clusters: int = 3,
    n_base: int = 15,
    max_iter_list: List[int] = [50, 100, 200, 300],
) -> Dict:
    """Benchmark SDGCA convergence with different iteration limits."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )

    base_clusterings = generate_base_clusterings_simple(X, n_clusters, n_base)

    results = {
        "max_iter": [],
        "time": [],
        "ari": [],
        "nmi": [],
    }

    for max_iter in max_iter_list:
        print(f"\nBenchmarking with max_iter={max_iter}...")

        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=0.1,
            eta=0.7,
            theta=0.6,
            max_iter=max_iter,
            verbose=False,
        )

        start_time = time.time()
        labels = sdgca.fit_predict(base_clusterings)
        elapsed_time = time.time() - start_time

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results["max_iter"].append(max_iter)
        results["time"].append(elapsed_time)
        results["ari"].append(ari)
        results["nmi"].append(nmi)

        print(f"  Time: {elapsed_time:.3f}s, ARI: {ari:.3f}, NMI: {nmi:.3f}")

    return results


def run_all_benchmarks():
    """Run all SDGCA benchmarks."""
    print("=" * 70)
    print("SDGCA Performance Benchmarks")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1. Scaling with Dataset Size")
    print("=" * 70)
    scaling_results = benchmark_sdgca_scaling()

    print("\n" + "=" * 70)
    print("2. Scaling with Number of Base Clusterings")
    print("=" * 70)
    base_results = benchmark_sdgca_base_clusterings()

    print("\n" + "=" * 70)
    print("3. Different Parameter Settings")
    print("=" * 70)
    param_results = benchmark_sdgca_parameters()

    print("\n" + "=" * 70)
    print("4. Convergence with Different Iteration Limits")
    print("=" * 70)
    convergence_results = benchmark_sdgca_convergence()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nDataset Size Scaling:")
    for i, n in enumerate(scaling_results["n_samples"]):
        print(
            f"  {n:4d} samples: {scaling_results['time'][i]:6.3f}s, "
            f"ARI: {scaling_results['ari'][i]:.3f}, NMI: {scaling_results['nmi'][i]:.3f}"
        )

    print("\nBase Clusterings Scaling:")
    for i, n in enumerate(base_results["n_base"]):
        print(
            f"  {n:2d} base: {base_results['time'][i]:6.3f}s, "
            f"ARI: {base_results['ari'][i]:.3f}, NMI: {base_results['nmi'][i]:.3f}"
        )

    print("\nParameter Settings:")
    for i, name in enumerate(param_results["param_set"]):
        print(
            f"  {name:20s}: {param_results['time'][i]:6.3f}s, "
            f"ARI: {param_results['ari'][i]:.3f}, NMI: {param_results['nmi'][i]:.3f}"
        )

    print("\nConvergence:")
    for i, max_iter in enumerate(convergence_results["max_iter"]):
        print(
            f"  max_iter={max_iter:3d}: {convergence_results['time'][i]:6.3f}s, "
            f"ARI: {convergence_results['ari'][i]:.3f}, NMI: {convergence_results['nmi'][i]:.3f}"
        )

    return {
        "scaling": scaling_results,
        "base_clusterings": base_results,
        "parameters": param_results,
        "convergence": convergence_results,
    }


if __name__ == "__main__":
    np.random.seed(42)
    results = run_all_benchmarks()
    print("\nBenchmarks completed!")

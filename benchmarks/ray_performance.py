"""Performance benchmark comparing sequential vs Ray parallel execution."""

import time
import numpy as np
from consensus_clustering.utils.graph_generator import generate_clustered_graph, adjacency_to_features
from consensus_clustering.clustering.base_generation import generate_base_clusterings
from consensus_clustering.ray_parallel import is_ray_available, shutdown_ray_if_initialized


def benchmark_base_clustering(n_nodes, n_clusters, m_base, n_runs=3):
    """
    Benchmark base clustering generation with different graph sizes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in graph
    n_clusters : int
        Number of clusters
    m_base : int
        Number of base clusterings
    n_runs : int
        Number of benchmark runs for averaging
    """
    print(f"\nBenchmark: n_nodes={n_nodes}, n_clusters={n_clusters}, m_base={m_base}")
    print("-" * 70)

    adjacency, true_labels = generate_clustered_graph(
        n_nodes=n_nodes,
        n_clusters=n_clusters,
        cluster_density=0.7,
        noise_density=0.05,
        random_state=42
    )

    X = adjacency_to_features(adjacency, method='adjacency')
    print(f"Graph generated: {X.shape[0]} nodes, {X.shape[1]} features")

    seq_times = []
    for run in range(n_runs):
        start = time.time()
        result_seq = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=False
        )
        seq_time = time.time() - start
        seq_times.append(seq_time)

    avg_seq_time = np.mean(seq_times)
    std_seq_time = np.std(seq_times)
    print(f"Sequential: {avg_seq_time:.2f}s ± {std_seq_time:.2f}s")

    ray_times = []
    for run in range(n_runs):
        start = time.time()
        result_ray = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=True
        )
        ray_time = time.time() - start
        ray_times.append(ray_time)

    avg_ray_time = np.mean(ray_times)
    std_ray_time = np.std(ray_times)
    print(f"Ray parallel: {avg_ray_time:.2f}s ± {std_ray_time:.2f}s")

    speedup = avg_seq_time / avg_ray_time
    print(f"Speedup: {speedup:.2f}x")

    w_diff = np.abs(result_seq['W'] - result_ray['W']).max()
    print(f"Max W difference: {w_diff:.2e}")

    shutdown_ray_if_initialized()

    return avg_seq_time, avg_ray_time


def main():
    """Run performance benchmarks with different graph sizes."""
    print("=" * 70)
    print("Ray Performance Benchmark")
    print("=" * 70)

    assert is_ray_available()

    configs = [
        {"n_nodes": 1000, "n_clusters": 10, "m_base": 10},
        {"n_nodes": 2000, "n_clusters": 15, "m_base": 10},
        {"n_nodes": 4000, "n_clusters": 25, "m_base": 10},
        {"n_nodes": 10000, "n_clusters": 30, "m_base": 10},
    ]

    results = []
    for config in configs:
        seq_time, ray_time = benchmark_base_clustering(**config, n_runs=3)
        results.append({
            "config": config,
            "seq_time": seq_time,
            "ray_time": ray_time,
            "speedup": seq_time / ray_time if ray_time else None
        })

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Nodes':<10} {'Clusters':<10} {'Base':<10} {'Sequential':<15} {'Ray':<15} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        c = r["config"]
        seq_str = f"{r['seq_time']:.2f}s"
        ray_str = f"{r['ray_time']:.2f}s" if r['ray_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{c['n_nodes']:<10} {c['n_clusters']:<10} {c['m_base']:<10} {seq_str:<15} {ray_str:<15} {speedup_str:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()

# ======================================================================
# Nodes      Clusters   Base       Sequential      Ray             Speedup
# ----------------------------------------------------------------------
# 1000       10         10         0.63s           11.68s          0.05x
# 2000       15         10         2.93s           7.18s           0.41x
# 4000       25         10         5.87s           7.36s           0.80x
# 10000      30         10         42.38s          42.84s          0.99x
# ======================================================================
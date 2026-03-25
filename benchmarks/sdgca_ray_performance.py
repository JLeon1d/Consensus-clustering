"""Performance benchmark for SDGCA algorithm with Ray support comparison."""

import time

import numpy as np

from consensus_clustering.clustering.base_generation import generate_base_clusterings
from consensus_clustering.core.sdgca import SDGCA
from consensus_clustering.metrics import clustering_measure
from consensus_clustering.ray_parallel import is_ray_available, shutdown_ray_if_initialized
from consensus_clustering.utils.graph_generator import adjacency_to_features, generate_clustered_graph


def benchmark_sdgca(n_nodes, n_clusters, m_base, lambda_param=0.1, eta=0.7, theta=0.6, n_runs=3):
    """
    Benchmark complete SDGCA algorithm with different graph sizes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in graph
    n_clusters : int
        Number of clusters
    m_base : int
        Number of base clusterings
    lambda_param : float
        SDGCA lambda parameter
    eta : float
        SDGCA eta parameter
    theta : float
        SDGCA theta parameter
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
    print(f"Graph: {X.shape[0]} nodes, {X.shape[1]} features")

    # Sequential benchmark
    seq_times = []
    seq_accs = []
    for run in range(n_runs):
        start = time.time()

        # Generate base clusterings without Ray
        base_data = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=False
        )
        
        # Convert labels list to base_clusterings matrix (n_samples, m_base)
        # SDGCA expects 1-indexed labels
        base_clusterings = np.column_stack([labels + 1 for labels in base_data['labels']])

        # Run SDGCA
        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=lambda_param,
            eta=eta,
            theta=theta,
            max_iter=100,
            verbose=False
        )
        labels = sdgca.fit_predict(base_clusterings)

        seq_time = time.time() - start
        seq_times.append(seq_time)

        metrics = clustering_measure(true_labels, labels)
        seq_accs.append(metrics.get('ACC', 0.0))

    avg_seq_time = np.mean(seq_times)
    std_seq_time = np.std(seq_times)
    avg_seq_acc = np.mean(seq_accs)
    print(f"Sequential: {avg_seq_time:.2f}s ± {std_seq_time:.2f}s, ACC: {avg_seq_acc:.4f}")

    # Ray parallel benchmark
    ray_times = []
    ray_accs = []
    for run in range(n_runs):
        start = time.time()

        # Generate base clusterings with Ray
        base_data = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=True
        )
        
        # Convert labels list to base_clusterings matrix (n_samples, m_base)
        # SDGCA expects 1-indexed labels
        base_clusterings = np.column_stack([labels + 1 for labels in base_data['labels']])

        # Run SDGCA
        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=lambda_param,
            eta=eta,
            theta=theta,
            max_iter=100,
            verbose=False
        )
        labels = sdgca.fit_predict(base_clusterings)

        ray_time = time.time() - start
        ray_times.append(ray_time)

        metrics = clustering_measure(true_labels, labels)
        ray_accs.append(metrics.get('ACC', 0.0))

    avg_ray_time = np.mean(ray_times)
    std_ray_time = np.std(ray_times)
    avg_ray_acc = np.mean(ray_accs)
    print(f"Ray parallel: {avg_ray_time:.2f}s ± {std_ray_time:.2f}s, ACC: {avg_ray_acc:.4f}")

    speedup = avg_seq_time / avg_ray_time if avg_ray_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    shutdown_ray_if_initialized()

    return avg_seq_time, avg_ray_time, avg_seq_acc, avg_ray_acc


def main():
    """Run SDGCA performance benchmarks with different graph sizes."""
    print("=" * 70)
    print("SDGCA Algorithm Performance Benchmark (Sequential vs Ray)")
    print("=" * 70)

    if not is_ray_available():
        print("\nERROR: Ray is not available. Please install Ray to run this benchmark.")
        print("Install with: pip install ray")
        return

    configs = [
        {"n_nodes": 200, "n_clusters": 5, "m_base": 10},
        {"n_nodes": 500, "n_clusters": 10, "m_base": 10},
        {"n_nodes": 1000, "n_clusters": 10, "m_base": 10},
    ]

    results = []
    for config in configs:
        seq_time, ray_time, seq_acc, ray_acc = benchmark_sdgca(**config, n_runs=1)
        results.append({
            "config": config,
            "seq_time": seq_time,
            "ray_time": ray_time,
            "seq_acc": seq_acc,
            "ray_acc": ray_acc,
            "speedup": seq_time / ray_time if ray_time else None
        })

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Nodes':<10} {'Clusters':<10} {'Base':<10} {'Sequential':<15} {'Ray':<15} {'Speedup':<10} {'ACC Diff':<10}")
    print("-" * 70)

    for r in results:
        c = r["config"]
        seq_str = f"{r['seq_time']:.2f}s"
        ray_str = f"{r['ray_time']:.2f}s" if r['ray_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        acc_diff = abs(r['seq_acc'] - r['ray_acc']) if r['ray_acc'] else 0
        acc_diff_str = f"{acc_diff:.4f}"
        print(f"{c['n_nodes']:<10} {c['n_clusters']:<10} {c['m_base']:<10} {seq_str:<15} {ray_str:<15} {speedup_str:<10} {acc_diff_str:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()

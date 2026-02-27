"""Performance benchmark for complete ACMK algorithm with Ray support."""

import time
import numpy as np
from consensus_clustering.utils.graph_generator import generate_clustered_graph, adjacency_to_features
from consensus_clustering.clustering.base_generation import generate_base_clusterings
from consensus_clustering.core.acmk import ACMK
from consensus_clustering.ray_parallel import is_ray_available, shutdown_ray_if_initialized


def benchmark_acmk(n_nodes, n_clusters, m_base, lambda_=0.1, n_runs=3):
    """
    Benchmark complete ACMK algorithm with different graph sizes.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in graph
    n_clusters : int
        Number of clusters
    m_base : int
        Number of base clusterings
    lambda_ : float
        ACMK regularization parameter
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
    
    seq_times = []
    seq_accs = []
    for run in range(n_runs):
        start = time.time()
        
        base_data = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=False
        )
        
        acmk = ACMK(n_clusters=n_clusters, m_base=m_base, lambda_=lambda_)
        acmk.fit(X, **base_data)
        labels = acmk.predict(method='spectral')
        
        seq_time = time.time() - start
        seq_times.append(seq_time)
        
        from consensus_clustering.metrics import clustering_measure
        metrics = clustering_measure(true_labels, labels)
        seq_accs.append(metrics.get('ACC', 0.0))
    
    avg_seq_time = np.mean(seq_times)
    std_seq_time = np.std(seq_times)
    avg_seq_acc = np.mean(seq_accs)
    print(f"Sequential: {avg_seq_time:.2f}s ± {std_seq_time:.2f}s, ACC: {avg_seq_acc:.4f}")
    
    ray_times = []
    ray_accs = []
    for run in range(n_runs):
        start = time.time()
            
        base_data = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42 + run,
            use_ray=True
        )
            
        acmk = ACMK(n_clusters=n_clusters, m_base=m_base, lambda_=lambda_)
        acmk.fit(X, **base_data)
        labels = acmk.predict(method='spectral')
            
        ray_time = time.time() - start
        ray_times.append(ray_time)
            
        from consensus_clustering.metrics import clustering_measure
        metrics = clustering_measure(true_labels, labels)
        ray_accs.append(metrics.get('ACC', 0.0))

    avg_ray_time = np.mean(ray_times)
    std_ray_time = np.std(ray_times)
    avg_ray_acc = np.mean(ray_accs)
    print(f"Ray parallel: {avg_ray_time:.2f}s ± {std_ray_time:.2f}s, ACC: {avg_ray_acc:.4f}")
        
    speedup = avg_seq_time / avg_ray_time
    print(f"Speedup: {speedup:.2f}x")
        
    shutdown_ray_if_initialized()

    return avg_seq_time, avg_ray_time, avg_seq_acc, avg_ray_acc


def main():
    """Run ACMK performance benchmarks with different graph sizes."""
    print("=" * 70)
    print("ACMK Algorithm Performance Benchmark")
    print("=" * 70)
    
    assert is_ray_available()

    configs = [
        {"n_nodes": 200, "n_clusters": 5, "m_base": 10},
        {"n_nodes": 500, "n_clusters": 10, "m_base": 10},
        {"n_nodes": 1000, "n_clusters": 10, "m_base": 10},
    ]
    
    results = []
    for config in configs:
        seq_time, ray_time, seq_acc, ray_acc = benchmark_acmk(**config, n_runs=1)
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


# ======================================================================
# Summary
# ======================================================================
# Nodes      Clusters   Base       Sequential      Ray             Speedup    ACC Diff
# ----------------------------------------------------------------------
# 200        5          10         6.83s           19.52s          0.35x      0.0000
# 500        10         10         57.44s          74.72s          0.77x      0.0000
# 1000       10         10         176.81s         184.33s         0.96x      0.0000
# ======================================================================

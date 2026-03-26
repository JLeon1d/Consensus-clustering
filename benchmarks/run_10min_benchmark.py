"""10-minute benchmark for ACMK and SDGCA with progress logging."""

import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmark_framework import BenchmarkFramework

# Import algorithms
try:
    from consensus_clustering.clustering.base_generation import generate_base_clusterings
    from consensus_clustering.core.acmk import ACMK
    from consensus_clustering.core.sdgca import SDGCA
    from consensus_clustering.utils.graph_generator import adjacency_to_features, generate_clustered_graph
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from clustering.base_generation import generate_base_clusterings
    from core.acmk import ACMK
    from core.sdgca import SDGCA
    from utils.graph_generator import adjacency_to_features, generate_clustered_graph


def generate_base_clusterings_simple(X, n_clusters, n_base, random_state=42):
    """Generate base clusterings using k-means."""
    n_samples = X.shape[0]
    base_clusterings = np.zeros((n_samples, n_base), dtype=int)
    
    for i in range(n_base):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=10)
        base_clusterings[:, i] = kmeans.fit_predict(X) + 1
    
    return base_clusterings


def acmk_benchmark_func(config: dict, use_ray: bool) -> dict:
    """ACMK benchmark function."""
    print(f"    [ACMK] Generating graph with {config['n_samples']} nodes...")
    
    adjacency, true_labels = generate_clustered_graph(
        n_nodes=config['n_samples'],
        n_clusters=config['n_clusters'],
        cluster_density=0.7,
        noise_density=0.05,
        random_state=config.get('random_state', 42)
    )
    
    X = adjacency_to_features(adjacency, method='adjacency')
    
    print(f"    [ACMK] Generating {config['n_base']} base clusterings...")
    base_data = generate_base_clusterings(
        X,
        n_clusters=config['n_clusters'],
        m_base=config['n_base'],
        random_state=config.get('random_state', 42),
        use_ray=use_ray
    )
    
    print(f"    [ACMK] Running ACMK optimization...")
    acmk = ACMK(
        n_clusters=config['n_clusters'],
        m_base=config['n_base'],
        lambda_=config.get('lambda_', 0.1),
        max_iter=20,
        verbose=False
    )
    acmk.fit(X, **base_data)
    labels = acmk.predict(method='spectral')
    
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    
    return {'ari': ari, 'nmi': nmi, 'accuracy': ari}


def sdgca_benchmark_func(config: dict, use_ray: bool) -> dict:
    """SDGCA benchmark function."""
    print(f"    [SDGCA] Generating data with {config['n_samples']} samples...")
    
    X, y_true = make_blobs(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        centers=config['n_clusters'],
        cluster_std=1.0,
        random_state=config.get('random_state', 42)
    )
    
    print(f"    [SDGCA] Generating {config['n_base']} base clusterings...")
    base_clusterings = generate_base_clusterings_simple(
        X, config['n_clusters'], config['n_base'], config.get('random_state', 42)
    )
    
    print(f"    [SDGCA] Running SDGCA optimization...")
    sdgca = SDGCA(
        n_clusters=config['n_clusters'],
        lambda_param=0.09,
        eta=0.75,
        theta=0.65,
        max_iter=300,
        verbose=False
    )
    
    labels = sdgca.fit_predict(base_clusterings)
    
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    
    return {'ari': ari, 'nmi': nmi, 'accuracy': ari}


def main():
    """Run 10-minute benchmark."""
    print("="*70)
    print("10-Minute Benchmark - ACMK and SDGCA")
    print("="*70)
    print("\nThis benchmark will run for approximately 10 minutes.")
    print("Progress will be shown for each step.\n")
    
    framework = BenchmarkFramework()
    
    start_time = time.time()
    
    # ACMK: 2 configs, designed for ~5 minutes total
    print("\n" + "="*70)
    print("PART 1: ACMK Benchmarks (~5 minutes)")
    print("="*70)
    
    acmk_configs = [
        {
            'n_samples': 400,
            'n_clusters': 5,
            'n_base': 10,
            'lambda_': 0.1,
            'random_state': 42
        },
        {
            'n_samples': 600,
            'n_clusters': 8,
            'n_base': 10,
            'lambda_': 0.1,
            'random_state': 43
        },
    ]
    
    for i, config in enumerate(acmk_configs):
        print(f"\n--- ACMK Config {i+1}/{len(acmk_configs)} ---")
        print(f"Nodes: {config['n_samples']}, Clusters: {config['n_clusters']}, Base: {config['n_base']}")
        
        comparison = framework.compare_ray_vs_sequential(
            algorithm_name="ACMK",
            benchmark_func=acmk_benchmark_func,
            config=config,
            n_runs=1  # 1 run per mode for speed
        )
        
        if comparison['speedup']:
            print(f"\n✓ Config {i+1} complete:")
            print(f"  Sequential: {comparison['sequential']['mean_time']:.2f}s")
            print(f"  Ray: {comparison['ray']['mean_time']:.2f}s")
            print(f"  Speedup: {comparison['speedup']:.2f}x")
    
    # SDGCA: 2 configs, designed for ~5 minutes total
    print("\n" + "="*70)
    print("PART 2: SDGCA Benchmarks (~5 minutes)")
    print("="*70)
    
    sdgca_configs = [
        {
            'n_samples': 300,
            'n_features': 30,
            'n_clusters': 8,
            'n_base': 20,
            'random_state': 42
        },
        {
            'n_samples': 400,
            'n_features': 40,
            'n_clusters': 10,
            'n_base': 25,
            'random_state': 43
        },
    ]
    
    for i, config in enumerate(sdgca_configs):
        print(f"\n--- SDGCA Config {i+1}/{len(sdgca_configs)} ---")
        print(f"Samples: {config['n_samples']}, Features: {config['n_features']}, Clusters: {config['n_clusters']}, Base: {config['n_base']}")
        
        comparison = framework.compare_ray_vs_sequential(
            algorithm_name="SDGCA",
            benchmark_func=sdgca_benchmark_func,
            config=config,
            n_runs=1  # 1 run per mode for speed
        )
        
        if comparison['speedup']:
            print(f"\n✓ Config {i+1} complete:")
            print(f"  Sequential: {comparison['sequential']['mean_time']:.2f}s")
            print(f"  Ray: {comparison['ray']['mean_time']:.2f}s")
            print(f"  Speedup: {comparison['speedup']:.2f}x")
    
    # Generate reports
    print("\n" + "="*70)
    print("Generating Reports...")
    print("="*70)
    
    framework.generate_markdown_report("results/10min_benchmark_results.md")
    framework.save_json("results/10min_benchmark_results.json")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ Benchmark Complete!")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"\nResults saved to:")
    print(f"  - results/10min_benchmark_results.md")
    print(f"  - results/10min_benchmark_results.json")


if __name__ == "__main__":
    main()

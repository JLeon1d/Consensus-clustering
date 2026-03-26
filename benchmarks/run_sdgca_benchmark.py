"""SDGCA benchmark with Ray comparison - designed to run ~10 minutes without Ray."""

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmark_framework import BenchmarkFramework

# Import SDGCA
try:
    from consensus_clustering import SDGCA
except ImportError:
    # Try alternative import path
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from core.sdgca import SDGCA


def generate_base_clusterings_simple(
    X: np.ndarray, n_clusters: int, n_base: int, random_state: int = 42
) -> np.ndarray:
    """Generate base clusterings using k-means with different initializations."""
    n_samples = X.shape[0]
    base_clusterings = np.zeros((n_samples, n_base), dtype=int)
    
    for i in range(n_base):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state + i,
            n_init=10,
            max_iter=300
        )
        base_clusterings[:, i] = kmeans.fit_predict(X) + 1  # 1-indexed
    
    return base_clusterings


def sdgca_benchmark_func(config: dict, use_ray: bool) -> dict:
    """
    Run SDGCA benchmark with given configuration.
    
    Note: SDGCA itself doesn't use Ray, but we include the parameter
    for consistency with the framework. Ray could be used for base clustering
    generation in future versions.
    
    Parameters
    ----------
    config : dict
        Configuration with keys: n_samples, n_features, n_clusters, n_base
    use_ray : bool
        Whether to use Ray (currently not used in SDGCA)
        
    Returns
    -------
    metrics : dict
        Dictionary with performance metrics
    """
    n_samples = config['n_samples']
    n_features = config['n_features']
    n_clusters = config['n_clusters']
    n_base = config['n_base']
    random_state = config.get('random_state', 42)
    
    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    
    # Generate base clusterings
    base_clusterings = generate_base_clusterings_simple(
        X, n_clusters, n_base, random_state
    )
    
    # Run SDGCA
    sdgca = SDGCA(
        n_clusters=n_clusters,
        lambda_param=0.09,
        eta=0.75,
        theta=0.65,
        max_iter=300,
        verbose=False
    )
    
    labels = sdgca.fit_predict(base_clusterings)
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    
    return {
        'ari': ari,
        'nmi': nmi,
        'accuracy': ari  # Use ARI as accuracy measure
    }


def main():
    """Run SDGCA benchmarks."""
    print("="*70)
    print("SDGCA Algorithm Benchmark - Ray vs Sequential")
    print("="*70)
    print("\nNote: SDGCA currently doesn't use Ray internally,")
    print("but this benchmark measures baseline performance for future optimization.")
    
    framework = BenchmarkFramework(output_dir="benchmarks/results")
    
    # Configurations designed to run ~10 minutes total
    # SDGCA scales with O(n^2) due to co-association matrix computation
    # Based on previous runs: 100 samples ~0.5s, 200 samples ~2s, 300 samples ~5s
    configs = [
        {
            'n_samples': 200,
            'n_features': 20,
            'n_clusters': 5,
            'n_base': 20,
            'random_state': 42
        },
        {
            'n_samples': 300,
            'n_features': 30,
            'n_clusters': 8,
            'n_base': 20,
            'random_state': 43
        },
        {
            'n_samples': 400,
            'n_features': 40,
            'n_clusters': 10,
            'n_base': 25,
            'random_state': 44
        },
        {
            'n_samples': 500,
            'n_features': 50,
            'n_clusters': 12,
            'n_base': 25,
            'random_state': 45
        },
    ]
    
    # Run benchmarks for each configuration
    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*70}")
        
        comparison = framework.compare_ray_vs_sequential(
            algorithm_name="SDGCA",
            benchmark_func=sdgca_benchmark_func,
            config=config,
            n_runs=2  # 2 runs per configuration
        )
        
        # Print comparison
        if comparison['speedup']:
            print(f"\nSpeedup: {comparison['speedup']:.2f}x")
            print(f"Sequential: {comparison['sequential']['mean_time']:.2f}s ± {comparison['sequential']['std_time']:.2f}s")
            print(f"Ray: {comparison['ray']['mean_time']:.2f}s ± {comparison['ray']['std_time']:.2f}s")
        else:
            print(f"\nSequential: {comparison['sequential']['mean_time']:.2f}s ± {comparison['sequential']['std_time']:.2f}s")
            print("(Ray performance same as sequential - no parallelization yet)")
    
    # Generate reports
    print("\n" + "="*70)
    print("Generating reports...")
    print("="*70)
    
    framework.generate_markdown_report("benchmarks/results/sdgca_benchmark_results.md")
    framework.save_json("benchmarks/results/sdgca_benchmark_results.json")
    
    print("\n✓ SDGCA benchmarks completed!")


if __name__ == "__main__":
    main()

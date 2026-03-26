"""Quick test benchmark to verify framework works."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from benchmark_framework import BenchmarkFramework


def quick_test_func(config: dict, use_ray: bool) -> dict:
    """Quick test benchmark function."""
    n_samples = config['n_samples']
    n_clusters = config['n_clusters']
    
    # Generate simple data
    X, y_true = make_blobs(n_samples=n_samples, n_features=10, centers=n_clusters, random_state=42)
    
    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate ARI
    ari = adjusted_rand_score(y_true, labels)
    
    return {'ari': ari, 'nmi': ari}  # Use ARI for both for simplicity


def main():
    """Run quick test benchmark."""
    print("Quick Test Benchmark")
    print("=" * 50)
    
    framework = BenchmarkFramework()
    
    # Small config for quick test
    config = {
        'n_samples': 100,
        'n_clusters': 3,
    }
    
    comparison = framework.compare_ray_vs_sequential(
        algorithm_name="QuickTest",
        benchmark_func=quick_test_func,
        config=config,
        n_runs=2
    )
    
    print("\nResults:")
    print(f"Sequential: {comparison['sequential']['mean_time']:.3f}s")
    print(f"Ray: {comparison['ray']['mean_time']:.3f}s")
    if comparison['speedup']:
        print(f"Speedup: {comparison['speedup']:.2f}x")
    
    # Generate report
    framework.generate_markdown_report("results/quick_test_results.md")
    print("\n✓ Quick test completed!")


if __name__ == "__main__":
    main()

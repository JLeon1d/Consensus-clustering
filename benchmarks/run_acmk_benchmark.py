"""ACMK benchmark with Ray comparison - designed to run ~10 minutes without Ray."""

import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmark_framework import BenchmarkFramework

# Import ACMK components
try:
    from consensus_clustering.clustering.base_generation import generate_base_clusterings
    from consensus_clustering.core.acmk import ACMK
    from consensus_clustering.utils.graph_generator import adjacency_to_features, generate_clustered_graph
except ImportError:
    # Try alternative import path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from clustering.base_generation import generate_base_clusterings
    from core.acmk import ACMK
    from utils.graph_generator import adjacency_to_features, generate_clustered_graph


def acmk_benchmark_func(config: dict, use_ray: bool) -> dict:
    """
    Run ACMK benchmark with given configuration.
    
    Parameters
    ----------
    config : dict
        Configuration with keys: n_samples, n_features, n_clusters, n_base, lambda_
    use_ray : bool
        Whether to use Ray for parallel processing
        
    Returns
    -------
    metrics : dict
        Dictionary with performance metrics
    """
    n_samples = config['n_samples']
    n_clusters = config['n_clusters']
    n_base = config['n_base']
    lambda_ = config.get('lambda_', 0.1)
    random_state = config.get('random_state', 42)
    
    # Generate synthetic graph data
    adjacency, true_labels = generate_clustered_graph(
        n_nodes=n_samples,
        n_clusters=n_clusters,
        cluster_density=0.7,
        noise_density=0.05,
        random_state=random_state
    )
    
    X = adjacency_to_features(adjacency, method='adjacency')
    
    # Generate base clusterings
    base_data = generate_base_clusterings(
        X,
        n_clusters=n_clusters,
        m_base=n_base,
        random_state=random_state,
        use_ray=use_ray
    )
    
    # Run ACMK
    acmk = ACMK(
        n_clusters=n_clusters,
        m_base=n_base,
        lambda_=lambda_,
        max_iter=20,
        verbose=False
    )
    acmk.fit(X, **base_data)
    labels = acmk.predict(method='spectral')
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    
    return {
        'ari': ari,
        'nmi': nmi,
        'accuracy': ari  # Use ARI as accuracy measure
    }


def main():
    """Run ACMK benchmarks."""
    print("="*70)
    print("ACMK Algorithm Benchmark - Ray vs Sequential")
    print("="*70)
    
    framework = BenchmarkFramework(output_dir="benchmarks/results")
    
    # Configurations designed to run ~10 minutes total without Ray
    # Adjusted based on the previous benchmark results showing ~177s for 1000 nodes
    configs = [
        {
            'n_samples': 300,
            'n_features': 300,  # Will be determined by adjacency matrix
            'n_clusters': 5,
            'n_base': 10,
            'lambda_': 0.1,
            'random_state': 42
        },
        {
            'n_samples': 500,
            'n_features': 500,
            'n_clusters': 8,
            'n_base': 10,
            'lambda_': 0.1,
            'random_state': 43
        },
        {
            'n_samples': 800,
            'n_features': 800,
            'n_clusters': 10,
            'n_base': 10,
            'lambda_': 0.1,
            'random_state': 44
        },
    ]
    
    # Run benchmarks for each configuration
    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*70}")
        
        comparison = framework.compare_ray_vs_sequential(
            algorithm_name="ACMK",
            benchmark_func=acmk_benchmark_func,
            config=config,
            n_runs=2  # 2 runs per configuration
        )
        
        # Print comparison
        if comparison['speedup']:
            print(f"\nSpeedup: {comparison['speedup']:.2f}x")
            print(f"Sequential: {comparison['sequential']['mean_time']:.2f}s ± {comparison['sequential']['std_time']:.2f}s")
            print(f"Ray: {comparison['ray']['mean_time']:.2f}s ± {comparison['ray']['std_time']:.2f}s")
    
    # Generate reports
    print("\n" + "="*70)
    print("Generating reports...")
    print("="*70)
    
    framework.generate_markdown_report("benchmarks/results/acmk_benchmark_results.md")
    framework.save_json("benchmarks/results/acmk_benchmark_results.json")
    
    print("\n✓ ACMK benchmarks completed!")


if __name__ == "__main__":
    main()

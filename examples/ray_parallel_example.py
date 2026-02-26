"""Example demonstrating Ray parallel processing for base clustering generation."""

import numpy as np
import time

from consensus_clustering.clustering.base_generation import generate_base_clusterings
from consensus_clustering.ray_parallel import is_ray_available, shutdown_ray_if_initialized


def main():
    """Compare sequential vs parallel base clustering generation."""
    print("Ray Parallel Processing Example")
    print("=" * 70)
    
    ray_available = is_ray_available()
    print(f"\nRay available: {ray_available}")
    
    if not ray_available:
        print("Ray is not installed. Install with: pip install ray>=2.9.0")
        print("Continuing with sequential execution only.\n")
    
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    n_clusters = 5
    m_base = 20
    
    X = np.random.randn(n_samples, n_features)
    y_true = np.random.randint(0, n_clusters, size=n_samples)
    
    print(f"\nData: {X.shape}, Clusters: {n_clusters}, Base clusterings: {m_base}")
    
    print("\nSequential execution...")
    start_time = time.time()
    result_seq = generate_base_clusterings(
        X,
        n_clusters=n_clusters,
        m_base=m_base,
        random_state=42,
        y_true=y_true,
        use_ray=False,
    )
    seq_time = time.time() - start_time
    print(f"Time: {seq_time:.2f}s")
    
    if result_seq.get('metrics'):
        avg_acc = np.mean([m.get('ACC', 0) for m in result_seq['metrics']])
        avg_nmi = np.mean([m.get('NMI', 0) for m in result_seq['metrics']])
        print(f"Average ACC: {avg_acc:.4f}, NMI: {avg_nmi:.4f}")
    
    if ray_available:
        print("\nParallel execution with Ray...")
        start_time = time.time()
        result_par = generate_base_clusterings(
            X,
            n_clusters=n_clusters,
            m_base=m_base,
            random_state=42,
            y_true=y_true,
            use_ray=True,
        )
        par_time = time.time() - start_time
        print(f"Time: {par_time:.2f}s")
        
        if result_par.get('metrics'):
            avg_acc = np.mean([m.get('ACC', 0) for m in result_par['metrics']])
            avg_nmi = np.mean([m.get('NMI', 0) for m in result_par['metrics']])
            print(f"Average ACC: {avg_acc:.4f}, NMI: {avg_nmi:.4f}")
        
        speedup = seq_time / par_time
        print(f"Speedup: {speedup:.2f}x")
        
        w_diff = np.abs(result_seq['W'] - result_par['W']).max()
        print(f"\nConsistency check - Max W difference: {w_diff:.2e}")
        
        shutdown_ray_if_initialized()
    
    print("\n" + "=" * 70)
    print("Example completed")


if __name__ == "__main__":
    main()
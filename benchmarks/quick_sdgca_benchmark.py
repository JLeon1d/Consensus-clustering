"""Quick SDGCA benchmark for testing."""

import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from consensus_clustering import SDGCA


def quick_benchmark():
    """Run a quick benchmark on small dataset."""
    print("Quick SDGCA Benchmark")
    print("=" * 50)
    
    np.random.seed(42)
    n_samples = 1500
    n_features = 40
    n_clusters = 8
    n_base = 60
    
    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
    print(f"Base clusterings: {n_base}")
    
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    
    print("\nGenerating base clusterings...")
    base_clusterings = np.zeros((n_samples, n_base), dtype=int)
    for i in range(n_base):
        kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        base_clusterings[:, i] = kmeans.fit_predict(X) + 1
    
    print("\nRunning SDGCA...")
    sdgca = SDGCA(
        n_clusters=n_clusters,
        lambda_param=0.1,
        eta=0.7,
        theta=0.6,
        max_iter=50,
        verbose=False,
    )
    
    start_time = time.time()
    labels = sdgca.fit_predict(base_clusterings)
    elapsed_time = time.time() - start_time
    
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    
    print(f"\nResults:")
    print(f"  Time: {elapsed_time:.3f}s")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Clusters found: {len(np.unique(labels))}")
    
    print("\n" + "=" * 50)
    print("Benchmark completed!")
    
    return elapsed_time, ari, nmi


if __name__ == "__main__":
    quick_benchmark()

"""Simple example demonstrating SDGCA usage."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from consensus_clustering import SDGCA


def main():
    print("=" * 70)
    print("SDGCA Example")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 150
    n_features = 10
    n_clusters = 3
    n_base = 15

    print(f"\nGenerating synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  True clusters: {n_clusters}")

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )

    print(f"\nGenerating {n_base} base clusterings using k-means...")
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
        max_iter=100,
        verbose=True,
    )

    labels = sdgca.fit_predict(base_clusterings)

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"\nNumber of clusters found: {len(np.unique(labels))}")
    print(f"Cluster distribution: {np.bincount(labels)[1:]}")

    print("\nSDGCA matrices computed:")
    print(f"  Co-association matrix (CA): {sdgca.CA_.shape}")
    print(f"  Normalized weighted CA (NWCA): {sdgca.NWCA_.shape}")
    print(f"  Final refined matrix (W): {sdgca.W_.shape}")

    if sdgca.S_ is not None:
        print(f"  Similarity guidance (S): {sdgca.S_.shape}")
    if sdgca.D_ is not None:
        print(f"  Dissimilarity guidance (D): {sdgca.D_.shape}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

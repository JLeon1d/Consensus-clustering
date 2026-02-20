"""Simple example of using ACMK consensus clustering."""

import numpy as np
from sklearn.datasets import make_blobs

from consensus_clustering import ACMK, clustering_measure, generate_base_clusterings


def main():
    """Run a simple ACMK clustering example."""
    print("=" * 60)
    print("ACMK Consensus Clustering - Simple Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y_true = make_blobs(
        n_samples=200, n_features=10, centers=5, cluster_std=1.0, random_state=42
    )
    print(f"   Data shape: {X.shape}")
    print(f"   Number of true clusters: {len(np.unique(y_true))}")

    # Generate base clusterings
    print("\n2. Generating base clusterings...")
    n_clusters = 5
    m_base = 10

    base_data = generate_base_clusterings(
        X, n_clusters=n_clusters, m_base=m_base, random_state=42, y_true=y_true
    )

    print(f"   Number of base clusterings: {m_base}")
    print(f"   Consensus matrix shape: {base_data['W'].shape}")

    # Show base clustering quality
    if "metrics" in base_data:
        print("\n   Base clustering metrics:")
        for i, metrics in enumerate(base_data["metrics"][:3]):  # Show first 3
            print(
                f"     Base {i+1}: ACC={metrics['acc']:.3f}, "
                f"NMI={metrics['nmi']:.3f}, Purity={metrics['purity']:.3f}"
            )
        if m_base > 3:
            print(f"     ... and {m_base - 3} more")

    # Run ACMK algorithm
    print("\n3. Running ACMK algorithm...")
    acmk = ACMK(
        n_clusters=n_clusters,
        m_base=m_base,
        lambda_=0.1,
        k_power=3,
        max_iter=20,
        verbose=False,
    )

    acmk.fit(X, **base_data)
    print("   ACMK fitting complete!")

    # Get predictions
    print("\n4. Getting cluster predictions...")
    labels_spectral = acmk.predict(method="spectral")
    labels_kmeans = acmk.predict(method="kmeans")

    # Evaluate results
    print("\n5. Evaluating results...")
    metrics_spectral = clustering_measure(y_true, labels_spectral)
    metrics_kmeans = clustering_measure(y_true, labels_kmeans)

    print("\n   Spectral Clustering Results:")
    print(f"     ACC:    {metrics_spectral['acc']:.4f}")
    print(f"     NMI:    {metrics_spectral['nmi']:.4f}")
    print(f"     Purity: {metrics_spectral['purity']:.4f}")

    print("\n   K-means (on transformed space) Results:")
    print(f"     ACC:    {metrics_kmeans['acc']:.4f}")
    print(f"     NMI:    {metrics_kmeans['nmi']:.4f}")
    print(f"     Purity: {metrics_kmeans['purity']:.4f}")

    # Show learned weights
    print("\n6. Learned base clustering weights (alpha):")
    print(f"   {acmk.alpha_}")
    print(f"   Min weight: {acmk.alpha_.min():.4f}")
    print(f"   Max weight: {acmk.alpha_.max():.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
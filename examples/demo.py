"""Demo script for ACMK consensus clustering with original MATLAB data."""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

from consensus_clustering import ACMK, clustering_measure, generate_base_clusterings
from consensus_clustering.utils.data_io import load_data


def main():
    """Run ACMK demo with original data or synthetic data."""
    print("=" * 70)
    print("ACMK Consensus Clustering - Demo")
    print("=" * 70)

    # Try to load original MATLAB data
    data_path = Path("original_code/orlraws10P_uni.mat")

    if data_path.exists():
        print("\n✓ Found original MATLAB data file")
        print(f"  Loading from: {data_path}")
        X, y = load_data(data_path)
        print(f"  Data shape: {X.shape}")
        print(f"  Number of samples: {X.shape[0]}")
        print(f"  Number of features: {X.shape[1]}")
        if y is not None:
            print(f"  Number of classes: {len(np.unique(y))}")
            n_clusters = len(np.unique(y))
        else:
            print("  No labels found, using n_clusters=10")
            n_clusters = 10
    else:
        print("\n✗ Original MATLAB data not found")
        print("  Generating synthetic data instead...")

        X, y = make_blobs(
            n_samples=200, n_features=50, centers=10, cluster_std=1.5, random_state=42
        )
        n_clusters = 10
        print(f"  Generated data shape: {X.shape}")
        print(f"  Number of clusters: {n_clusters}")

    # Parameters
    m_base = 10
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    k_power = 3

    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of base clusterings: {m_base}")
    print(f"  Lambda values to test: {lambda_values}")
    print(f"  Power of affinity matrix (k): {k_power}")

    # Generate base clusterings
    print("\n" + "=" * 70)
    print("Step 1: Generating Base Clusterings")
    print("=" * 70)

    base_data = generate_base_clusterings(
        X, n_clusters=n_clusters, m_base=m_base, random_state=42, y_true=y
    )

    print(f"✓ Generated {m_base} base clusterings")
    print(f"  Consensus matrix shape: {base_data['W'].shape}")

    if "metrics" in base_data and y is not None:
        print("\n  Base clustering quality:")
        avg_acc = np.mean([m["acc"] for m in base_data["metrics"]])
        avg_nmi = np.mean([m["nmi"] for m in base_data["metrics"]])
        avg_purity = np.mean([m["purity"] for m in base_data["metrics"]])
        print(f"    Average ACC:    {avg_acc:.4f}")
        print(f"    Average NMI:    {avg_nmi:.4f}")
        print(f"    Average Purity: {avg_purity:.4f}")

    # Test different lambda values
    print("\n" + "=" * 70)
    print("Step 2: Running ACMK with Different Lambda Values")
    print("=" * 70)

    results = []

    for i, lambda_val in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] Testing lambda = {lambda_val}")

        acmk = ACMK(
            n_clusters=n_clusters,
            m_base=m_base,
            lambda_=lambda_val,
            k_power=k_power,
            max_iter=20,
            verbose=False,
        )

        acmk.fit(X, **base_data)

        labels_spectral = acmk.predict(method="spectral")
        labels_kmeans = acmk.predict(method="kmeans")

        result = {"lambda": lambda_val, "alpha": acmk.alpha_.copy()}

        if y is not None:
            metrics_spectral = clustering_measure(y, labels_spectral)
            metrics_kmeans = clustering_measure(y, labels_kmeans)

            result["spectral"] = metrics_spectral
            result["kmeans"] = metrics_kmeans

            print(f"  Spectral: ACC={metrics_spectral['acc']:.4f}, "
                  f"NMI={metrics_spectral['nmi']:.4f}")
            print(f"  K-means:  ACC={metrics_kmeans['acc']:.4f}, "
                  f"NMI={metrics_kmeans['nmi']:.4f}")
        else:
            print(f"  Completed (no ground truth for evaluation)")

        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if y is not None:
        print("\nBest Results:")

        best_spectral = max(results, key=lambda r: r["spectral"]["acc"])
        best_kmeans = max(results, key=lambda r: r["kmeans"]["acc"])

        print(f"\n  Best Spectral Clustering:")
        print(f"    Lambda: {best_spectral['lambda']}")
        print(f"    ACC:    {best_spectral['spectral']['acc']:.4f}")
        print(f"    NMI:    {best_spectral['spectral']['nmi']:.4f}")
        print(f"    Purity: {best_spectral['spectral']['purity']:.4f}")

        print(f"\n  Best K-means Clustering:")
        print(f"    Lambda: {best_kmeans['lambda']}")
        print(f"    ACC:    {best_kmeans['kmeans']['acc']:.4f}")
        print(f"    NMI:    {best_kmeans['kmeans']['nmi']:.4f}")
        print(f"    Purity: {best_kmeans['kmeans']['purity']:.4f}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
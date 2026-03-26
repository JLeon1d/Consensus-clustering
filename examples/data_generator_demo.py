"""Demonstration of ClusterDataGenerator with different modes and clusterability levels."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from consensus_clustering.generators import ClusterDataGenerator


def visualize_data(X, y, title, hopkins=None):
    """Visualize 2D projection of data."""
    # Use PCA if data has more than 2 dimensions
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    
    title_text = title
    if hopkins is not None:
        title_text += f"\nHopkins Statistic: {hopkins:.3f}"
    plt.title(title_text)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def demo_modes():
    """Demonstrate different generation modes."""
    print("=" * 70)
    print("ClusterDataGenerator - Mode Demonstration")
    print("=" * 70)
    
    gen = ClusterDataGenerator(random_state=42)
    
    modes = ['blobs', 'moons', 'circles', 'anisotropic', 'varied', 'noisy']
    
    plt.figure(figsize=(15, 10))
    
    for idx, mode in enumerate(modes, 1):
        print(f"\nGenerating '{mode}' data...")
        
        X, y = gen.generate(
            n_samples=500,
            n_features=10,
            n_clusters=5,
            mode=mode,
            clusterability=0.7
        )
        
        metrics = gen.estimate_clusterability(X)
        print(f"  Hopkins statistic: {metrics['hopkins']:.3f}")
        print(f"  Interpretation: {metrics['interpretation']}")
        
        # Visualize
        plt.subplot(2, 3, idx)
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=20)
        plt.title(f"{mode.capitalize()}\nHopkins: {metrics['hopkins']:.3f}")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_generator_modes.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'data_generator_modes.png'")


def demo_clusterability():
    """Demonstrate different clusterability levels."""
    print("\n" + "=" * 70)
    print("ClusterDataGenerator - Clusterability Demonstration")
    print("=" * 70)
    
    gen = ClusterDataGenerator(random_state=42)
    
    clusterability_levels = [1.0, 0.7, 0.5, 0.3, 0.0]
    
    plt.figure(figsize=(15, 6))
    
    for idx, clust_level in enumerate(clusterability_levels, 1):
        print(f"\nGenerating data with clusterability={clust_level:.1f}...")
        
        X, y = gen.generate(
            n_samples=500,
            n_features=10,
            n_clusters=5,
            mode='blobs',
            clusterability=clust_level
        )
        
        metrics = gen.estimate_clusterability(X)
        print(f"  Hopkins statistic: {metrics['hopkins']:.3f}")
        print(f"  Interpretation: {metrics['interpretation']}")
        
        # Visualize
        plt.subplot(1, 5, idx)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=20)
        plt.title(f"Clusterability: {clust_level:.1f}\nHopkins: {metrics['hopkins']:.3f}")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_generator_clusterability.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'data_generator_clusterability.png'")


def demo_large_scale():
    """Demonstrate large-scale data generation."""
    print("\n" + "=" * 70)
    print("ClusterDataGenerator - Large Scale Generation")
    print("=" * 70)
    
    gen = ClusterDataGenerator(random_state=42)
    
    configs = [
        {"n_samples": 1000, "n_features": 50, "n_clusters": 10},
        {"n_samples": 5000, "n_features": 100, "n_clusters": 20},
        {"n_samples": 10000, "n_features": 200, "n_clusters": 50},
    ]
    
    for config in configs:
        print(f"\nGenerating {config['n_samples']} samples, "
              f"{config['n_features']} features, "
              f"{config['n_clusters']} clusters...")
        
        import time
        start = time.time()
        
        X, y = gen.generate(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            n_clusters=config['n_clusters'],
            mode='blobs',
            clusterability=0.7
        )
        
        elapsed = time.time() - start
        
        print(f"  Generated in {elapsed:.3f}s")
        print(f"  Data shape: {X.shape}")
        print(f"  Unique labels: {len(np.unique(y))}")
        
        # Calculate Hopkins statistic (may be slow for large data)
        if config['n_samples'] <= 5000:
            metrics = gen.estimate_clusterability(X)
            print(f"  Hopkins statistic: {metrics['hopkins']:.3f}")
            print(f"  Interpretation: {metrics['interpretation']}")


def demo_hopkins_comparison():
    """Compare Hopkins statistic across different scenarios."""
    print("\n" + "=" * 70)
    print("Hopkins Statistic Comparison")
    print("=" * 70)
    
    gen = ClusterDataGenerator(random_state=42)
    
    scenarios = [
        ("Perfect clusters", 'blobs', 1.0),
        ("Good clusters", 'blobs', 0.7),
        ("Weak clusters", 'blobs', 0.3),
        ("Random data", 'blobs', 0.0),
        ("Moons (non-convex)", 'moons', 0.8),
        ("Noisy clusters", 'noisy', 0.6),
    ]
    
    print(f"\n{'Scenario':<25} {'Mode':<15} {'Target':<10} {'Hopkins':<10} {'Interpretation'}")
    print("-" * 80)
    
    for name, mode, clust_level in scenarios:
        X, y = gen.generate(
            n_samples=500,
            n_features=10,
            n_clusters=5,
            mode=mode,
            clusterability=clust_level
        )
        
        metrics = gen.estimate_clusterability(X)
        
        print(f"{name:<25} {mode:<15} {clust_level:<10.1f} "
              f"{metrics['hopkins']:<10.3f} {metrics['interpretation']}")


if __name__ == "__main__":
    # Run all demonstrations
    demo_modes()
    demo_clusterability()
    demo_large_scale()
    demo_hopkins_comparison()
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)

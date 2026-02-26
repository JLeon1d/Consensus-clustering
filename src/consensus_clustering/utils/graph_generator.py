"""Simple graph generators for testing and benchmarking."""

import numpy as np


def generate_clustered_graph(n_nodes, n_clusters, cluster_density=0.8, noise_density=0.1, random_state=None):
    """
    Generate a simple clustered graph with clear community structure.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes
    n_clusters : int
        Number of clusters/communities
    cluster_density : float, default=0.8
        Probability of edge within cluster (0-1)
    noise_density : float, default=0.1
        Probability of edge between clusters (0-1)
    random_state : int or None
        Random seed for reproducibility
        
    Returns
    -------
    adjacency_matrix : np.ndarray
        Adjacency matrix of shape (n_nodes, n_nodes)
    true_labels : np.ndarray
        True cluster assignments of shape (n_nodes,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    nodes_per_cluster = n_nodes // n_clusters
    true_labels = np.repeat(np.arange(n_clusters), nodes_per_cluster)
    
    if len(true_labels) < n_nodes:
        true_labels = np.concatenate([true_labels, np.full(n_nodes - len(true_labels), n_clusters - 1)])
    
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if true_labels[i] == true_labels[j]:
                if np.random.rand() < cluster_density:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < noise_density:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
    
    return adjacency_matrix, true_labels


def adjacency_to_features(adjacency_matrix, method='degree'):
    """
    Convert adjacency matrix to feature matrix for clustering.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix of shape (n_nodes, n_nodes)
    method : str, default='degree'
        Feature extraction method:
        - 'degree': Use degree vector as features
        - 'adjacency': Use adjacency matrix rows as features
        - 'laplacian': Use Laplacian eigenvectors
        
    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (n_nodes, n_features)
    """
    if method == 'degree':
        degree = adjacency_matrix.sum(axis=1, keepdims=True)
        return degree
    elif method == 'adjacency':
        return adjacency_matrix
    elif method == 'laplacian':
        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
        laplacian = degree_matrix - adjacency_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        n_features = min(10, adjacency_matrix.shape[0] // 2)
        return eigenvectors[:, :n_features]
    else:
        raise ValueError(f"Unknown method: {method}")
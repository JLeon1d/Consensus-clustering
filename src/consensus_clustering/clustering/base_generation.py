"""Generate base clusterings for consensus clustering."""

from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from .kmeans import litekmeans
from ..metrics import clustering_measure


def generate_base_clusterings(
    X: np.ndarray,
    n_clusters: int,
    m_base: int = 10,
    n_init: int = 1,
    random_state: Optional[int] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Generate multiple base clusterings using k-means.

    This function runs k-means m_base times with different random
    initializations to create a diverse set of base clusterings.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    m_base : int, default=10
        Number of base clusterings to generate
    n_init : int, default=1
        Number of k-means initializations per base clustering
    random_state : int or None, default=None
        Random seed for reproducibility
    y_true : np.ndarray or None, default=None
        True labels for evaluation (optional)

    Returns
    -------
    base_data : dict
        Dictionary containing:
        - 'W': Initial consensus matrix (n_samples x n_samples)
        - 'G': List of cluster assignment matrices (m_base matrices)
        - 'F': List of cluster center matrices (m_base matrices)
        - 'labels': List of cluster labels (m_base arrays)
        - 'metrics': Evaluation metrics if y_true provided (optional)

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)
    >>> print(base_data['W'].shape)
    (100, 100)
    >>> print(len(base_data['G']))
    10
    """
    n_samples = X.shape[0]

    W = np.zeros((n_samples, n_samples))
    G_list = []
    F_list = []
    labels_list = []
    metrics_list = [] if y_true is not None else None

    if random_state is not None:
        np.random.seed(random_state)

    for i in range(m_base):
        seed = None if random_state is None else random_state + i
        labels, centers, _ = litekmeans(
            X, n_clusters=n_clusters, max_iter=100, n_init=n_init, random_state=seed
        )

        G = csr_matrix((n_samples, n_clusters))
        G_dense = np.zeros((n_samples, n_clusters))
        for j in range(n_samples):
            G_dense[j, labels[j]] = 1
        G = csr_matrix(G_dense)

        G_list.append(G)
        F_list.append(centers)
        labels_list.append(labels)

        W += G_dense @ G_dense.T

        if y_true is not None:
            metrics = clustering_measure(y_true, labels)
            metrics_list.append(metrics)

    W = W / m_base

    base_data = {
        "W": W,
        "G": G_list,
        "F": F_list,
        "labels": labels_list,
    }

    if y_true is not None:
        base_data["metrics"] = metrics_list

    return base_data


def save_base_clusterings(
    base_data: Dict[str, any], filepath: str, format: str = "pickle"
) -> None:
    """
    Save base clustering results to file.

    Parameters
    ----------
    base_data : dict
        Base clustering data from generate_base_clusterings
    filepath : str
        Output file path
    format : str, default='pickle'
        Output format ('pickle', 'mat', 'npz')

    Examples
    --------
    >>> base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)
    >>> save_base_clusterings(base_data, 'base_clusterings.pkl')
    """
    from ..utils.data_io import save_results

    save_data = base_data.copy()
    save_data["G"] = [G.toarray() if hasattr(G, "toarray") else G for G in base_data["G"]]

    save_results(save_data, filepath, format=format)


def load_base_clusterings(filepath: str, format: str = "auto") -> Dict[str, any]:
    """
    Load base clustering results from file.

    Parameters
    ----------
    filepath : str
        Input file path
    format : str, default='auto'
        Input format ('auto', 'pickle', 'mat', 'npz')

    Returns
    -------
    base_data : dict
        Base clustering data

    Examples
    --------
    >>> base_data = load_base_clusterings('base_clusterings.pkl')
    """
    import pickle
    from pathlib import Path

    filepath = Path(filepath)

    if format == "auto":
        format = filepath.suffix[1:]

    if format in ["pickle", "pkl"]:
        with open(filepath, "rb") as f:
            base_data = pickle.load(f)
    elif format == "mat":
        from scipy.io import loadmat

        data = loadmat(filepath)
        base_data = {
            "W": data["W"],
            "G": [data[f"G_{i}"] for i in range(len([k for k in data.keys() if k.startswith("G_")]))],
            "F": [data[f"F_{i}"] for i in range(len([k for k in data.keys() if k.startswith("F_")]))],
        }
    elif format == "npz":
        data = np.load(filepath, allow_pickle=True)
        base_data = dict(data)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if "G" in base_data:
        base_data["G"] = [
            csr_matrix(G) if not hasattr(G, "toarray") else G for G in base_data["G"]
        ]

    return base_data
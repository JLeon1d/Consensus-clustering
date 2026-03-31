from typing import Optional

import numpy as np
import ray


@ray.remote(num_cpus=1)
def compute_jaccard_block_remote(
    a_block: np.ndarray, 
    b: np.ndarray,
    start_idx: int
) -> np.ndarray:
    """Compute Jaccard similarity for a block of rows."""
    intersection = a_block @ b.T
    
    a_squared = (a_block ** 2).sum(axis=1, keepdims=True)
    b_squared = (b ** 2).sum(axis=1, keepdims=True)
    union = a_squared + b_squared.T - intersection
    
    # Avoid division by zero
    union = np.maximum(union, 1e-10)
    
    similarity = intersection / union
    return similarity


@ray.remote(num_cpus=1)
def compute_d_block_remote(
    start_i: int,
    end_i: int,
    bcs: np.ndarray,
    dis_of_cluster: np.ndarray,
    n_samples: int,
    m_base: int
) -> np.ndarray:
    """Compute dissimilarity matrix block using precomputed cluster dissimilarities."""
    n_block = end_i - start_i
    D_block = np.zeros((n_block, n_samples))
    
    for local_i in range(n_block):
        global_i = start_i + local_i
        for j in range(n_samples):
            d_sum = 0.0
            for m in range(m_base):
                cluster_i = int(bcs[global_i, m] - 1)  # Convert to 0-indexed
                cluster_j = int(bcs[j, m] - 1)
                d_sum += dis_of_cluster[cluster_i, cluster_j]
            D_block[local_i, j] = d_sum / m_base
    
    return D_block


def compute_jaccard_parallel(
    a: np.ndarray, 
    b: Optional[np.ndarray] = None,
    block_size: int = 100
) -> np.ndarray:
    """
    Compute Jaccard similarity matrix in parallel with block processing.
    
    Parameters
    ----------
    a : np.ndarray
        First array of shape (n, d)
    b : np.ndarray or None
        Second array of shape (m, d). If None, compute similarity of a with itself.
    block_size : int
        Number of rows per block
    
    Returns
    -------
    similarity : np.ndarray
        Jaccard similarity matrix of shape (n, m) or (n, n)
    """
    if b is None:
        b = a
    
    n = a.shape[0]
    
    # Put b in Ray object store once to avoid repeated serialization
    b_ref = ray.put(b)
    
    # Process in blocks to reduce task overhead
    futures = []
    for i in range(0, n, block_size):
        end_i = min(i + block_size, n)
        a_block = a[i:end_i]
        futures.append(compute_jaccard_block_remote.remote(a_block, b_ref, i))
    
    blocks = ray.get(futures)
    similarity = np.vstack(blocks)
    
    return similarity


def compute_d_parallel(
    bcs: np.ndarray,
    dis_of_cluster: np.ndarray,
    block_size: int = 50
) -> np.ndarray:
    """
    Compute dissimilarity matrix D in parallel with block processing.
    
    Parameters
    ----------
    bcs : np.ndarray
        Base clusterings of shape (n_samples, m_base)
    dis_of_cluster : np.ndarray
        Dissimilarity matrix between clusters
    block_size : int
        Number of rows per block
    
    Returns
    -------
    d : np.ndarray
        Dissimilarity matrix of shape (n_samples, n_samples)
    """
    n_samples, m_base = bcs.shape
    
    # Put shared data in Ray object store once to avoid repeated serialization
    bcs_ref = ray.put(bcs)
    dis_of_cluster_ref = ray.put(dis_of_cluster)
    
    futures = []
    for i in range(0, n_samples, block_size):
        end_i = min(i + block_size, n_samples)
        futures.append(
            compute_d_block_remote.remote(
                i, end_i, bcs_ref, dis_of_cluster_ref, n_samples, m_base
            )
        )
    
    blocks = ray.get(futures)
    D = np.vstack(blocks)
    
    return D


def compute_nwca_parallel(
    bcs: np.ndarray,
    base_cls_segs: np.ndarray,
    norm_k: np.ndarray,
    block_size: int = 100
) -> np.ndarray:
    """
    Compute NWCA matrix using vectorized operations.
    
    This is more efficient than element-wise computation.
    
    Parameters
    ----------
    bcs : np.ndarray
        Base clusterings
    base_cls_segs : np.ndarray
        Cluster segments
    norm_k : np.ndarray
        Normalization factors
    block_size : int
        Block size for processing
    
    Returns
    -------
    nwca : np.ndarray
        NWCA matrix
    """
    n = bcs.shape[0]
    m = bcs.shape[1]
    
    nwca = np.zeros((n, n))
    
    for t in range(m):
        labels_t = bcs[:, t]
        
        n_clusters_t = int(labels_t.max())
        binary_t = np.zeros((n, n_clusters_t))
        for i in range(n):
            binary_t[i, int(labels_t[i]) - 1] = 1
        
        co_occur_t = binary_t @ binary_t.T
        
        # This is a simplified version - full implementation would use NECI
        nwca += co_occur_t
    
    nwca = nwca / m
    
    max_val = nwca.max()
    if max_val > 0:
        nwca = nwca / max_val
    
    np.fill_diagonal(nwca, 1.0)
    
    return nwca

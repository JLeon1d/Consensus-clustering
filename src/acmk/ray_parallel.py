from typing import List, Tuple

import numpy as np
import ray
from scipy.sparse import issparse


@ray.remote(num_cpus=1)
def compute_GF_batch_remote(G_batch: List[np.ndarray], F_batch: List[np.ndarray]) -> List[np.ndarray]:
    return [G_j @ F_j for G_j, F_j in zip(G_batch, F_batch)]


@ray.remote(num_cpus=1)
def update_F_batch_remote(G_batch: List[np.ndarray], AX: np.ndarray) -> List[np.ndarray]:
    """Update cluster centers F[j] for a batch of base clusterings."""

    F_batch = []
    for G_j in G_batch:
        GG = G_j.T @ G_j
        gg = 1.0 / np.maximum(np.diag(GG), np.finfo(float).eps)
        GX = G_j.T @ AX
        F_batch.append(GX * gg[:, np.newaxis])

    return F_batch


@ray.remote(num_cpus=1)
def compute_H_block_remote(
    G_i_batch: List[np.ndarray], 
    G_all: List[np.ndarray],
    i_start: int
) -> np.ndarray:
    m = len(G_all)
    batch_size = len(G_i_batch)
    H_block = np.zeros((batch_size, m))

    for local_i, G_i in enumerate(G_i_batch):
        for j in range(m):
            tmp = G_i.T @ G_all[j] @ G_all[j].T
            H_block[local_i, j] = np.sum(G_i.T * tmp)

    return H_block


@ray.remote(num_cpus=1)
def compute_f_batch_remote(W: np.ndarray, G_batch: List[np.ndarray]) -> np.ndarray:
    f_batch = np.zeros(len(G_batch))
    for i, G_j in enumerate(G_batch):
        tmp2 = W.T @ G_j
        f_batch[i] = np.sum(tmp2 * G_j)

    return f_batch


def compute_GF_parallel(
    G: List[np.ndarray], 
    F: List[np.ndarray],
    batch_size: int = 4
) -> List[np.ndarray]:
    """
    Compute GF with batching to reduce task overhead.

    Parameters
    ----------
    G : list of np.ndarray
        Cluster assignment matrices
    F : list of np.ndarray
        Cluster center matrices
    batch_size : int
        Number of operations per Ray task

    Returns
    -------
    GF : list of np.ndarray
        List of G[j] @ F[j] products
    """
    m = len(G)
    
    futures = []
    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        G_batch = G[i:end_i]
        F_batch = F[i:end_i]
        futures.append(compute_GF_batch_remote.remote(G_batch, F_batch))

    batched_results = ray.get(futures)
    GF = []
    for batch in batched_results:
        GF.extend(batch)
    
    return GF


def update_F_parallel(
    G: List[np.ndarray], 
    AX: np.ndarray,
    batch_size: int = 4
) -> List[np.ndarray]:
    """
    Update all F matrices in parallel with batching.

    Parameters
    ----------
    G : list of np.ndarray
        Cluster assignment matrices
    AX : np.ndarray
        Transformed data matrix (A^k @ X)
    batch_size : int
        Number of operations per Ray task
    
    Returns
    -------
    F : list of np.ndarray
        Updated cluster center matrices
    """
    m = len(G)

    # Put AX in object store once
    AX_ref = ray.put(AX)

    futures = []
    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        G_batch = G[i:end_i]
        futures.append(update_F_batch_remote.remote(G_batch, AX_ref))

    batched_results = ray.get(futures)
    F = []
    for batch in batched_results:
        F.extend(batch)
    
    return F


def compute_H_matrix_parallel(
    G: List[np.ndarray],
    batch_size: int = 4
) -> np.ndarray:
    """
    Compute H matrix in parallel with block-based approach.
    
    Parameters
    ----------
    G : list of np.ndarray
        Cluster assignment matrices
    batch_size : int
        Number of rows per Ray task
    
    Returns
    -------
    H : np.ndarray
        H matrix of shape (m, m)
    """
    m = len(G)
    
    # Put G list in object store once
    G_ref = ray.put(G)

    futures = []
    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        G_batch = G[i:end_i]
        futures.append(compute_H_block_remote.remote(G_batch, G_ref, i))

    H_blocks = ray.get(futures)
    H = np.vstack(H_blocks)
    H = H + H.T

    return H


def compute_f_vector_parallel(
    W: np.ndarray, 
    G: List[np.ndarray],
    batch_size: int = 4
) -> np.ndarray:
    """
    Compute f vector in parallel with batching.
    
    Parameters
    ----------
    W : np.ndarray
        Consensus matrix
    G : list of np.ndarray
        Cluster assignment matrices
    batch_size : int
        Number of elements per Ray task
    
    Returns
    -------
    f : np.ndarray
        f vector of length m
    """
    m = len(G)
    
    # Put W in object store once
    W_ref = ray.put(W)

    futures = []
    for i in range(0, m, batch_size):
        end_i = min(i + batch_size, m)
        G_batch = G[i:end_i]
        futures.append(compute_f_batch_remote.remote(W_ref, G_batch))

    f_batches = ray.get(futures)
    f = np.concatenate(f_batches)
    f = -2.0 * f
    
    return f


@ray.remote(num_cpus=1)
def optimize_G_single_remote(
    AX: np.ndarray,
    G_i: np.ndarray,
    F_i: np.ndarray,
    W: np.ndarray,
    alpha_i: float,
    lambda_: float,
    max_iter: int,
    i: int,
    G_all: List[np.ndarray],
    alpha: np.ndarray
) -> np.ndarray:
    """
    Optimize a single cluster assignment matrix G[i].
    
    This is the core discrete optimization for one base clustering,
    extracted from optimize_g.py to enable parallelization.
    
    Parameters
    ----------
    AX : np.ndarray
        Transformed data matrix A^k X (n x d)
    G_i : np.ndarray
        Current cluster assignment matrix for base clustering i (n x c)
    F_i : np.ndarray
        Cluster center matrix for base clustering i (c x d)
    W : np.ndarray
        Weighted consensus matrix (n x n)
    alpha_i : float
        Weight for base clustering i
    lambda_ : float
        Regularization parameter
    max_iter : int
        Maximum iterations for greedy optimization
    i : int
        Index of current base clustering
    G_all : list of np.ndarray
        All cluster assignment matrices (for computing GG_i)
    alpha : np.ndarray
        All weights (for computing GG_i)
    
    Returns
    -------
    G_updated : np.ndarray
        Updated cluster assignment matrix
    """
    n, d = AX.shape
    c = F_i.shape[0]
    m = len(G_all)
    
    # Make a copy to avoid modifying input
    G_updated = G_i.copy()
    if issparse(G_updated):
        G_updated = G_updated.toarray()
    
    idx = np.argmax(G_updated, axis=1)
    alphaW = alpha_i * W.T
    
    # Precompute GG_i (contribution from all other base clusterings)
    GG_i = np.zeros((n, n))
    for j in range(m):
        if j != i:
            GG_i += alpha[j] * (G_all[j] @ G_all[j].T)
    
    for iteration in range(max_iter):
        changed = False
        
        for j in range(n):
            AXj = AX[j, :]
            
            obj = np.zeros(c)
            
            # Term 1: Distance to cluster centers
            for k in range(c):
                obj[k] = np.sum((AXj - F_i[k, :]) ** 2)
            
            # Create temporary G without sample j
            G_j = G_updated.copy()
            G_j[j, :] = 0
            
            # Term 2: Consensus term with W
            W_j1 = alphaW[j, :].copy()
            W_j1[j] = 0
            W_j2 = alphaW[:, j].copy()
            W_j2[j] = 0
            
            obj2 = np.sum(G_j * (W_j2[:, np.newaxis] + W_j1[:, np.newaxis]), axis=0)
            obj += -2 * lambda_ * obj2
            
            # Term 3: Interaction with other base clusterings
            GG_i_j = GG_i[:, j].copy()
            GG_i_j[j] = 0
            obj3 = np.sum(G_j * GG_i_j[:, np.newaxis], axis=0)
            obj += 4 * lambda_ * alpha_i * obj3
            
            # Term 4: Self-interaction
            obj4 = np.sum(G_j, axis=0) * 2 * alpha_i * alpha_i * lambda_
            obj += obj4
            
            min_idx = np.argmin(obj)
            
            if idx[j] != min_idx:
                G_updated[j, :] = 0
                G_updated[j, min_idx] = 1
                idx[j] = min_idx
                changed = True
        
        # Early stopping if no changes
        if not changed:
            break
    
    return G_updated


def optimize_G_parallel(
    AX: np.ndarray,
    G: List[np.ndarray],
    F: List[np.ndarray],
    W: np.ndarray,
    alpha: np.ndarray,
    lambda_: float,
    max_iter: int = 5,
) -> List[np.ndarray]:
    """
    Parallelize G optimization across base clusterings using Ray.
    
    Each G[i] can be optimized independently in parallel since they
    don't directly depend on each other during the optimization step.
    
    Parameters
    ----------
    AX : np.ndarray
        Transformed data matrix A^k X (n x d)
    G : list of np.ndarray
        List of cluster assignment matrices (each n x c)
    F : list of np.ndarray
        List of cluster center matrices (each c x d)
    W : np.ndarray
        Weighted consensus matrix (n x n)
    alpha : np.ndarray
        Weight vector (m,)
    lambda_ : float
        Regularization parameter
    max_iter : int, default=5
        Maximum number of iterations for each G_i
    
    Returns
    -------
    G_updated : list of np.ndarray
        Updated cluster assignment matrices
    
    Notes
    -----
    This is a Phase 1 optimization that parallelizes the discrete
    optimization of G matrices. Expected speedup: 1.10x-1.20x for m=10.
    """
    m = len(G)
    
    # Put shared data in object store once to avoid repeated serialization
    AX_ref = ray.put(AX)
    W_ref = ray.put(W)
    G_ref = ray.put(G)
    alpha_ref = ray.put(alpha)
    
    futures = []
    for i in range(m):
        futures.append(optimize_G_single_remote.remote(
            AX_ref,
            G[i],
            F[i],
            W_ref,
            alpha[i],
            lambda_,
            max_iter,
            i,
            G_ref,
            alpha_ref
        ))
    
    G_updated = ray.get(futures)
    
    return G_updated


@ray.remote(num_cpus=1)
def compute_GG_term_remote(G_v: np.ndarray, alpha_v: float) -> np.ndarray:
    """Compute alpha[v] * (G[v] @ G[v].T) for a single base clustering."""
    return alpha_v * (G_v @ G_v.T)


def compute_GG_parallel(G: List[np.ndarray], alpha: np.ndarray) -> np.ndarray:
    """
    Compute GG = sum(alpha[v] * (G[v] @ G[v].T)) in parallel.
    
    This is used in obj_f2 for W matrix optimization. Parallelizing this
    computation can provide speedup when m is large enough.
    
    Parameters
    ----------
    G : list of np.ndarray
        List of cluster assignment matrices (each n x c)
    alpha : np.ndarray
        Weight vector (m,)
    
    Returns
    -------
    GG : np.ndarray
        Weighted sum of G[v] @ G[v].T matrices (n x n)
    
    Notes
    -----
    This is a Phase 2 optimization. Expected speedup depends on m and n.
    For small m (e.g., 10), overhead may dominate. For larger m (e.g., 20+),
    this can provide meaningful speedup.
    """
    m = len(G)
    
    # Put G and alpha in object store once to avoid repeated serialization
    G_ref = ray.put(G)
    alpha_ref = ray.put(alpha)
    
    futures = []
    for v in range(m):
        futures.append(compute_GG_term_remote.remote(G[v], alpha[v]))
    
    GG_terms = ray.get(futures)
    GG = sum(GG_terms)
    
    return GG


def get_optimal_batch_size(m: int, n: int) -> int:
    """
    Determine optimal batch size based on problem size.
    
    Phase 2 optimization: Adaptive batch sizing to balance parallelism
    and overhead. Larger problems benefit from smaller batches (more
    parallelism), while smaller problems need larger batches to reduce
    Ray task overhead.
    
    Parameters
    ----------
    m : int
        Number of base clusterings
    n : int
        Number of samples
    
    Returns
    -------
    batch_size : int
        Optimal batch size for the given problem size
    
    Notes
    -----
    Strategy:
    - For very large n (>5000): Use smaller batches for maximum parallelism
    - For medium n (1000-5000): Use medium batches
    - For small n (<1000): Use larger batches to reduce overhead
    - Always ensure at least 2 batches for some parallelism
    """
    if n > 5000:
        # Large problems: maximize parallelism
        return max(2, m // 8)
    elif n > 2000:
        # Medium-large problems
        return max(2, m // 6)
    elif n > 1000:
        # Medium problems
        return max(2, m // 4)
    else:
        # Small problems: reduce overhead
        return max(2, m // 2)


@ray.remote(num_cpus=1)
def compute_obj1_batch_remote(
    AAX: np.ndarray,
    GF_batch: List[np.ndarray]
) -> float:
    """
    Compute objective contribution from a batch of GF matrices.
    
    Used in obj_f1_d2 for parallel objective computation.
    """
    obj = 0.0
    for GF_i in GF_batch:
        diff = AAX - GF_i
        obj += np.sum(diff ** 2)
    return obj


def compute_obj1_parallel(
    AAX: np.ndarray,
    GF: List[np.ndarray],
    batch_size: int = 4
) -> float:
    """
    Compute sum of ||AAX - GF[i]||^2 in parallel.
    
    Phase 2 optimization for obj_f1_d2. Parallelizes the loop over
    base clusterings when computing the objective function.
    
    Parameters
    ----------
    AAX : np.ndarray
        A^k @ X matrix
    GF : list of np.ndarray
        List of G[i] @ F[i] matrices
    batch_size : int
        Number of GF matrices to process per task
    
    Returns
    -------
    obj : float
        Sum of squared Frobenius norms
    
    Notes
    -----
    Only beneficial for large m (e.g., m >= 20). For typical m=10,
    overhead may dominate.
    """
    m = len(GF)
    
    # Put AAX in object store once to avoid repeated serialization
    AAX_ref = ray.put(AAX)
    
    futures = []
    for i in range(0, m, batch_size):
        batch = GF[i:i+batch_size]
        futures.append(compute_obj1_batch_remote.remote(AAX_ref, batch))
    
    obj_terms = ray.get(futures)
    return sum(obj_terms)

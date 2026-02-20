"""Clustering evaluation metrics: ACC, NMI, and Purity."""

from typing import Dict

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from .hungarian import best_map


def clustering_measure(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (n_samples,)
    y_pred : np.ndarray
        Predicted cluster labels (n_samples,)

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'acc': Clustering accuracy (after optimal label matching)
        - 'nmi': Normalized Mutual Information
        - 'purity': Clustering purity

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([1, 1, 0, 0, 2, 2])
    >>> metrics = clustering_measure(y_true, y_pred)
    >>> print(f"ACC: {metrics['acc']:.3f}, NMI: {metrics['nmi']:.3f}")
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    y_true = _normalize_labels(y_true)
    y_pred = _normalize_labels(y_pred)

    acc = accuracy(y_true, y_pred)
    nmi = normalized_mutual_info(y_true, y_pred)
    purity = clustering_purity(y_true, y_pred)

    return {"acc": acc, "nmi": nmi, "purity": purity}


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize labels to consecutive integers starting from 0."""
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[label] for label in labels])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm for optimal matching.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    acc : float
        Clustering accuracy in [0, 1]
    """
    y_pred_matched = best_map(y_true, y_pred)

    acc = np.mean(y_true == y_pred_matched)

    return float(acc)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    nmi : float
        Normalized Mutual Information in [0, 1]
    """
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="max")
    return float(nmi)


def clustering_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering purity.

    Purity is the fraction of samples that are correctly assigned to
    their majority class within each cluster.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    purity : float
        Clustering purity in [0, 1]
    """
    clusters = np.unique(y_pred)

    correct = 0
    for cluster in clusters:
        cluster_mask = y_pred == cluster
        true_labels_in_cluster = y_true[cluster_mask]

        if len(true_labels_in_cluster) > 0:
            unique, counts = np.unique(true_labels_in_cluster, return_counts=True)
            correct += counts.max()

    purity = correct / len(y_true)
    return float(purity)


def mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mutual Information between two clusterings.

    Parameters
    ----------
    y_true : np.ndarray
        First clustering labels
    y_pred : np.ndarray
        Second clustering labels

    Returns
    -------
    mi : float
        Mutual Information value
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    n = len(y_true)
    n_class = max(y_true.max(), y_pred.max()) + 1

    G = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            G[i, j] = np.sum((y_true == i) & (y_pred == j)) + np.finfo(float).eps

    P1 = G.sum(axis=1) / n
    P2 = G.sum(axis=0) / n

    H1 = -np.sum(P1 * np.log2(P1 + np.finfo(float).eps))
    H2 = -np.sum(P2 * np.log2(P2 + np.finfo(float).eps))

    P12 = G / n
    PPP = P12 / (P2[np.newaxis, :] * P1[:, np.newaxis] + np.finfo(float).eps)
    PPP = np.maximum(PPP, np.finfo(float).eps)

    MI = np.sum(P12 * np.log2(PPP))

    MI_hat = MI / max(H1, H2) if max(H1, H2) > 0 else 0.0

    return float(np.real(MI_hat))
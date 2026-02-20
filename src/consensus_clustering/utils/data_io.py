"""Data loading and saving utilities for various formats."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def load_data(
    filepath: Union[str, Path], format: str = "auto"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load data from various file formats.

    Parameters
    ----------
    filepath : str or Path
        Path to the data file
    format : str, default='auto'
        File format: 'auto', 'mat', 'npy', 'npz', 'csv', 'pickle'
        If 'auto', format is inferred from file extension

    Returns
    -------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features)
    y : np.ndarray or None
        Labels array of shape (n_samples,), or None if not available

    Examples
    --------
    >>> X, y = load_data('data/dataset.mat')
    >>> X, y = load_data('data/dataset.npy', format='npy')
    """
    filepath = Path(filepath)

    if format == "auto":
        format = filepath.suffix[1:]  # Remove the dot

    if format == "mat":
        return _load_mat(filepath)
    elif format == "npy":
        return _load_npy(filepath)
    elif format == "npz":
        return _load_npz(filepath)
    elif format == "csv":
        return _load_csv(filepath)
    elif format in ["pickle", "pkl"]:
        return _load_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_mat(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load MATLAB .mat file."""
    data = loadmat(filepath)
    # Common variable names in MATLAB files
    X = data.get("X", data.get("data", data.get("fea", None)))
    y = data.get("y", data.get("labels", data.get("gnd", None)))

    if X is None:
        # Try to find the largest array
        arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        if arrays:
            X = max(arrays.values(), key=lambda x: x.size)

    if X is None:
        raise ValueError(f"Could not find data matrix in {filepath}")

    # Ensure y is 1D if it exists
    if y is not None:
        y = np.asarray(y).ravel()

    return X, y


def _load_npy(filepath: Path) -> Tuple[np.ndarray, None]:
    """Load NumPy .npy file."""
    X = np.load(filepath)
    return X, None


def _load_npz(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load NumPy .npz file."""
    data = np.load(filepath)
    X = data.get("X", data.get("data", None))
    y = data.get("y", data.get("labels", None))

    if X is None:
        # Use the first array
        X = data[data.files[0]]

    if y is not None:
        y = np.asarray(y).ravel()

    return X, y


def _load_csv(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load CSV file."""
    df = pd.read_csv(filepath)

    # Check if there's a 'label' or 'y' column
    label_cols = [col for col in df.columns if col.lower() in ["label", "labels", "y", "target"]]

    if label_cols:
        y = df[label_cols[0]].values
        X = df.drop(columns=label_cols).values
    else:
        X = df.values
        y = None

    return X, y


def _load_pickle(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
    elif isinstance(data, dict):
        X = data.get("X", data.get("data", None))
        y = data.get("y", data.get("labels", None))
    elif isinstance(data, np.ndarray):
        X = data
        y = None
    else:
        raise ValueError(f"Unexpected data format in pickle file: {type(data)}")

    if y is not None:
        y = np.asarray(y).ravel()

    return X, y


def save_results(
    results: Union[Dict[str, Any], np.ndarray],
    filepath: Union[str, Path],
    format: str = "pickle",
) -> None:
    """
    Save results to file.

    Parameters
    ----------
    results : dict or np.ndarray
        Results to save
    filepath : str or Path
        Output file path
    format : str, default='pickle'
        Output format: 'pickle', 'mat', 'npy', 'npz', 'csv'

    Examples
    --------
    >>> results = {'labels': labels, 'metrics': metrics}
    >>> save_results(results, 'output/results.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format in ["pickle", "pkl"]:
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    elif format == "mat":
        if not isinstance(results, dict):
            results = {"data": results}
        savemat(filepath, results)
    elif format == "npy":
        if isinstance(results, dict):
            results = results.get("data", list(results.values())[0])
        np.save(filepath, results)
    elif format == "npz":
        if not isinstance(results, dict):
            results = {"data": results}
        np.savez(filepath, **results)
    elif format == "csv":
        if isinstance(results, dict):
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
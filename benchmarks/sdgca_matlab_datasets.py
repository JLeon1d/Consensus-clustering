"""Benchmark SDGCA on datasets from the original MATLAB repository.

This script tests SDGCA on small datasets from the SDGCA GitHub repository.
The datasets are accessed via git and loaded using scipy.io.loadmat.
"""

import os
import subprocess
import tempfile
import traceback
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from consensus_clustering import SDGCA


DATASET_PARAMS = {
    "Ecoli": {"lambda_param": 0.09, "eta": 0.75, "theta": 0.65},
    "Aggregation": {"lambda_param": 0.08, "eta": 0.65, "theta": 0.7},
    "MF": {"lambda_param": 0.05, "eta": 0.75, "theta": 0.95},
}


def load_matlab_dataset_from_git(
    dataset_name: str, git_remote: str = "sdgca", git_branch: str = "main"
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a MATLAB dataset from the SDGCA git repository.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'Ecoli', 'Aggregation')
    git_remote : str
        Git remote name
    git_branch : str
        Git branch name

    Returns
    -------
    members : np.ndarray
        Base clusterings matrix (n_samples, n_base_clusterings)
    gt : np.ndarray
        Ground truth labels
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        dataset_path = f"datasets/{dataset_name}.mat"
        git_path = f"{git_remote}/{git_branch}:{dataset_path}"

        result = subprocess.run(
            ["git", "show", git_path],
            capture_output=True,
            check=True,
        )

        with open(tmp_path, "wb") as f:
            f.write(result.stdout)

        data = loadmat(tmp_path)

        members = data.get("members")
        gt = data.get("gt")

        os.unlink(tmp_path)

        if members is None or gt is None:
            print(f"Warning: Could not find 'members' or 'gt' in {dataset_name}.mat")
            return None

        gt = gt.flatten()
        if gt.min() == 0:
            gt = gt + 1

        return members, gt

    except subprocess.CalledProcessError as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return None


def benchmark_on_matlab_dataset(
    dataset_name: str,
    n_base: int = 20,
    n_runs: int = 5,
    max_iter: int = 100,
) -> Dict:
    import traceback
    """Benchmark SDGCA on a MATLAB dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    n_base : int
        Number of base clusterings to use
    n_runs : int
        Number of runs for averaging
    max_iter : int
        Maximum iterations for SDGCA

    Returns
    -------
    results : dict
        Benchmark results
    """
    print(f"\nLoading dataset: {dataset_name}")
    data = load_matlab_dataset_from_git(dataset_name)

    if data is None:
        return {"error": "Failed to load dataset"}

    members, gt = data
    n_samples, pool_size = members.shape
    n_clusters = len(np.unique(gt))

    print(f"  Samples: {n_samples}, Pool size: {pool_size}, Clusters: {n_clusters}")

    params = DATASET_PARAMS.get(dataset_name, {"lambda_param": 0.09, "eta": 0.75, "theta": 0.65})

    ari_scores = []
    nmi_scores = []
    times = []

    for run in range(n_runs):
        np.random.seed(19 + run)
        selected_indices = np.random.choice(pool_size, size=min(n_base, pool_size), replace=False)
        base_clusterings = np.asarray(members)[:, selected_indices]

        sdgca = SDGCA(
            n_clusters=n_clusters,
            lambda_param=params["lambda_param"],
            eta=params["eta"],
            theta=params["theta"],
            max_iter=max_iter,
            verbose=False,
        )

        import time
        start_time = time.time()
        labels = sdgca.fit_predict(base_clusterings)
        elapsed = time.time() - start_time

        if labels.min() == 0:
            labels = labels + 1

        ari = adjusted_rand_score(gt, labels)
        nmi = normalized_mutual_info_score(gt, labels)

        ari_scores.append(ari)
        nmi_scores.append(nmi)
        times.append(elapsed)

        print(f"  Run {run + 1}/{n_runs}: ARI={ari:.3f}, NMI={nmi:.3f}, Time={elapsed:.3f}s")

    results = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_base": min(n_base, pool_size),
        "ari_mean": np.mean(ari_scores),
        "ari_std": np.std(ari_scores),
        "nmi_mean": np.mean(nmi_scores),
        "nmi_std": np.std(nmi_scores),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "params": params,
    }

    print(f"\n  Results for {dataset_name}:")
    print(f"    ARI: {results['ari_mean']:.3f} ± {results['ari_std']:.3f}")
    print(f"    NMI: {results['nmi_mean']:.3f} ± {results['nmi_std']:.3f}")
    print(f"    Time: {results['time_mean']:.3f} ± {results['time_std']:.3f}s")

    return results


def run_matlab_benchmarks(datasets=None, n_base=20, n_runs=5):
    """Run benchmarks on multiple MATLAB datasets.
    
    import traceback

    Parameters
    ----------
    datasets : list, optional
        List of dataset names. If None, uses default small datasets.
    n_base : int
        Number of base clusterings
    n_runs : int
        Number of runs per dataset

    Returns
    -------
    all_results : dict
        Results for all datasets
    """
    if datasets is None:
        datasets = ["Ecoli", "Aggregation", "MF"]

    print("=" * 70)
    print("SDGCA Benchmarks on MATLAB Datasets")
    print("=" * 70)
    print(f"Configuration: {n_base} base clusterings, {n_runs} runs per dataset")

    all_results = {}

    for dataset_name in datasets:
        try:
            results = benchmark_on_matlab_dataset(
                dataset_name, n_base=n_base, n_runs=n_runs
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\nError benchmarking {dataset_name}: {e}")
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Samples':<8} {'Clusters':<9} {'ARI':<12} {'NMI':<12} {'Time (s)':<10}")
    print("-" * 70)

    for dataset_name, results in all_results.items():
        if "error" in results:
            print(f"{dataset_name:<15} Error: {results['error']}")
        else:
            print(
                f"{dataset_name:<15} "
                f"{results['n_samples']:<8} "
                f"{results['n_clusters']:<9} "
                f"{results['ari_mean']:.3f}±{results['ari_std']:.3f}  "
                f"{results['nmi_mean']:.3f}±{results['nmi_std']:.3f}  "
                f"{results['time_mean']:.3f}±{results['time_std']:.3f}"
            )

    return all_results


if __name__ == "__main__":
    results = run_matlab_benchmarks(datasets=["Ecoli", "Aggregation"], n_base=20, n_runs=5)
    print("\nBenchmarks completed!")

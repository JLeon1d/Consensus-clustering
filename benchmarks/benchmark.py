import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.acmk import ACMK
from src.sdgca import SDGCA
from src.clustering.base_generation import generate_base_clusterings
from src.generators.data_generator import ClusterDataGenerator


def run_acmk_benchmark(
    n: int,
    k: int,
    m: int,
    max_iter: int,
    use_ray: bool,
    clusterability: float = 0.7,
    verbose: bool = True
) -> Dict:
    """Run ACMK benchmark with specified parameters."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"ACMK Benchmark: n={n}, k={k}, m={m}, max_iter={max_iter}")
        print(f"Mode: {'Ray Parallel' if use_ray else 'Sequential'}")
        print(f"Clusterability: {clusterability:.2f}")
        print(f"{'='*70}")
    
    if verbose:
        print("Generating synthetic data...")
    generator = ClusterDataGenerator(random_state=42)
    X, y_true = generator.generate(
        n_samples=n,
        n_features=50,
        n_clusters=k,
        mode='blobs',
        clusterability=clusterability
    )
    
    if verbose:
        print("Generating similarity matrix...")
    W = rbf_kernel(X, gamma=0.1)
    W = (W + W.T) / 2
    
    if verbose:
        print("Generating base clusterings...")
    base_start = time.time()
    base_data = generate_base_clusterings(
        X, n_clusters=k, m_base=m, use_ray=use_ray
    )
    base_time = time.time() - base_start
    if verbose:
        print(f"  Base clustering time: {base_time:.2f}s")
    
    G = base_data['G']
    F = base_data['F']
    
    if verbose:
        print("Running ACMK...")
    acmk_start = time.time()
    acmk = ACMK(
        n_clusters=k,
        m_base=m,
        max_iter=max_iter,
        use_ray=use_ray,
        verbose=False
    )
    acmk.fit(X, W, G, F)
    acmk_time = time.time() - acmk_start
    
    total_time = base_time + acmk_time
    
    if verbose:
        print(f"\n  ACMK time: {acmk_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    return {
        'algorithm': 'ACMK',
        'n_samples': n,
        'n_clusters': k,
        'm_base': m,
        'max_iter': max_iter,
        'use_ray': use_ray,
        'base_clustering_time': base_time,
        'algorithm_time': acmk_time,
        'total_time': total_time
    }


def run_sdgca_benchmark(
    n: int,
    k: int,
    m: int,
    max_iter: int,
    use_ray: bool,
    clusterability: float = 0.7,
    verbose: bool = True
) -> Dict:
    """Run SDGCA benchmark with specified parameters."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"SDGCA Benchmark: n={n}, k={k}, m={m}, max_iter={max_iter}")
        print(f"Mode: {'Ray Parallel' if use_ray else 'Sequential'}")
        print(f"Clusterability: {clusterability:.2f}")
        print(f"{'='*70}")
    
    if verbose:
        print("Generating synthetic data...")
    generator = ClusterDataGenerator(random_state=42)
    X, y_true = generator.generate(
        n_samples=n,
        n_features=50,
        n_clusters=k,
        mode='blobs',
        clusterability=clusterability
    )
    
    if verbose:
        print("Generating base clusterings...")
    base_start = time.time()
    base_data = generate_base_clusterings(
        X, n_clusters=k, m_base=m, use_ray=use_ray
    )
    base_time = time.time() - base_start
    if verbose:
        print(f"  Base clustering time: {base_time:.2f}s")
    
    base_clusterings = np.column_stack(base_data['labels'])
    
    if verbose:
        print("Running SDGCA...")
    sdgca_start = time.time()
    sdgca = SDGCA(
        n_clusters=k,
        max_iter=max_iter,
        use_ray=use_ray,
        verbose=verbose
    )
    labels = sdgca.fit_predict(base_clusterings)
    sdgca_time = time.time() - sdgca_start
    
    total_time = base_time + sdgca_time
    
    if verbose:
        print(f"\n  SDGCA time: {sdgca_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    return {
        'algorithm': 'SDGCA',
        'n_samples': n,
        'n_clusters': k,
        'm_base': m,
        'max_iter': max_iter,
        'use_ray': use_ray,
        'base_clustering_time': base_time,
        'algorithm_time': sdgca_time,
        'total_time': total_time
    }




def _write_json_atomic(path: str, payload: Dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)



def _render_results_markdown(results: Dict) -> str:
    lines = ["# Benchmark Results", "", f"**Timestamp:** {results.get('timestamp', 'N/A')}", ""]

    if 'comparison' in results:
        comp = results['comparison']
        lines.extend([
            f"## {comp['algorithm']} Comparison",
            "",
            "| Mode | Total Time | Speedup |",
            "|------|------------|----------|",
            f"| Sequential | {comp['sequential']['total_time']:.2f}s | 1.00x |",
            f"| Ray Parallel | {comp['ray']['total_time']:.2f}s | **{comp['speedup']:.2f}x** |",
        ])
    elif 'comparisons' in results:
        for name, comp in results['comparisons'].items():
            lines.extend([
                f"## {name.upper()} Comparison",
                "",
                "| Mode | Total Time | Speedup |",
                "|------|------------|----------|",
                f"| Sequential | {comp['sequential']['total_time']:.2f}s | 1.00x |",
                f"| Ray Parallel | {comp['ray']['total_time']:.2f}s | **{comp['speedup']:.2f}x** |",
                "",
            ])
    elif 'result' in results:
        res = results['result']
        lines.extend([
            f"## {res['algorithm']} Benchmark",
            "",
            f"- **Dataset:** n={res['n_samples']}, k={res['n_clusters']}, m={res['m_base']}",
            f"- **Mode:** {'Ray Parallel' if res['use_ray'] else 'Sequential'}",
            f"- **Total Time:** {res['total_time']:.2f}s",
            f"  - Base clustering: {res['base_clustering_time']:.2f}s",
            f"  - Algorithm: {res['algorithm_time']:.2f}s",
        ])
    elif 'results' in results:
        lines.append("## Completed Runs")
        lines.append("")
        for item in results['results']:
            res = item['result']
            lines.extend([
                f"### {item['algorithm'].upper()} n={item['n']} run={item['run']} mode={item['mode']}",
                "",
                f"- **Total Time:** {res['total_time']:.2f}s",
                f"- **Base clustering:** {res['base_clustering_time']:.2f}s",
                f"- **Algorithm:** {res['algorithm_time']:.2f}s",
                "",
            ])
        if results.get('failures'):
            lines.append("## Failures")
            lines.append("")
            for failure in results['failures']:
                lines.append(
                    f"- {failure['algorithm'].upper()} n={failure['n']} run={failure['run']} mode={failure['mode']}: {failure['error']}"
                )

    return "\n".join(lines).rstrip() + "\n"



def save_results(results: Dict, output_dir: str = 'results', output_name: Optional[str] = None):
    """Save benchmark results to JSON and markdown files."""
    os.makedirs(output_dir, exist_ok=True)

    if output_name is None:
        output_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    json_path = os.path.join(output_dir, f'{output_name}.json')
    md_path = os.path.join(output_dir, f'{output_name}.md')

    _write_json_atomic(json_path, results)
    with open(md_path, 'w') as f:
        f.write(_render_results_markdown(results))

    print(f"\n✓ Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")



def _append_completed_run(results: Dict, algorithm: str, mode: str, n: int, k: int, result: Dict, run: int = 1):
    results['results'].append({
        'algorithm': algorithm,
        'mode': mode,
        'run': run,
        'n': n,
        'k': k,
        'result': result,
    })



def _save_progress(results: Dict, args, output_name: str):
    results['timestamp'] = datetime.now().isoformat()
    save_results(results, args.output_dir, output_name)



def _run_and_record(
    results: Dict,
    args,
    output_name: str,
    algorithm: str,
    benchmark_func: Callable[..., Dict],
    max_iter: int,
    use_ray: bool,
    verbose: bool,
) -> Dict:
    result = benchmark_func(args.n, args.k, args.m, max_iter, use_ray, args.clusterability, verbose)
    _append_completed_run(
        results,
        algorithm=algorithm,
        mode='ray' if use_ray else 'sequential',
        n=args.n,
        k=args.k,
        result=result,
    )
    _save_progress(results, args, output_name)
    return result



def _build_comparison(algorithm: str, sequential: Dict, ray: Dict) -> Dict:
    return {
        'algorithm': algorithm.upper(),
        'sequential': sequential,
        'ray': ray,
        'speedup': sequential['total_time'] / ray['total_time'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmarking framework for ACMK and SDGCA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ACMK with Ray on 1500 samples
  python benchmarks/benchmark.py --algorithm acmk --n 1500 --k 12 --m 10 --ray
  
  # Run SDGCA sequentially on 2500 samples
  python benchmarks/benchmark.py --algorithm sdgca --n 2500 --k 15 --m 10
  
  # Compare Ray vs Sequential for ACMK
  python benchmarks/benchmark.py --algorithm acmk --n 1000 --k 10 --m 10 --compare
  
  # Run both algorithms with comparison
  python benchmarks/benchmark.py --algorithm both --n 1000 --k 10 --m 10 --compare
        """
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        required=True,
        choices=['acmk', 'sdgca', 'both'],
        help='Algorithm to benchmark (acmk, sdgca, or both)'
    )
    
    parser.add_argument(
        '--n',
        type=int,
        required=True,
        help='Number of samples'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='Number of clusters'
    )
    
    parser.add_argument(
        '--m',
        type=int,
        default=10,
        help='Number of base clusterings (default: 10)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=None,
        help='Maximum iterations (default: 15 for ACMK, 200 for SDGCA)'
    )
    
    parser.add_argument(
        '--ray',
        action='store_true',
        help='Use Ray parallelization'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare Ray vs Sequential (ignores --ray flag)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Base output filename without extension (default: auto-generated timestamp)'
    )
    
    parser.add_argument(
        '--clusterability',
        type=float,
        default=0.7,
        help='Clusterability level (0.0-1.0): 1.0=perfect clusters, 0.5=weak, 0.0=random (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    if not 0.0 <= args.clusterability <= 1.0:
        parser.error("clusterability must be between 0.0 and 1.0")
    
    verbose = not args.quiet
    
    if args.max_iter is None:
        if args.algorithm == 'acmk':
            max_iter = 15
        elif args.algorithm == 'sdgca':
            max_iter = 200
        else:  # both
            max_iter_acmk = 15
            max_iter_sdgca = 200
    else:
        max_iter = args.max_iter
        max_iter_acmk = max_iter
        max_iter_sdgca = args.max_iter
    
    output_name = args.output_name or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = {
        'timestamp': datetime.now().isoformat(),
        'status': 'running',
        'parameters': {
            'algorithm': args.algorithm,
            'n': args.n,
            'k': args.k,
            'm': args.m,
            'max_iter': args.max_iter,
            'use_ray': args.ray,
            'compare': args.compare,
            'clusterability': args.clusterability,
            'output_name': output_name,
        },
        'results': [],
        'failures': [],
    }
    save_results(results, args.output_dir, output_name)

    try:
        if args.compare:
            if args.algorithm == 'both':
                acmk_seq = _run_and_record(results, args, output_name, 'acmk', run_acmk_benchmark, max_iter_acmk, False, verbose)
                acmk_ray = _run_and_record(results, args, output_name, 'acmk', run_acmk_benchmark, max_iter_acmk, True, verbose)
                sdgca_seq = _run_and_record(results, args, output_name, 'sdgca', run_sdgca_benchmark, max_iter_sdgca, False, verbose)
                sdgca_ray = _run_and_record(results, args, output_name, 'sdgca', run_sdgca_benchmark, max_iter_sdgca, True, verbose)
                results['comparisons'] = {
                    'acmk': _build_comparison('acmk', acmk_seq, acmk_ray),
                    'sdgca': _build_comparison('sdgca', sdgca_seq, sdgca_ray),
                }
            else:
                benchmark_func = run_acmk_benchmark if args.algorithm == 'acmk' else run_sdgca_benchmark
                seq_result = _run_and_record(results, args, output_name, args.algorithm, benchmark_func, max_iter, False, verbose)
                ray_result = _run_and_record(results, args, output_name, args.algorithm, benchmark_func, max_iter, True, verbose)
                results['comparison'] = _build_comparison(args.algorithm, seq_result, ray_result)
        else:
            if args.algorithm == 'acmk':
                result = _run_and_record(results, args, output_name, 'acmk', run_acmk_benchmark, max_iter, args.ray, verbose)
                results['result'] = result
            elif args.algorithm == 'sdgca':
                result = _run_and_record(results, args, output_name, 'sdgca', run_sdgca_benchmark, max_iter, args.ray, verbose)
                results['result'] = result
            elif args.algorithm == 'both':
                acmk_result = _run_and_record(results, args, output_name, 'acmk', run_acmk_benchmark, max_iter_acmk, args.ray, verbose)
                sdgca_result = _run_and_record(results, args, output_name, 'sdgca', run_sdgca_benchmark, max_iter_sdgca, args.ray, verbose)
                results['result'] = {
                    'algorithm': 'BOTH',
                    'acmk': acmk_result,
                    'sdgca': sdgca_result,
                }

        results['status'] = 'completed'
        _save_progress(results, args, output_name)
    except Exception as e:
        results['status'] = 'failed'
        results['failures'].append({
            'algorithm': args.algorithm,
            'n': args.n,
            'k': args.k,
            'mode': 'compare' if args.compare else ('ray' if args.ray else 'sequential'),
            'error': repr(e),
            'timestamp': datetime.now().isoformat(),
        })
        _save_progress(results, args, output_name)
        raise


if __name__ == '__main__':
    main()

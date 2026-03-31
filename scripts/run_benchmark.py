#!/usr/bin/env python3
"""Benchmark runner with incremental checkpoint saving.

Saves partial results after every completed sub-run so interrupted jobs still
leave usable results. Resumes automatically if the output file already exists.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.benchmark import run_acmk_benchmark, run_sdgca_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description='Run ACMK or SDGCA benchmarks with checkpointing')
    parser.add_argument('algorithm', choices=['acmk', 'sdgca'], help='Algorithm to benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', required=True, help='Dataset sizes to benchmark')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters (default: 5)')
    parser.add_argument('--k-mode', choices=['fixed', 'sqrt', 'n_div_10'], default='fixed',
                        help='How to choose k per size: fixed, sqrt(n), or n/10 (default: fixed)')
    parser.add_argument('--m', type=int, default=10, help='Number of base clusterings (default: 10)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per size (default: 1)')
    parser.add_argument('--clusterability', type=float, default=0.9, help='Clusterability 0.0-1.0 (default: 0.9)')
    parser.add_argument('--max-iter', type=int, help='Max iterations (default: 15 for ACMK, 200 for SDGCA)')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    return parser.parse_args()


def choose_k(n: int, mode: str, fixed_k: int) -> int:
    if mode == 'fixed':
        return fixed_k
    if mode == 'sqrt':
        return int(round(math.sqrt(n)))
    if mode == 'n_div_10':
        return max(1, int(round(n / 10)))
    raise ValueError(f'Unsupported k mode: {mode}')


def load_or_init(path: str, args) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'status': 'running',
        'config': {
            'algorithm': args.algorithm,
            'sizes': args.sizes,
            'k': args.k,
            'k_mode': args.k_mode,
            'm': args.m,
            'runs': args.runs,
            'clusterability': args.clusterability,
            'max_iter': args.max_iter,
        },
        'results': [],
        'completed': [],
        'failures': [],
    }


def save_checkpoint(path: str, state: dict):
    state['updated_at'] = datetime.now().isoformat()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def already_done(state: dict, n: int, run_idx: int, use_ray: bool) -> bool:
    key = f'{n}:run{run_idx}:{"ray" if use_ray else "seq"}'
    return key in state['completed']


def mark_done(state: dict, n: int, run_idx: int, use_ray: bool):
    key = f'{n}:run{run_idx}:{"ray" if use_ray else "seq"}'
    if key not in state['completed']:
        state['completed'].append(key)


def run_one(state: dict, path: str, algorithm: str, n: int, k: int, run_idx: int, use_ray: bool, args):
    if already_done(state, n, run_idx, use_ray):
        return

    max_iter = args.max_iter if args.max_iter is not None else (15 if algorithm == 'acmk' else 200)
    verbose = not args.quiet
    try:
        if algorithm == 'acmk':
            result = run_acmk_benchmark(n=n, k=k, m=args.m, max_iter=max_iter,
                                        use_ray=use_ray, clusterability=args.clusterability, verbose=verbose)
        else:
            result = run_sdgca_benchmark(n=n, k=k, m=args.m, max_iter=max_iter,
                                         use_ray=use_ray, clusterability=args.clusterability, verbose=verbose)
        state['results'].append({
            'algorithm': algorithm,
            'n': n,
            'k': k,
            'run': run_idx,
            'mode': 'ray' if use_ray else 'sequential',
            'result': result,
        })
        mark_done(state, n, run_idx, use_ray)
        save_checkpoint(path, state)
    except Exception as e:
        state['failures'].append({
            'algorithm': algorithm,
            'n': n,
            'k': k,
            'run': run_idx,
            'mode': 'ray' if use_ray else 'sequential',
            'error': repr(e),
            'timestamp': datetime.now().isoformat(),
        })
        save_checkpoint(path, state)
        raise


def main():
    args = parse_args()
    state = load_or_init(args.output, args)
    save_checkpoint(args.output, state)

    for run_idx in range(1, args.runs + 1):
        for n in args.sizes:
            k = choose_k(n, args.k_mode, args.k)
            run_one(state, args.output, args.algorithm, n, k, run_idx, False, args)
            run_one(state, args.output, args.algorithm, n, k, run_idx, True, args)

    state['status'] = 'completed'
    save_checkpoint(args.output, state)


if __name__ == '__main__':
    main()

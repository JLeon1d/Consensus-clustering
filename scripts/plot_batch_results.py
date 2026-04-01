#!/usr/bin/env python3
"""Generate benchmark plots from a benchmark results JSON file."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def aggregate(results, algorithm: str, metric: str):
    grouped = defaultdict(list)
    for item in results:
        if item['algorithm'] != algorithm:
            continue
        n = item['n']
        mode = item['mode']
        value = item['result'][metric]
        grouped[(n, mode)].append(value)

    ns = sorted({n for n, _ in grouped.keys()})
    seq = []
    ray = []
    for n in ns:
        seq_vals = grouped.get((n, 'sequential'), [])
        ray_vals = grouped.get((n, 'ray'), [])
        seq.append(sum(seq_vals) / len(seq_vals) if seq_vals else None)
        ray.append(sum(ray_vals) / len(ray_vals) if ray_vals else None)
    return ns, seq, ray


def make_plot(ns, seq, ray, title, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(ns, seq, marker='o', linewidth=2, label='Sequential')
    plt.plot(ns, ray, marker='s', linewidth=2, label='Ray Parallel')
    plt.xlabel('n (samples)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _k_label(stem: str) -> str:
    if 'ndiv10' in stem or 'n_div_10' in stem:
        return 'k = n/10'
    if 'sqrtk' in stem or 'sqrt' in stem:
        return 'k = √n'
    return ''


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results from a JSON file')
    parser.add_argument('input', nargs='?', default='benchmarks/batch_results_sqrtk.json',
                        help='Path to benchmark results JSON (default: benchmarks/batch_results_sqrtk.json)')
    parser.add_argument('--output-dir', default='benchmarks/plots',
                        help='Directory to save plots (default: benchmarks/plots)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    k_label = _k_label(stem)
    k_suffix = f', {k_label}' if k_label else ''
    data = load_results(str(input_path))
    results = data['results']

    ns, seq, ray = aggregate(results, 'acmk', 'algorithm_time')
    if any(v is not None for v in seq + ray):
        make_plot(ns, seq, ray, f'ACMK Algorithm Time vs n{k_suffix}', 'Algorithm time (s)',
                  output_dir / f'acmk_algorithm_time_{stem}.png')

    ns, seq, ray = aggregate(results, 'sdgca', 'algorithm_time')
    if any(v is not None for v in seq + ray):
        make_plot(ns, seq, ray, f'SDGCA Algorithm Time vs n{k_suffix}', 'Algorithm time (s)',
                  output_dir / f'sdgca_algorithm_time_{stem}.png')

    ns, seq, ray = aggregate(results, 'base_clustering', 'algorithm_time')
    if any(v is not None for v in seq + ray):
        make_plot(ns, seq, ray, f'Base Clustering Time vs n{k_suffix}', 'Time (s)',
                  output_dir / f'base_clustering_time_{stem}.png')

    print('Generated plots:')
    for p in sorted(output_dir.glob('*.png')):
        print(p)


if __name__ == '__main__':
    main()

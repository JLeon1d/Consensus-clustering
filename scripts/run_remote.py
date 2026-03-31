#!/usr/bin/env python3
"""Unified remote benchmark launcher."""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.remote_utils import RemoteServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run benchmarks on remote server')

    parser.add_argument('algorithm', choices=['acmk', 'sdgca'], help='Algorithm to benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', required=True, help='Problem sizes (n)')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters (default: 5)')
    parser.add_argument('--k-mode', choices=['fixed', 'sqrt', 'n_div_10'], default='fixed',
                        help='How to choose k for each n (default: fixed)')
    parser.add_argument('--m', type=int, default=10, help='Number of base clusterings (default: 10)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per size (default: 1)')
    parser.add_argument('--clusterability', type=float, default=0.9, help='Clusterability (default: 0.9)')
    parser.add_argument('--max-iter', type=int, help='Max iterations (default: 15 for ACMK, 200 for SDGCA)')
    parser.add_argument('--output', type=str, help='Output JSON file name (default: auto-generated)')

    return parser.parse_args()


def main():
    """Upload code and run benchmark remotely."""
    args = parse_args()

    sizes_str = '_'.join(str(s) for s in args.sizes)
    output_name = args.output or f"{args.algorithm}_n{sizes_str}"
    output_file = f"benchmarks/{output_name}.json"
    log_file = f"{output_name}_output.txt"

    print(f"Uploading code to remote server...")

    with RemoteServer() as server:
        server.upload_directory(
            local_dir='.',
            remote_dir='Consensus-clustering',
            skip_dirs={'.git', '__pycache__', 'venv', '.pytest_cache', '.vscode'}
        )

        print(f"\nLaunching {args.algorithm.upper()} benchmark...")

        cmd_parts = [
            'cd Consensus-clustering &&',
            'nohup python3 scripts/run_benchmark.py',
            args.algorithm,
            '--sizes'] + [str(s) for s in args.sizes] + [
            '--k', str(args.k),
            '--k-mode', args.k_mode,
            '--m', str(args.m),
            '--runs', str(args.runs),
            '--clusterability', str(args.clusterability),
            '--output', output_file,
        ]

        if args.max_iter is not None:
            cmd_parts.extend(['--max-iter', str(args.max_iter)])

        cmd_parts.append(f'> {log_file} 2>&1 &')

        cmd = ' '.join(cmd_parts)

        stdout, stderr, exit_code = server.execute_command(cmd)

        if exit_code == 0:
            print("✓ Benchmark launched successfully in background")
            print(f"\nLog file:     {log_file}")
            print(f"Results file: {output_file}")
            print("\nTo check status:")
            print(f"  ssh u3@ithse.ru -p 1022 'tail -f Consensus-clustering/{log_file}'")
            print("\nTo download results:")
            print(f"  scp -P 1022 u3@ithse.ru:Consensus-clustering/{output_file} ./benchmarks/")
        else:
            print(f"✗ Failed to launch benchmark (exit code: {exit_code})")
            if stderr:
                print(f"Error: {stderr}")


if __name__ == '__main__':
    main()

"""Run all benchmarks and generate combined report."""

import sys
import time
from pathlib import Path

# Ensure we can import benchmark scripts
sys.path.insert(0, str(Path(__file__).parent))

from run_acmk_benchmark import main as run_acmk
from run_sdgca_benchmark import main as run_sdgca


def main():
    """Run all benchmarks sequentially."""
    print("="*70)
    print("Running All Benchmarks")
    print("="*70)
    print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Run ACMK benchmarks
    print("\n" + "="*70)
    print("1. Running ACMK Benchmarks")
    print("="*70)
    try:
        run_acmk()
    except Exception as e:
        print(f"✗ ACMK benchmarks failed: {e}")
    
    # Run SDGCA benchmarks
    print("\n" + "="*70)
    print("2. Running SDGCA Benchmarks")
    print("="*70)
    try:
        run_sdgca()
    except Exception as e:
        print(f"✗ SDGCA benchmarks failed: {e}")
    
    total_time = time.time() - start_time
    
    # Generate combined summary
    print("\n" + "="*70)
    print("All Benchmarks Completed!")
    print("="*70)
    print(f"Total execution time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in: {results_dir}")
    print("  - acmk_benchmark_results.md")
    print("  - acmk_benchmark_results.json")
    print("  - sdgca_benchmark_results.md")
    print("  - sdgca_benchmark_results.json")


if __name__ == "__main__":
    main()

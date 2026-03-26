"""Run benchmarks on remote server and download results."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from remote_utils import RemoteServer


def run_benchmarks_remotely():
    """Execute benchmarks on remote server and download results."""
    
    with RemoteServer() as server:
        print("\n" + "="*70)
        print("Running Benchmarks on Remote Server")
        print("="*70 + "\n")
        
        # Navigate to project directory and run benchmarks
        print("1. Starting benchmark execution...")
        
        cmd = " && ".join([
            "cd ~/consensus_clustering_benchmarks",
            "source venv/bin/activate",
            "cd code/benchmarks",
            "python3 run_all_benchmarks.py"
        ])
        
        stdout, stderr, code = server.execute_command(cmd, verbose=True)
        
        if code != 0:
            print(f"✗ Benchmarks failed with exit code {code}")
            if stderr:
                print(f"Error: {stderr}")
            return False
        
        # Download results
        print("\n2. Downloading results...")
        
        local_results_dir = Path(__file__).parent.parent / "benchmarks" / "results"
        local_results_dir.mkdir(exist_ok=True)
        
        result_files = [
            "acmk_benchmark_results.md",
            "acmk_benchmark_results.json",
            "sdgca_benchmark_results.md",
            "sdgca_benchmark_results.json"
        ]
        
        for filename in result_files:
            remote_path = f"~/consensus_clustering_benchmarks/code/benchmarks/results/{filename}"
            local_path = local_results_dir / filename
            
            try:
                server.download_file(remote_path, str(local_path))
            except Exception as e:
                print(f"Warning: Could not download {filename}: {e}")
        
        print("\n" + "="*70)
        print("✓ Remote benchmarks completed!")
        print("="*70)
        print(f"\nResults downloaded to: {local_results_dir}")
        
        return True


if __name__ == "__main__":
    success = run_benchmarks_remotely()
    sys.exit(0 if success else 1)

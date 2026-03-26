"""Benchmarking framework for comparing with and without Ray."""

import time
from typing import Callable, Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict
import json


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    algorithm: str
    n_samples: int
    n_features: int
    n_clusters: int
    n_base: int
    use_ray: bool
    execution_time: float
    accuracy: Optional[float] = None
    ari: Optional[float] = None
    nmi: Optional[float] = None
    memory_mb: Optional[float] = None
    error: Optional[str] = None


class BenchmarkFramework:
    """Framework for running and comparing benchmarks with/without Ray."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize benchmark framework."""
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        algorithm_name: str,
        benchmark_func: Callable,
        config: Dict,
        use_ray: bool,
        n_runs: int = 3
    ) -> List[BenchmarkResult]:
        """
        Run a benchmark multiple times and collect results.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm being benchmarked
        benchmark_func : Callable
            Function that runs the benchmark, should return (time, metrics_dict)
        config : Dict
            Configuration parameters for the benchmark
        use_ray : bool
            Whether to use Ray for parallel processing
        n_runs : int
            Number of times to run the benchmark
            
        Returns
        -------
        results : List[BenchmarkResult]
            List of benchmark results
        """
        run_results = []
        
        for run in range(n_runs):
            print(f"\n{'='*70}")
            print(f"Run {run + 1}/{n_runs} - {algorithm_name} ({'Ray' if use_ray else 'Sequential'})")
            print(f"Config: {config}")
            print(f"{'='*70}")
            
            try:
                print(f"  Starting run {run + 1}/{n_runs}...")
                start_time = time.time()
                metrics = benchmark_func(config, use_ray)
                execution_time = time.time() - start_time
                print(f"  ✓ Run {run + 1}/{n_runs} completed in {execution_time:.2f}s")
                
                result = BenchmarkResult(
                    algorithm=algorithm_name,
                    n_samples=config.get('n_samples', 0),
                    n_features=config.get('n_features', 0),
                    n_clusters=config.get('n_clusters', 0),
                    n_base=config.get('n_base', 0),
                    use_ray=use_ray,
                    execution_time=execution_time,
                    accuracy=metrics.get('accuracy'),
                    ari=metrics.get('ari'),
                    nmi=metrics.get('nmi'),
                    memory_mb=metrics.get('memory_mb')
                )
                
                print(f"✓ Completed in {execution_time:.2f}s")
                if result.ari is not None:
                    print(f"  ARI: {result.ari:.4f}")
                if result.nmi is not None:
                    print(f"  NMI: {result.nmi:.4f}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                result = BenchmarkResult(
                    algorithm=algorithm_name,
                    n_samples=config.get('n_samples', 0),
                    n_features=config.get('n_features', 0),
                    n_clusters=config.get('n_clusters', 0),
                    n_base=config.get('n_base', 0),
                    use_ray=use_ray,
                    execution_time=0.0,
                    error=str(e)
                )
            
            run_results.append(result)
            self.results.append(result)
        
        return run_results
    
    def compare_ray_vs_sequential(
        self,
        algorithm_name: str,
        benchmark_func: Callable,
        config: Dict,
        n_runs: int = 3
    ) -> Dict:
        """
        Compare Ray vs Sequential execution.
        
        Returns
        -------
        comparison : Dict
            Dictionary with comparison statistics
        """
        print(f"\n{'='*70}")
        print(f"Comparing Ray vs Sequential for {algorithm_name}")
        print(f"{'='*70}")
        
        # Run sequential benchmarks
        seq_results = self.run_benchmark(
            algorithm_name, benchmark_func, config, use_ray=False, n_runs=n_runs
        )
        
        # Run Ray benchmarks
        ray_results = self.run_benchmark(
            algorithm_name, benchmark_func, config, use_ray=True, n_runs=n_runs
        )
        
        # Calculate statistics
        seq_times = [r.execution_time for r in seq_results if r.error is None]
        ray_times = [r.execution_time for r in ray_results if r.error is None]
        
        comparison = {
            'algorithm': algorithm_name,
            'config': config,
            'sequential': {
                'mean_time': np.mean(seq_times) if seq_times else None,
                'std_time': np.std(seq_times) if seq_times else None,
                'min_time': np.min(seq_times) if seq_times else None,
                'max_time': np.max(seq_times) if seq_times else None,
            },
            'ray': {
                'mean_time': np.mean(ray_times) if ray_times else None,
                'std_time': np.std(ray_times) if ray_times else None,
                'min_time': np.min(ray_times) if ray_times else None,
                'max_time': np.max(ray_times) if ray_times else None,
            }
        }
        
        if seq_times and ray_times:
            comparison['speedup'] = np.mean(seq_times) / np.mean(ray_times)
        else:
            comparison['speedup'] = None
        
        return comparison
    
    def generate_markdown_report(self, filename: str = "benchmark_results.md"):
        """Generate a markdown report of all benchmark results."""
        if not self.results:
            print("No results to report")
            return
        
        # Group results by algorithm and Ray usage
        grouped = {}
        for result in self.results:
            key = (result.algorithm, result.use_ray)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Generate markdown
        lines = [
            "# Benchmark Results",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        # Create summary table
        algorithms = sorted(set(r.algorithm for r in self.results))
        
        for algo in algorithms:
            lines.append(f"### {algo}")
            lines.append("")
            lines.append("| Configuration | Sequential Time | Ray Time | Speedup | ARI | NMI |")
            lines.append("|--------------|----------------|----------|---------|-----|-----|")
            
            # Group by configuration
            configs = {}
            for result in self.results:
                if result.algorithm != algo:
                    continue
                
                config_key = (result.n_samples, result.n_features, result.n_clusters, result.n_base)
                if config_key not in configs:
                    configs[config_key] = {'seq': [], 'ray': []}
                
                if result.use_ray:
                    configs[config_key]['ray'].append(result)
                else:
                    configs[config_key]['seq'].append(result)
            
            for config_key, results_dict in sorted(configs.items()):
                n_samples, n_features, n_clusters, n_base = config_key
                config_str = f"n={n_samples}, f={n_features}, k={n_clusters}, m={n_base}"
                
                seq_results = results_dict['seq']
                ray_results = results_dict['ray']
                
                seq_time = np.mean([r.execution_time for r in seq_results if r.error is None]) if seq_results else None
                ray_time = np.mean([r.execution_time for r in ray_results if r.error is None]) if ray_results else None
                
                speedup = seq_time / ray_time if (seq_time and ray_time) else None
                
                # Get metrics from sequential runs
                ari_vals = [r.ari for r in seq_results if r.ari is not None]
                nmi_vals = [r.nmi for r in seq_results if r.nmi is not None]
                
                ari_str = f"{np.mean(ari_vals):.4f}" if ari_vals else "N/A"
                nmi_str = f"{np.mean(nmi_vals):.4f}" if nmi_vals else "N/A"
                
                seq_str = f"{seq_time:.2f}s" if seq_time else "N/A"
                ray_str = f"{ray_time:.2f}s" if ray_time else "N/A"
                speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
                
                lines.append(f"| {config_str} | {seq_str} | {ray_str} | {speedup_str} | {ari_str} | {nmi_str} |")
            
            lines.append("")
        
        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")
        
        for algo in algorithms:
            lines.append(f"### {algo} - Detailed")
            lines.append("")
            lines.append("| Run | n_samples | n_clusters | n_base | Ray | Time (s) | ARI | NMI | Error |")
            lines.append("|-----|-----------|------------|--------|-----|----------|-----|-----|-------|")
            
            run_counter = {}
            for result in self.results:
                if result.algorithm != algo:
                    continue
                
                key = (result.n_samples, result.n_clusters, result.n_base, result.use_ray)
                run_counter[key] = run_counter.get(key, 0) + 1
                
                ray_str = "✓" if result.use_ray else "✗"
                time_str = f"{result.execution_time:.2f}" if result.error is None else "N/A"
                ari_str = f"{result.ari:.4f}" if result.ari is not None else "N/A"
                nmi_str = f"{result.nmi:.4f}" if result.nmi is not None else "N/A"
                error_str = result.error if result.error else ""
                
                lines.append(
                    f"| {run_counter[key]} | {result.n_samples} | {result.n_clusters} | "
                    f"{result.n_base} | {ray_str} | {time_str} | {ari_str} | {nmi_str} | {error_str} |"
                )
            
            lines.append("")
        
        # Write to file
        report = "\n".join(lines)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to {filename}")
        return report
    
    def save_json(self, filename: str = "benchmark_results.json"):
        """Save results as JSON."""
        data = [asdict(r) for r in self.results]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ JSON saved to {filename}")

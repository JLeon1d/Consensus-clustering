# Benchmarks

Performance benchmarks for ACMK and SDGCA algorithms comparing Ray vs Sequential execution.

## Running Benchmarks

```bash
# All benchmarks
python3 run_all_benchmarks.py

# Individual algorithms
python3 run_acmk_benchmark.py
python3 run_sdgca_benchmark.py
```

## Results

Results are saved in `results/` directory:
- `*_benchmark_results.md` - Human-readable reports
- `*_benchmark_results.json` - Machine-readable data

## Configurations

**ACMK**: 3 configs (300-800 nodes), ~10 min without Ray
**SDGCA**: 4 configs (200-500 samples), ~10 min without Ray

Each config runs 2 times in both sequential and Ray modes.

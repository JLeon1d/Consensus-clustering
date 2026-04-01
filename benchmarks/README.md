# Benchmarks

## Quick start

```bash
# Benchmark base clustering only (k-means, no consensus step)
python scripts/run_benchmark.py base_clustering --sizes 1000 5000 10000 --k-mode n_div_10 --m 10 --output benchmarks/base_clustering_run.json

# Run ACMK (sequential + Ray) on n=1000, k=10
python scripts/run_benchmark.py acmk --sizes 1000 --k 10 --m 10 --output benchmarks/acmk_n1000.json

# Run SDGCA on multiple sizes with sqrt(n) clusters
python scripts/run_benchmark.py sdgca --sizes 500 1000 2000 --k-mode sqrt --output benchmarks/sdgca_run.json
```

Results are saved incrementally — if the job is interrupted, re-running with the same `--output` file resumes from where it left off.

## Arguments

| Argument | Description | Default |
|---|---|---|
| `algorithm` | `acmk`, `sdgca`, or `base_clustering` | required |
| `--sizes` | One or more dataset sizes | required |
| `--output` | Output JSON file path | required |
| `--k` | Number of clusters | `5` |
| `--k-mode` | `fixed`, `sqrt` (√n), or `n_div_10` (n/10) | `fixed` |
| `--m` | Number of base clusterings | `10` |
| `--runs` | Runs per size (each run does sequential + Ray) | `1` |
| `--max-iter` | Max iterations | `15` (ACMK) / `200` (SDGCA) |
| `--clusterability` | Data clusterability 0.0–1.0 | `0.9` |
| `--quiet` | Suppress verbose output | off |

## Output format

Each completed sub-run is appended to the JSON file immediately. The file contains:
- `config` — the parameters used
- `results` — list of completed runs with timing
- `completed` — list of completed run keys (used for resume)
- `failures` — any errors encountered

## Remote benchmarks

Use `scripts/run_remote.py` to upload the project and run benchmarks on the remote server in the background. See `scripts/run_remote.py --help` for options.

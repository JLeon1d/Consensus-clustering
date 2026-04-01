#!/bin/bash
# Example benchmark runs

# Base clustering only: k = n/10, m=10, sizes 1000 5000 10000
python scripts/run_benchmark.py base_clustering \
    --sizes 1000 5000 10000 \
    --k-mode n_div_10 \
    --m 10 \
    --runs 1 \
    --clusterability 0.9 \
    --output benchmarks/base_clustering_example.json

# ACMK: fixed k=10, sizes 500 and 1000, 2 runs each
python scripts/run_benchmark.py acmk \
    --sizes 500 1000 \
    --k 10 \
    --m 10 \
    --runs 2 \
    --output benchmarks/acmk_example.json

# SDGCA: k = sqrt(n), sizes 500 1000 2000, 1 run each
python scripts/run_benchmark.py sdgca \
    --sizes 500 1000 2000 \
    --k-mode sqrt \
    --m 10 \
    --runs 1 \
    --output benchmarks/sdgca_example.json

# SDGCA: k = n/10 (large cluster counts), sizes 1000 2000 3000
python scripts/run_benchmark.py sdgca \
    --sizes 1000 2000 3000 \
    --k-mode n_div_10 \
    --m 10 \
    --runs 2 \
    --clusterability 0.9 \
    --output benchmarks/sdgca_ndiv10.json

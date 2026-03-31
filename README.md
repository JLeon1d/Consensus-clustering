# Consensus Clustering — ACMK and SDGCA

Python implementation of two consensus clustering algorithms with optional Ray parallelization:

- **ACMK** — Adaptive Consensus Multiple Kernel clustering (ADMM-based)
- **SDGCA** — Similarity and Dissimilarity Guided Co-association clustering

## Installation

Requires Python 3.9–3.12 (Ray is not yet available for 3.13+).

### With Ray (recommended)

```bash
./setup_with_ray.sh
source venv/bin/activate
```

The script finds a compatible Python version (3.9–3.12), creates `venv/`, and installs all dependencies including Ray.

### Without Ray

```bash
pip install -e .
```

### Verify

```bash
python -c "from src import ACMK, SDGCA; print('OK')"
python -c "from src.utils.ray_utils import is_ray_available; print('Ray:', is_ray_available())"
```

## Usage

### ACMK

```python
import numpy as np
from src import ACMK
from src.clustering.base_generation import generate_base_clusterings
from sklearn.metrics.pairwise import rbf_kernel

X = np.random.randn(200, 10)
W = rbf_kernel(X, gamma=0.1)

base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)

acmk = ACMK(n_clusters=5, m_base=10)
acmk.fit(X, W, base_data['G'], base_data['F'])
labels = acmk.predict()
```

### SDGCA

```python
import numpy as np
from src import SDGCA
from src.clustering.base_generation import generate_base_clusterings

X = np.random.randn(200, 10)
base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)
base_clusterings = np.column_stack(base_data['labels'])

sdgca = SDGCA(n_clusters=5)
labels = sdgca.fit_predict(base_clusterings)
```

### Ray parallelization

Pass `use_ray=True` to `generate_base_clusterings` or to the algorithm constructors. Falls back to sequential if Ray is not installed.

```python
base_data = generate_base_clusterings(X, n_clusters=5, m_base=10, use_ray=True)

acmk = ACMK(n_clusters=5, m_base=10, use_ray=True)
sdgca = SDGCA(n_clusters=5, use_ray=True)
```

## Running benchmarks

```bash
python scripts/run_benchmark.py acmk --sizes 500 1000 --k 10 --m 10 --output benchmarks/acmk.json
python scripts/run_benchmark.py sdgca --sizes 500 1000 2000 --k-mode sqrt --output benchmarks/sdgca.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for full argument reference. Use `scripts/run_remote.py` to run benchmarks on a remote server.

## Running tests

```bash
pytest tests/
```

## Requirements

- Python 3.9–3.12
- numpy, scipy, scikit-learn
- ray >= 2.9.0 (optional)

## Algorithms

**ACMK** combines multiple base clusterings through ADMM optimization with adaptive kernel weighting.

**SDGCA** constructs a refined co-association matrix using both similarity and dissimilarity guidance. Based on: *"Similarity and Dissimilarity Guided Co-association Matrix Construction for Ensemble Clustering"*.

## License

MIT

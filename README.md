# Consensus Clustering - ACMK Algorithm

Python3 implementation of the ACMK (Adaptive Consensus Multiple Kernel) clustering algorithm with Ray framework support for parallel processing.

## Installation

### Quick Install

```bash
pip3 install -e .
```

That's it! The `pyproject.toml` automatically installs all dependencies.

### Verify Installation

```bash
python3 -c "from consensus_clustering import ACMK; print('✓ Works!')"
```

## Quick Start

### Basic Usage

```python
from consensus_clustering import ACMK
from consensus_clustering.clustering import generate_base_clusterings
import numpy as np

# Generate synthetic data
X = np.random.randn(100, 10)

# Generate base clusterings
base_data = generate_base_clusterings(X, n_clusters=5, m_base=10)

# Run ACMK
acmk = ACMK(n_clusters=5, m_base=10, lambda_=0.1)
acmk.fit(X, **base_data)

# Get results
labels = acmk.predict(method='spectral')
print(f"Clustering complete! Labels shape: {labels.shape}")
```

### Run Examples

```bash
python3 examples/simple_example.py
python3 examples/demo.py
```

### Run Tests

```bash
# All tests
pytest

# Skip Ray tests (if Ray not installed)
pytest -m "not ray"
```

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- pyyaml >= 6.0
- ray >= 2.9.0 (optional, for parallel processing)

## Project Structure

```
Consensus-clustering/
├── src/consensus_clustering/    # Main package
│   ├── core/                    # ACMK algorithm
│   ├── clustering/              # K-means and base generation
│   ├── optimization/            # L-BFGS-B and objectives
│   ├── metrics/                 # Clustering evaluation
│   └── utils/                   # Data I/O and linear algebra
├── tests/                       # Test suite
├── examples/                    # Usage examples
├── original_code/               # Original MATLAB implementation
└── pyproject.toml              # Package configuration
```

## License

MIT
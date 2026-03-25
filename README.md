# Consensus Clustering - ACMK Algorithm

Python3 implementation of the ACMK (Adaptive Consensus Multiple Kernel) clustering algorithm with Ray framework support for parallel processing.

## Installation

### Quick Setup with Ray (Recommended)

Run the automated setup script that creates a virtual environment with Python 3.11 and installs everything including Ray:

```bash
./setup_with_ray.sh
```

This script will:
- Check for Python 3.11 (required for Ray)
- Create a virtual environment
- Install all dependencies including Ray
- Verify the installation

After setup, activate the environment:
```bash
source venv/bin/activate
```

### Manual Installation Options

#### Basic Installation (without Ray)

```bash
pip install -e .
```

#### Full Installation (with Ray and all features)

Requires Python 3.9-3.12:
```bash
pip install -e ".[all]"
```

#### Install Only Ray Support

```bash
pip install -e ".[ray]"
```

### Verify Installation

```bash
python -c "from consensus_clustering import ACMK; print('✓ Works!')"
```

### Check Ray Status

```bash
python -c "from consensus_clustering.ray_parallel import is_ray_available; print('Ray available:', is_ray_available())"
```

### Python Version Requirements

- **Core package**: Python 3.9+
- **Ray support**: Python 3.9-3.12 (Ray not yet available for Python 3.13+)

If you have Python 3.13+, use the `setup_with_ray.sh` script which will create a venv with Python 3.11.

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

### Parallel Processing with Ray (Optional)

```python
from consensus_clustering.clustering import generate_base_clusterings

# Enable Ray for parallel base clustering generation
base_data = generate_base_clusterings(
    X,
    n_clusters=5,
    m_base=10,
    use_ray=True  # Enable parallel processing
)
```

Ray will automatically parallelize the generation of base clusterings across available CPU cores. If Ray is not installed, the function gracefully falls back to sequential execution.

### Run Examples

```bash
python3 examples/simple_example.py
python3 examples/demo.py
python3 examples/ray_parallel_example.py  # Ray parallel processing demo
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

## Features

### Optional Ray Parallelization

The package includes optional Ray support for parallel processing:

- **Automatic Detection**: Ray is automatically detected if installed
- **Graceful Fallback**: Falls back to sequential execution if Ray unavailable
- **Simple API**: Just add `use_ray=True` parameter
- **Transparent**: Same function signatures with or without Ray

**Benefits:**
- Faster base clustering generation on multi-core systems
- Scales to distributed computing with Ray clusters
- No code changes needed when Ray is unavailable

**Installation:**
```bash
pip install ray>=2.9.0
```

**Usage:**
```python
# Works with or without Ray installed
base_data = generate_base_clusterings(X, n_clusters=5, use_ray=True)
```

## Project Structure

```
Consensus-clustering/
├── src/consensus_clustering/    # Main package
│   ├── core/                    # ACMK algorithm
│   ├── clustering/              # K-means and base generation
│   ├── optimization/            # L-BFGS-B and objectives
│   ├── metrics/                 # Clustering evaluation
│   ├── ray_parallel/            # Ray parallelization utilities
│   └── utils/                   # Data I/O and linear algebra
├── tests/                       # Test suite
├── examples/                    # Usage examples
├── original_code/               # Original MATLAB implementation
└── pyproject.toml              # Package configuration
```

## License

MIT
# Setup Instructions for Consensus Clustering with Ray

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the setup script that handles everything:

```bash
./setup_with_ray.sh
```

This will:
1. Check for Python 3.11 (required for Ray)
2. Create a virtual environment with Python 3.11
3. Install all dependencies including Ray
4. Verify the installation

If Python 3.11 is not found, the script will show installation instructions.

### Option 2: Manual Setup

#### Step 1: Install Python 3.11

**On macOS:**
```bash
brew install python@3.11
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install python3.11 python3.11-venv
```

**On other systems:**
Download from https://www.python.org/downloads/

#### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Package with Ray

```bash
pip install --upgrade pip
pip install -e ".[all]"
```

## Verify Installation

After installation, verify everything works:

```bash
# Activate environment
source venv/bin/activate

# Check Ray is available
python -c "from consensus_clustering.ray_parallel import is_ray_available; print('Ray available:', is_ray_available())"

# Run example
python examples/ray_parallel_example.py

# Run tests
pytest tests/
```

## Usage

### Basic Usage with Ray

```python
from consensus_clustering.clustering import generate_base_clusterings
import numpy as np

# Generate data
X = np.random.randn(500, 50)

# Generate base clusterings with Ray parallelization
base_data = generate_base_clusterings(
    X,
    n_clusters=5,
    m_base=20,
    random_state=42,
    use_ray=True  # Enable Ray
)

print(f"Generated {len(base_data['G'])} base clusterings")
print(f"Consensus matrix shape: {base_data['W'].shape}")
```

### Full ACMK Pipeline

```python
from consensus_clustering import ACMK
from consensus_clustering.clustering import generate_base_clusterings
import numpy as np

# Generate data
X = np.random.randn(500, 50)

# Generate base clusterings with Ray
base_data = generate_base_clusterings(
    X, n_clusters=5, m_base=20, use_ray=True
)

# Run ACMK algorithm
acmk = ACMK(n_clusters=5, m_base=20, lambda_=0.1)
acmk.fit(X, **base_data)

# Get final clustering
labels = acmk.predict(method='spectral')
print(f"Final clustering: {labels}")
```

## Troubleshooting

### Python 3.11 Not Found

If you see "Python 3.11 not found", install it:

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
```

### Ray Installation Fails

If Ray installation fails:

1. **Check Python version**: Ray requires Python 3.9-3.12
   ```bash
   python --version
   ```

2. **Try installing Ray separately**:
   ```bash
   pip install ray>=2.9.0
   ```

3. **Check system compatibility**: Ray may not be available for all platforms

### Import Errors

If you get import errors:

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Reinstall package**:
   ```bash
   pip install -e ".[all]"
   ```

### Performance Issues

If Ray is slower than sequential:

1. **Check CPU count**:
   ```python
   from consensus_clustering.ray_parallel import get_ray_status
   print(get_ray_status())
   ```

2. **Use Ray only for large workloads**: `m_base >= 10`, datasets with 1000+ samples

3. **Adjust Ray resources**:
   ```python
   from consensus_clustering.ray_parallel import init_ray_if_needed
   init_ray_if_needed(use_ray=True, num_cpus=4)
   ```

## Development Setup

For development with testing and linting tools:

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest tests/

# Run tests without Ray
pytest -m "not ray"

# Run only Ray tests
pytest -m ray

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

## Environment Variables

Optional Ray configuration:

```bash
# Set Ray temp directory
export RAY_TMPDIR=/path/to/tmp

# Set Ray logging level
export RAY_LOG_LEVEL=INFO

# Disable Ray dashboard
export RAY_DISABLE_DASHBOARD=1
```

## Uninstall

To remove the environment:

```bash
# Deactivate if active
deactivate

# Remove virtual environment
rm -rf venv
```

## Support

For issues or questions:

1. Check the documentation: `docs/RAY_PARALLEL.md`
2. Run the example: `python examples/ray_parallel_example.py`
3. Check Ray status: `python -c "from consensus_clustering.ray_parallel import get_ray_status; print(get_ray_status())"`

## Summary

- ✅ Use `./setup_with_ray.sh` for automated setup
- ✅ Requires Python 3.11 for Ray support
- ✅ Package works without Ray (sequential mode)it
- ✅ Ray provides 2-4x speedup on multi-core systems
- ✅ Complete documentation in `docs/RAY_PARALLEL.md`
# Setup Instructions

## Quick Start

### Automated Setup

Run the setup script:

```bash
./setup_with_ray.sh
```

This will:
1. Check for Python 3.11 (required for Ray)
2. Create a virtual environment
3. Install all dependencies including Ray
4. Verify the installation

### Manual Setup

#### Install Python 3.11

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3.11 python3.11-venv
```

#### Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Package

```bash
pip install --upgrade pip
pip install -e ".[all]"
```

## Verify Installation

```bash
source venv/bin/activate

# Check Ray availability
python -c "from consensus_clustering.ray_parallel import is_ray_available; print('Ray available:', is_ray_available())"

# Run example
python examples/ray_parallel_example.py

# Run tests
pytest tests/
```

## Usage

### Basic Usage

```python
from consensus_clustering.clustering import generate_base_clusterings
import numpy as np

X = np.random.randn(500, 50)

base_data = generate_base_clusterings(
    X,
    n_clusters=5,
    m_base=20,
    random_state=42,
    use_ray=True
)

print(f"Generated {len(base_data['G'])} base clusterings")
```

### Full Pipeline

```python
from consensus_clustering import ACMK
from consensus_clustering.clustering import generate_base_clusterings
import numpy as np

X = np.random.randn(500, 50)

base_data = generate_base_clusterings(
    X, n_clusters=5, m_base=20, use_ray=True
)

acmk = ACMK(n_clusters=5, m_base=20, lambda_=0.1)
acmk.fit(X, **base_data)

labels = acmk.predict(method='spectral')
```

## Troubleshooting

### Python 3.11 Not Found

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

1. Check Python version (Ray requires 3.9-3.12):
   ```bash
   python --version
   ```

2. Install Ray separately:
   ```bash
   pip install ray>=2.9.0
   ```

### Import Errors

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Reinstall package:
   ```bash
   pip install -e ".[all]"
   ```

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest tests/

# Run tests without Ray
pytest -m "not ray"

# Run only Ray tests
pytest -m ray
```

## Uninstall

```bash
deactivate
rm -rf venv
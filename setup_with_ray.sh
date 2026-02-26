#!/bin/bash

set -e

echo "========================================================================"
echo "Consensus Clustering - Setup with Ray Support"
echo "========================================================================"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

find_python() {
    for version in python3.12 python3.11 python3.10 python3.9; do
        if command_exists $version; then
            echo $version
            return 0
        fi
    done
    return 1
}

echo ""
echo "Checking for compatible Python version (3.9-3.12)..."

PYTHON_CMD=$(find_python) || true

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "No compatible Python version found (need 3.9-3.12)"
    echo ""
    echo "Installing Python 3.11..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command_exists brew; then
            echo "Using Homebrew to install Python 3.11..."
            brew install python@3.11
            PYTHON_CMD="python3.11"
        else
            echo "Homebrew not found. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "Using apt to install Python 3.11..."
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
            PYTHON_CMD="python3.11"
        elif command_exists yum; then
            echo "Using yum to install Python 3.11..."
            sudo yum install -y python3.11 python3.11-devel
            PYTHON_CMD="python3.11"
        else
            echo "Could not detect package manager. Please install Python 3.11 manually:"
            echo "  https://www.python.org/downloads/"
            exit 1
        fi
    else
        echo "Unsupported OS. Please install Python 3.11 manually:"
        echo "  https://www.python.org/downloads/"
        exit 1
    fi
    
    if ! command_exists $PYTHON_CMD; then
        echo "Python installation failed"
        exit 1
    fi
fi

echo "Found: $PYTHON_CMD"
$PYTHON_CMD --version

if [ -d "venv" ]; then
    echo ""
    echo "Removing old virtual environment..."
    rm -rf venv
fi

echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null

echo ""
echo "Installing consensus-clustering with Ray support..."
pip install -e ".[all]"

echo ""
echo "Verifying installation..."
python -c "from consensus_clustering import ACMK; print('Package imported successfully')"
python -c "from consensus_clustering.ray_parallel import is_ray_available, get_ray_status; status = get_ray_status(); print(f'Ray available: {status[\"available\"]}')"

echo ""
echo "========================================================================"
echo "Setup Complete"
echo "========================================================================"
echo ""
echo "Python version: $($PYTHON_CMD --version)"
echo "Virtual environment: venv/"
echo ""
echo "To activate:"
echo "  source venv/bin/activate"
echo ""
echo "To run example:"
echo "  python examples/ray_parallel_example.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "========================================================================"
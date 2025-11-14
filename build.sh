#!/bin/bash
# Build script for mtrx package

set -e

echo "Building mtrx package with pure C backend..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Build and install
echo "Building C extensions..."
pip install -e .

# Verify
echo ""
echo "Verifying installation..."
python -c "from c_matmul import C_BACKEND_AVAILABLE; print(f'✓ C backend available: {C_BACKEND_AVAILABLE}')"

echo ""
echo "✓ Build complete! Run 'python test/benchmark.py' to test performance."

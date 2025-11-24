#!/bin/bash
# Build script for mtrx package

set -e

echo "Building qkmx package with pure C backend..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Build and install
echo "Building C extensions..."
python3 -m pip install -e .

echo ""
echo "✓ Build complete!"

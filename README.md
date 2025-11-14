# QMX (Quick Matrix)

[![PyPI version](https://badge.fury.io/py/qmx.svg)](https://pypi.org/project/qmx/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A minimal tensor computation library built from scratch in Python with **pure C** acceleration.

**Standalone pip-installable package** - Use in any project!

✅ No C++ dependencies  
✅ No pybind11 required  
✅ Pure C with Python C API  
✅ 274-417x faster than pure Python

## Quick Start

```bash
pip install qmx
```

Or from source:
```bash
cd qmx
pip install -e .
```

## Features

- **Tensor operations**: Create, manipulate, and compute with multi-dimensional tensors
- **Matrix operations**: Matrix multiplication, transpose, reshape
- **Neural network primitives**: Linear layers, softmax, masking
- **Pure C acceleration**: 274-417x faster than pure Python
- **Simple API**: Clean `mx` namespace

## Installation

### From PyPI

```bash
pip install qmx
```

### From Source

```bash
git clone https://github.com/micwill755/qmx.git
cd qmx
pip install -e .
```

### Verify Installation

```bash
python -c "import mx; print('✓ QMX installed successfully')"
python -c "import c_matmul; print('✓ C backend available')"
```

**Requirements:**
- Python >= 3.7
- C11 compatible compiler (gcc, clang, or MSVC)
- Python development headers

**Troubleshooting:**
- macOS: `xcode-select --install`
- Linux: `sudo apt-get install python3-dev gcc`
- Windows: Install Visual Studio Build Tools

## Usage

### Basic Operations

```python
import mx

# Create tensors
x = mx.randn((2, 3, 4))
y = mx.ones((4, 5))
z = mx.zeros((3, 3))

# Matrix multiplication
result = mx.mat_mul(x, y)

# Softmax
probs = mx.softmax(x)
```

### Neural Network Layers

```python
from mx import Tensor, Linear

# Linear layer
layer = Linear(d_in=128, d_out=256)
x = Tensor((32, 128))  # batch_size=32, features=128
output = layer(x)
```

### C Backend (Direct Access)

```python
import c_matmul

# Flatten matrices for C backend
A = [1.0] * (128 * 256)  # 128x256 matrix
B = [1.0] * (256 * 512)  # 256x512 matrix

# Fast C matmul
C = c_matmul.matmul_f32(A, B, M=128, N=512, K=256)

# Also available: matmul_f64, matmul_i32, matmul_i8
```

## API Reference

### Tensor Creation
- `mx.Tensor(shape, v=0)` - Create tensor with value
- `mx.randn(shape)` - Random tensor (Gaussian distribution)
- `mx.zeros(shape)` - Tensor filled with zeros
- `mx.ones(shape)` - Tensor filled with ones

### Operations
- `mx.mat_mul(m1, m2)` - Matrix multiplication
- `mx.softmax(tensor)` - Softmax activation
- `mx.mask(m, window)` - Apply causal masking
- `mx.reshape(m, shape)` - Reshape tensor

### Modules
- `mx.Linear(d_in, d_out)` - Linear transformation layer
- `mx.Module` - Base module class

## Performance

Benchmark results (Pure C vs Python):

| Matrix Size | Python | C | Speedup |
|-------------|--------|---|----------|
| 64x64       | 90.72 ms  | 0.22 ms | **417x** |
| 128x128     | 575.99 ms | 1.82 ms | **316x** |
| 256x256     | 4550.17 ms | 16.58 ms | **274x** |

Run benchmark:
```bash
python test/benchmark.py
```

## Why QMX?

- **Fast**: Pure C implementation with 274-417x speedup
- **Simple**: Clean API with `mx` namespace
- **Lightweight**: No heavy dependencies (no PyTorch, TensorFlow, etc.)
- **Educational**: Learn how tensors work under the hood
- **Extensible**: Easy to add custom operations

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License

## Links

- **PyPI**: https://pypi.org/project/qmx/
- **GitHub**: https://github.com/yourusername/qmx
- **Documentation**: Coming soon
- **Issues**: https://github.com/yourusername/qmx/issues

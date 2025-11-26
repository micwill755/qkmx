# qkmx (Quick Matrix)

A minimal tensor computation library with **pure C** and **CUDA** acceleration. PyTorch-like API, zero dependencies beyond NumPy.

## Features

- **CPU & CUDA support** - Same API, automatic dispatch
- **Pure C backend** - No C++ required, compiles with gcc
- **CUDA Driver API** - JIT-compiled PTX kernels, no nvcc needed
- **PyTorch-like API** - Familiar tensor operations

## Quick Start

```python
import mx

# Create tensors (CPU by default)
a = mx.randn([2, 3])
b = mx.randn([3, 4])

# Matrix multiplication
c = a @ b

# Move to GPU
a_gpu = mx.randn([2, 3], device='cuda')
b_gpu = mx.randn([3, 4], device='cuda')
c_gpu = a_gpu @ b_gpu  # Runs on GPU

# Or move existing tensors
c_gpu = a.cuda() @ b.cuda()

# Check device
print(a_gpu.is_cuda)  # True
```

## Supported Operations

| Operation | CPU | CUDA |
|-----------|-----|------|
| Matmul (`@`) | ✅ | ✅ |
| GELU | ✅ | ✅ |
| Softmax | ✅ | ✅ |
| LayerNorm | ✅ | ✅ |
| RMSNorm | ✅ | ❌ |
| Element-wise (+, -, *) | ✅ | ❌ |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Python: mx.zeros([2,3], device='cuda')                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  src/mx/tensor.py      - Python Tensor class wrapper        │
│  src/mx/functional.py  - zeros(), randn(), gelu(), etc.     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  csrc/tensor_bindings.c - Python ↔ C bridge (CPython API)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  csrc/tensor.c         - Dispatch layer (routes to backend) │
│  csrc/dispatch.c       - Function pointer tables            │
└──────────┬─────────────────────────────────┬────────────────┘
           │                                 │
┌──────────▼──────────┐           ┌──────────▼──────────┐
│  csrc/cpu/          │           │  csrc/cuda_driver/  │
│  ├─ tensor_cpu.c    │           │  ├─ tensor_cuda.c   │
│  ├─ matmul_cpu.c    │           │  ├─ matmul_driver.c │
│  ├─ activation_cpu.c│           │  ├─ activation_driver│
│  └─ norm_cpu.c      │           │  └─ norm_driver.c   │
└─────────────────────┘           └─────────────────────┘
```

## File Structure

```
qkmx/
├── src/mx/                 # Python package
│   ├── tensor.py           # Tensor class wrapper
│   ├── functional.py       # High-level functions
│   └── __init__.py
├── csrc/                   # C source code
│   ├── tensor.h            # Tensor struct definition
│   ├── tensor.c            # Dispatch layer
│   ├── tensor_bindings.c   # Python-C bindings
│   ├── dispatch.c/h        # CPU/CUDA routing
│   ├── cpu/                # CPU implementations
│   │   ├── tensor_cpu.c    # Memory allocation
│   │   ├── matmul_cpu.c
│   │   ├── activation_cpu.c
│   │   └── norm_cpu.c
│   └── cuda_driver/        # CUDA Driver API implementations
│       ├── cuda_init.c     # Context initialization
│       ├── tensor_cuda.c   # GPU memory management
│       ├── matmul_driver.c # PTX matmul kernel
│       ├── activation_driver.c  # PTX gelu/softmax
│       └── norm_driver.c   # PTX layer_norm
├── setup.py                # Build configuration
└── test/                   # Test files
```

## Installation

```bash
# CPU only
pip install -e .

# With CUDA (requires CUDA toolkit)
pip install -e .  # Auto-detects CUDA
```

## How It Works

1. **Tensor struct** (`tensor.h`) holds data pointer, shape, dtype, and device
2. **Dispatch** (`dispatch.c`) routes operations based on `tensor->device`
3. **CPU ops** are simple C loops
4. **CUDA ops** use Driver API with embedded PTX assembly
5. **Python bindings** expose everything via CPython C API

## Why CUDA Driver API?

- **Pure C** - No nvcc or C++ compiler needed
- **JIT compilation** - PTX compiles at runtime for any GPU
- **Single binary** - Works across GPU architectures
- **No extern "C" issues** - Clean C linkage

## License

MIT License

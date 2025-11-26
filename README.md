# qkmx (Quick Matrix)

A minimal tensor computation library with **pure C** and **CUDA** acceleration. PyTorch-like API, zero dependencies beyond NumPy.

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

## License

MIT License

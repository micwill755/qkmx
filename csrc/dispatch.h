#ifndef DISPATCH_H
#define DISPATCH_H

#include "tensor.h"

// Function pointer types for each operation
typedef Tensor* (*gelu_fn)(const Tensor*);
typedef Tensor* (*matmul_fn)(const Tensor*, const Tensor*);
typedef Tensor* (*softmax_fn)(const Tensor*, int);
typedef Tensor* (*layer_norm_fn)(const Tensor*, const Tensor*, const Tensor*, float);

// Dispatch table structure
typedef struct {
    gelu_fn gelu;
    matmul_fn matmul;
    softmax_fn softmax;
    layer_norm_fn layer_norm;
    // Add more operations here
} DispatchTable;

/*
┌─────────────────────────────────────────────────────────┐
│ Python: y = tc.gelu(x)                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ tensor.c: tensor_gelu(t)                                │
│   ├─ get_dispatch_table(t->device)                      │
│   └─ dispatch->gelu(t)  ← Call function pointer         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│ DEVICE_CPU   │          │ DEVICE_CUDA  │
└──────┬───────┘          └──────┬───────┘
       │                         │
       ▼                         ▼
┌──────────────┐          ┌──────────────┐
│ gelu_cpu()   │          │ gelu_cuda()  │
│ (CPU loop)   │          │ (GPU kernel) │
└──────────────┘          └──────────────┘

*/

// Get dispatch table for device
const DispatchTable* get_dispatch_table(DeviceType device);

// CPU implementations (forward declarations)
Tensor* gelu_cpu(const Tensor* t);
Tensor* matmul_cpu(const Tensor* a, const Tensor* b);
Tensor* softmax_cpu(const Tensor* t, int dim);
Tensor* layer_norm_cpu(const Tensor* x, const Tensor* gamma, const Tensor* beta, float eps);

// CUDA implementations (forward declarations)
#ifdef USE_CUDA
Tensor* gelu_cuda(const Tensor* t);
Tensor* matmul_cuda(const Tensor* a, const Tensor* b);
Tensor* softmax_cuda(const Tensor* t, int dim);
Tensor* layer_norm_cuda(const Tensor* x, const Tensor* gamma, const Tensor* beta, float eps);
#endif

#endif

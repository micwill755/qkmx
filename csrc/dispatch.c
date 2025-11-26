#include "dispatch.h"
#include <stdio.h>

// CPU dispatch table
static const DispatchTable cpu_dispatch = {
    .gelu = gelu_cpu,
    .matmul = matmul_cpu,
    .softmax = softmax_cpu,
    .layer_norm = layer_norm_cpu,
};

#ifdef USE_CUDA
// CUDA dispatch table - use Driver API if available
#ifdef USE_CUDA_DRIVER_API
static const DispatchTable cuda_dispatch = {
    .gelu = gelu_cuda_driver,
    .matmul = matmul_cuda_driver,
    .softmax = softmax_cuda_driver,
    .layer_norm = layer_norm_cuda_driver,
};
#else
static const DispatchTable cuda_dispatch = {
    .gelu = gelu_cuda,
    .matmul = matmul_cuda,
    .softmax = softmax_cuda,
    .layer_norm = layer_norm_cuda,
};
#endif
#endif

// Get dispatch table based on device
const DispatchTable* get_dispatch_table(DeviceType device) {
    switch(device) {
        case DEVICE_CPU:
            return &cpu_dispatch;
        case DEVICE_CUDA:
#ifdef USE_CUDA
            return &cuda_dispatch;
#else
            fprintf(stderr, "CUDA not available, falling back to CPU\n");
            return &cpu_dispatch;
#endif
        default:
            return &cpu_dispatch;
    }
}

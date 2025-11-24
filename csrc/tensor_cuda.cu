#include "dispatch.h"
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

Tensor* gelu_cuda(const Tensor* t) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;
    out->device = DEVICE_CUDA;
    
    size_t bytes = t->size * sizeof(float);
    
    // Allocate GPU memory for output
    CUDA_CHECK(cudaMalloc(&out->data, bytes));
    
    // Assume input is already on GPU (t->data is GPU pointer)
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>((float*)t->data, (float*)out->data, t->size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return out;
}

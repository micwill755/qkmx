// tensor_cuda.c - CUDA tensor memory operations (Driver API)
#include "tensor_cuda.h"
#include "cuda_init.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef USE_CUDA

Tensor* tensor_create_cuda(int* shape, int ndim, DType dtype) {
    cuda_driver_init();  // Ensure CUDA is initialized
    
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->owns_data = 1;
    t->strides = NULL;
    t->device = DEVICE_CUDA;
    
    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape) {
        free(t);
        return NULL;
    }
    
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    size_t bytes;
    switch(dtype) {
        case DTYPE_FLOAT32: bytes = t->size * 4; break;
        case DTYPE_FLOAT16: bytes = t->size * 2; break;
        case DTYPE_INT8:    bytes = t->size; break;
        case DTYPE_INT4:    bytes = (t->size + 1) / 2; break;
        case DTYPE_UINT8:   bytes = t->size; break;
        default:            bytes = t->size * 4; break;
    }
    
    // Allocate GPU memory
    CUdeviceptr d_ptr;
    CUresult err = cuMemAlloc(&d_ptr, bytes);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "CUDA alloc failed: %s\n", errStr);
        free(t->shape);
        free(t);
        return NULL;
    }
    
    // Zero initialize
    cuMemsetD8(d_ptr, 0, bytes);
    t->data = (void*)d_ptr;

    return t;
}

void tensor_free_cuda(Tensor* t) {
    if (!t) return;
    if (t->owns_data && t->data) {
        cuMemFree((CUdeviceptr)t->data);
    }
    free(t->shape);
    free(t->strides);
    free(t);
}

void tensor_fill_cuda(Tensor* t, float value) {
    // For now, fill on CPU and copy
    // TODO: implement CUDA kernel for fill
    size_t bytes = t->size * sizeof(float);
    float* cpu_data = malloc(bytes);
    for (size_t i = 0; i < t->size; i++) {
        cpu_data[i] = value;
    }
    cuMemcpyHtoD((CUdeviceptr)t->data, cpu_data, bytes);
    free(cpu_data);
}

void tensor_copy_cuda(Tensor* dst, const Tensor* src) {
    if (dst->size != src->size) return;
    size_t bytes = src->size * sizeof(float);
    cuMemcpyDtoD((CUdeviceptr)dst->data, (CUdeviceptr)src->data, bytes);
}

Tensor* tensor_cpu_to_cuda(const Tensor* t) {
    if (t->device != DEVICE_CPU) {
        fprintf(stderr, "tensor_cpu_to_cuda: source must be CPU tensor\n");
        return NULL;
    }
    
    Tensor* gpu_t = tensor_create_cuda(t->shape, t->ndim, t->dtype);
    if (!gpu_t) return NULL;
    
    size_t bytes = t->size * sizeof(float);
    cuMemcpyHtoD((CUdeviceptr)gpu_t->data, t->data, bytes);
    
    return gpu_t;
}

Tensor* tensor_cuda_to_cpu(const Tensor* t) {
    if (t->device != DEVICE_CUDA) {
        fprintf(stderr, "tensor_cuda_to_cpu: source must be CUDA tensor\n");
        return NULL;
    }
    
    // Import CPU create function
    extern Tensor* tensor_create_cpu(int* shape, int ndim, DType dtype);
    
    Tensor* cpu_t = tensor_create_cpu(t->shape, t->ndim, t->dtype);
    if (!cpu_t) return NULL;
    
    size_t bytes = t->size * sizeof(float);
    cuMemcpyDtoH(cpu_t->data, (CUdeviceptr)t->data, bytes);
    
    return cpu_t;
}

#endif // USE_CUDA

// tensor_cuda.h - CUDA tensor memory operations
#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include "../tensor.h"

#ifdef USE_CUDA

// CUDA memory allocation
Tensor* tensor_create_cuda(int* shape, int ndim, DType dtype);
void tensor_free_cuda(Tensor* t);

// CUDA data operations
void tensor_fill_cuda(Tensor* t, float value);
void tensor_copy_cuda(Tensor* dst, const Tensor* src);

// Host <-> Device transfers
Tensor* tensor_cpu_to_cuda(const Tensor* t);
Tensor* tensor_cuda_to_cpu(const Tensor* t);

#endif // USE_CUDA

#endif // TENSOR_CUDA_H

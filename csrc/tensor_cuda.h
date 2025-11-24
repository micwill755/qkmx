#ifndef TENSOR_CUDA_H
#define TENSOR_CUDA_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// CUDA versions of operations
Tensor* tensor_matmul_cuda(const Tensor* a, const Tensor* b);
Tensor* tensor_op_cuda(const Tensor* a, const Tensor* b, TensorOperation op);
Tensor* tensor_gelu_cuda(const Tensor* t);
Tensor* tensor_softmax_cuda(const Tensor* t, int dim);
Tensor* tensor_layer_norm_cuda(const Tensor* x, const Tensor* gamma, 
                                const Tensor* beta, float eps);

// Memory transfer
Tensor* tensor_to_cuda(const Tensor* t);
Tensor* tensor_to_cpu(const Tensor* t);

#ifdef __cplusplus
}
#endif

#endif
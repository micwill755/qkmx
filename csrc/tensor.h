// tensor.h - Header for tensor C implementation
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_INT8,
    DTYPE_INT4,
    DTYPE_UINT8
} DType;

typedef enum {
    TENSOR_ADD,
    TENSOR_MUL,
    TENSOR_DIV
} TensorOperation;

typedef struct {
    void *data;
    int *shape;
    int *strides;    // bytes to skip per dimension
    int owns_data;   // 1 if owns memory, 0 if view
    int ndim;
    size_t size;
    DType dtype;
} Tensor;

// Core functions
Tensor* tensor_create(int* shape, int ndim, DType dtype);
Tensor* tensor_randn(int* shape, int ndim, DType dtype);
Tensor* tensor_rand(int* shape, int ndim, DType dtype);
void tensor_free(Tensor *t);
Tensor* tensor_op(const Tensor* a, const Tensor* b, TensorOperation op);
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void* tensor_get_data(const Tensor* t);
Tensor* tensor_reshape(const Tensor* t, int* new_shape, int new_ndim);
Tensor* tensor_transpose(const Tensor* t, int dim0, int dim1);
Tensor* tensor_layer_norm(const Tensor* x, const Tensor* gamma, 
                          const Tensor* beta, float eps);
Tensor* tensor_rms_norm(const Tensor* x, const Tensor* weight, float eps);
Tensor* tensor_triu(int size, int diagonal);
Tensor* tensor_masked_fill(const Tensor* t, const Tensor* mask, float value);
Tensor* tensor_rand(int* shape, int ndim, DType dtype);
Tensor* tensor_zeros(int* shape, int ndim, DType dtype);
Tensor* tensor_ones(int* shape, int ndim, DType dtype);
void tensor_free(Tensor *t);
Tensor* tensor_get_index(Tensor* t, int index);
Tensor* tensor_advanced_index(Tensor* t, Tensor* indices);
int tensor_set_index(Tensor* t, int index, Tensor* value);
int tensor_set_scalar(Tensor* t, int index, float value);
float tensor_get_scalar(Tensor* t, int* indices, int num_indices);
Tensor* tensor_scalar_op(const Tensor* t, float scalar, TensorOperation op);
Tensor* tensor_softmax(const Tensor* t, int dim);
Tensor* tensor_gelu(const Tensor* t);

#endif // TENSOR_H

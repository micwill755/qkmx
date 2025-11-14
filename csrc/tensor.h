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
    TENSOR_MUL,
    TENSOR_ADD
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
Tensor* tensor_reshape();
Tensor* tensor_transpose();


#endif // TENSOR_H

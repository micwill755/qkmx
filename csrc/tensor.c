// tensor.c - Core tensor implementation
#include "tensor.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

Tensor* tensor_create(int* shape, int ndim, DType dtype) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->shape = (int*)malloc(ndim * sizeof(int));
    t->size = 1;  // Initialize to 1
    
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    // Calculate bytes once
    size_t bytes;
    switch(dtype) {
        case DTYPE_FLOAT32: bytes = t->size * 4; break;
        case DTYPE_FLOAT16: bytes = t->size * 2; break;
        case DTYPE_INT8:    bytes = t->size * 1; break;
        case DTYPE_INT4:    bytes = (t->size + 1) / 2; break;
        case DTYPE_UINT8:   bytes = t->size * 1; break;
        default:            bytes = t->size * 4; break;
    }
    t->data = calloc(bytes, 1);  // Single allocation

    return t;
}

// Create tensor with random normal values
Tensor* tensor_randn(int* shape, int ndim, DType dtype) {
    Tensor* t = tensor_create(shape, ndim, dtype);
    
    // Seed random (call once at program start)
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    // Fill with random values
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = randn_float();
        }
    }
    
    return t;
}

// Uniform random [0, 1)
Tensor* tensor_rand(int* shape, int ndim, DType dtype) {
    Tensor* t = tensor_create(shape, ndim, dtype);
    
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = (float)rand() / RAND_MAX;
        }
    }
    
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        free(t->data);
        free(t->shape);
        free(t);
    }
}

#define TENSOR_OP_IMPL(type, a, b, out, size, op) \
    do { \
        type* a_data = (type*)(a)->data; \
        type* b_data = (type*)(b)->data; \
        type* out_data = (type*)(out)->data; \
        switch(op) { \
            case TENSOR_MUL: \
                for (size_t i = 0; i < (size); i++) { \
                    out_data[i] = a_data[i] * b_data[i]; \
                } \
                break; \
            case TENSOR_ADD: \
                for (size_t i = 0; i < (size); i++) { \
                    out_data[i] = a_data[i] + b_data[i]; \
                } \
                break; \
        } \
    } while(0)

Tensor* tensor_op(const Tensor* a, const Tensor* b, TensorOperation op) {
    if (a->size != b->size || a->dtype != b->dtype) return NULL;
    
    // TODO: add reference counting or inplace operations 
    Tensor* out = tensor_create(a->shape, a->ndim, a->dtype);

    switch (a->dtype) {
        case DTYPE_FLOAT32: TENSOR_OP_IMPL(float, a, b, out, a->size, op); break;
        case DTYPE_INT8:    TENSOR_OP_IMPL(int8_t, a, b, out, a->size, op); break;
        case DTYPE_UINT8:   TENSOR_OP_IMPL(uint8_t, a, b, out, a->size, op); break;
        default: break;
    }

    return out;
}

#define MATMUL_IMPL(type, a_data, b_data, out_data, M, K, N) \
    do { \
        for (int i = 0; i < (M); i++) { \
            for (int j = 0; j < (N); j++) { \
                type sum = 0; \
                for (int k = 0; k < (K); k++) { \
                    sum += (a_data)[i * (K) + k] * (b_data)[k * (N) + j]; \
                } \
                (out_data)[i * (N) + j] = sum; \
            } \
        } \
    } while(0)

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    if (a->dtype != b->dtype) return NULL;
    
    int M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int out_shape[2] = {M, N};
    Tensor* out = tensor_create(out_shape, 2, a->dtype);
    
    switch (a->dtype) {
        case DTYPE_FLOAT32:
            MATMUL_IMPL(float, (float*)a->data, (float*)b->data, 
                       (float*)out->data, M, K, N);
            break;
        case DTYPE_INT8:
            MATMUL_IMPL(int8_t, (int8_t*)a->data, (int8_t*)b->data,
                       (int8_t*)out->data, M, K, N);
            break;
        default:
            tensor_free(out);
            return NULL;
    }
    
    return out;
}

void* tensor_get_data(const Tensor* t) {
    return t->data;
}
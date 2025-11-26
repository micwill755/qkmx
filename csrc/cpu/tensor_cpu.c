// tensor_cpu.c - CPU tensor memory operations
#include "tensor_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tensor* tensor_create_cpu(int* shape, int ndim, DType dtype) {
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) return NULL;
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->owns_data = 1;
    t->strides = NULL;
    t->device = DEVICE_CPU;
    
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
    
    t->data = calloc(bytes, 1);
    if (!t->data) {
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

void tensor_free_cpu(Tensor* t) {
    if (!t) return;
    if (t->owns_data && t->data) {
        free(t->data);
    }
    free(t->shape);
    free(t->strides);
    free(t);
}

void tensor_fill_cpu(Tensor* t, float value) {
    if (t->dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = value;
        }
    }
}

void tensor_copy_cpu(Tensor* dst, const Tensor* src) {
    if (dst->size != src->size) return;
    size_t bytes = src->size * sizeof(float);  // Assumes FLOAT32
    memcpy(dst->data, src->data, bytes);
}

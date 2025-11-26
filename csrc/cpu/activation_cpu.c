#include "../tensor.h"
#include <math.h>

Tensor* gelu_cpu(const Tensor* t) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, DEVICE_CPU);
    
    if (t->dtype == DTYPE_FLOAT32) {
        float* in_data = (float*)t->data;
        float* out_data = (float*)out->data;
        
        for (size_t i = 0; i < t->size; i++) {
            float x = in_data[i];
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            out_data[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
    
    return out;
}

Tensor* softmax_cpu(const Tensor* t, int dim) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, DEVICE_CPU);
    
    if (t->dtype == DTYPE_FLOAT32) {
        float* in_data = (float*)t->data;
        float* out_data = (float*)out->data;
        
        if (dim < 0) dim += t->ndim;
        
        int dim_size = t->shape[dim];
        int outer_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= t->shape[i];
        int inner_size = 1;
        for (int i = dim + 1; i < t->ndim; i++) inner_size *= t->shape[i];
        
        for (int outer = 0; outer < outer_size; outer++) {
            for (int inner = 0; inner < inner_size; inner++) {
                int base_idx = outer * dim_size * inner_size + inner;
                
                float max_val = in_data[base_idx];
                for (int d = 1; d < dim_size; d++) {
                    int idx = base_idx + d * inner_size;
                    if (in_data[idx] > max_val) max_val = in_data[idx];
                }
                
                float sum = 0.0f;
                for (int d = 0; d < dim_size; d++) {
                    int idx = base_idx + d * inner_size;
                    out_data[idx] = expf(in_data[idx] - max_val);
                    sum += out_data[idx];
                }
                
                for (int d = 0; d < dim_size; d++) {
                    int idx = base_idx + d * inner_size;
                    out_data[idx] /= sum;
                }
            }
        }
    }
    
    return out;
}
